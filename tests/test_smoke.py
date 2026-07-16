import numpy as np
import pytest
import torch
from pathlib import Path

from ghxtox import plm_embed
from ghxtox.chemical_sites import CHEMICAL_SITE_TYPE_TO_INDEX, parse_chemical_sites
from ghxtox.data import collate_peptides
from ghxtox.esmfold_cache import parse_esmfold_pdb
from ghxtox.fasta import read_fasta
from ghxtox.features import RESIDUE_FEATURE_DIM, louvain_community_features, residue_feature_matrix
from ghxtox.geometry_features import STRUCTURE_FEATURE_DIM, structure_feature_matrix
from ghxtox.evaluate import _write_predictions, build_arg_parser as build_evaluate_arg_parser
from ghxtox.esmfold_predict import build_arg_parser as build_esmfold_predict_arg_parser
from ghxtox.models import GHXToxModel
from ghxtox.atom_graph import ATOM_FEATURE_DIM, EDGE_FEATURE_DIM, EXTENDED_ATOM_FEATURE_DIM, peptide_atom_graph
from ghxtox.train import SmoothedBCEWithLogitsLoss, _apply_coordinate_noise
from ghxtox.models.layers import PLDDTAwareEGNNLayer
from ghxtox.models.chemical_sites import ChemicalSiteInteractionBranch
from ghxtox.bootstrap_ci import bootstrap_confidence_intervals
from ghxtox.chemical_site_report import summarize as summarize_chemical_site
from ghxtox.sequence_similarity import alignment_identity, audit_similarity
from ghxtox.similarity_subset_eval import evaluate_similarity_subsets
from ghxtox.prepare_cdhit import prepare_cdhit_fasta
from ghxtox.cdhit_clusters import extract_strict_subset, parse_clusters
from ghxtox.subset_processed import subset_processed
from ghxtox.report_figures import generate_evaluation_figure
from ghxtox.folds import assign_reference_groups, load_fold_indices
from ghxtox.oof import _best_threshold
from ghxtox.utils import resolve_inference_checkpoint
from ghxtox.native_cdhit_audit import (
    combine_bucket_outputs,
    prepare_inputs,
    prepare_length_buckets,
    summarize_audit,
)


def test_feature_shape():
    features = residue_feature_matrix("ACDE")
    assert features.shape == (4, RESIDUE_FEATURE_DIM)


def test_esm2_complete_cache_does_not_load_model(tmp_path, monkeypatch):
    input_path = tmp_path / "input.pt"
    output_path = tmp_path / "output.pt"
    cache_dir = tmp_path / "cache"
    record = {"sample_id": "cached_sample", "sequence": "AC"}
    torch.save({"records": [record]}, input_path)
    embedding = torch.randn(2, 1280)
    plm_embed._save_cached_embedding(
        cache_dir,
        record["sample_id"],
        record["sequence"],
        embedding,
        "esm2_t33_650M_UR50D",
    )

    def fail_if_loaded(*_args, **_kwargs):
        raise AssertionError("ESM2 model should not load when every embedding is cached")

    monkeypatch.setattr(plm_embed, "_load_esm2", fail_if_loaded)
    stats = plm_embed.attach_esm2_embeddings(
        input_path,
        output_path,
        "esm2_t33_650M_UR50D",
        "cuda",
        batch_size=2,
        cache_dir=cache_dir,
    )
    payload = torch.load(output_path, map_location="cpu", weights_only=False)
    assert stats == {"attached": 1, "generated": 0, "cached": 1, "dim": 1280}
    assert torch.equal(payload["records"][0]["plm_features"], embedding)


def test_louvain_features_are_deterministic():
    first = louvain_community_features("ACDEACDE")
    second = louvain_community_features("ACDEACDE")
    assert torch.allclose(first, second)
    assert first.shape == (8, 3)


def test_structure_features_shape():
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 1.0, 0.0],
            [11.4, 1.0, 1.0],
        ]
    )
    plddt = torch.tensor([0.9, 0.8, 0.7, 0.6])
    features = structure_feature_matrix(coords, plddt)
    assert features.shape == (4, STRUCTURE_FEATURE_DIM)


def test_parse_esmfold_pdb_extracts_backbone(tmp_path):
    pdb_text = """PARENT N/A
ATOM      1  N   ALA A   1      11.706  -2.171  22.367  1.00 35.69           N
ATOM      2  CA  ALA A   1      11.634  -2.684  21.001  1.00 40.14           C
ATOM      3  C   ALA A   1      10.191  -2.982  20.603  1.00 36.76           C
ATOM      4  O   ALA A   1       9.532  -3.823  21.220  1.00 37.87           O
ATOM      5  CB  ALA A   1      12.493  -3.937  20.857  1.00 36.20           C
ATOM      6  N   GLY A   2       9.435  -2.253  19.982  1.00 42.84           N
ATOM      7  CA  GLY A   2       7.982  -2.080  20.049  1.00 40.68           C
ATOM      8  C   GLY A   2       7.569  -0.620  20.215  1.00 42.99           C
ATOM      9  O   GLY A   2       6.557  -0.193  19.652  1.00 40.78           O
END
"""
    pdb_path = tmp_path / "sample.pdb"
    pdb_path.write_text(pdb_text, encoding="utf-8")
    parsed = parse_esmfold_pdb(pdb_path)
    assert parsed.sequence == "AG"
    assert parsed.coords.shape == (2, 3)
    assert parsed.backbone_coords.shape == (2, 5, 3)
    assert parsed.backbone_mask.shape == (2, 5)
    assert parsed.backbone_mask[0].tolist() == [True, True, True, True, True]
    assert parsed.backbone_mask[1].tolist() == [True, True, True, True, False]
    assert np.allclose(parsed.backbone_coords[0, 0] - parsed.backbone_coords[0, 1], [0.072, 0.513, 1.366], atol=1e-3)


def test_parse_esmfold_pdb_extracts_functional_group_center(tmp_path):
    pdb_text = """ATOM      1  N   LYS A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  LYS A   1       1.000   0.000   0.000  1.00 80.00           C
ATOM      3  C   LYS A   1       2.000   0.000   0.000  1.00 80.00           C
ATOM      4  O   LYS A   1       3.000   0.000   0.000  1.00 80.00           O
ATOM      5  NZ  LYS A   1       1.000   4.000   0.000  1.00 80.00           N
END
"""
    pdb_path = tmp_path / "lys.pdb"
    pdb_path.write_text(pdb_text, encoding="utf-8")
    parsed = parse_esmfold_pdb(pdb_path)
    assert parsed.functional_group_mask.tolist() == [True]
    assert np.allclose(parsed.functional_group_coords[0], [0.0, 4.0, 0.0], atol=1e-6)


def test_parse_chemical_sites_separates_tyr_ring_and_hydroxyl(tmp_path):
    atoms = {
        "N": (0.0, 0.0, 0.0),
        "CA": (1.0, 0.0, 0.0),
        "C": (2.0, 0.0, 0.0),
        "O": (3.0, 0.0, 0.0),
        "CB": (1.0, 1.0, 0.0),
        "CG": (1.0, 2.0, 0.0),
        "CD1": (0.0, 3.0, 0.0),
        "CD2": (2.0, 3.0, 0.0),
        "CE1": (0.0, 4.0, 0.0),
        "CE2": (2.0, 4.0, 0.0),
        "CZ": (1.0, 5.0, 0.0),
        "OH": (1.0, 6.0, 0.0),
    }
    lines = []
    for serial, (name, (x, y, z)) in enumerate(atoms.items(), start=1):
        element = name[0]
        lines.append(
            f"ATOM  {serial:5d} {name:^4s} TYR A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00          {element:>2s}"
        )
    pdb_path = tmp_path / "tyr.pdb"
    pdb_path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    sites = parse_chemical_sites(pdb_path)
    assert sites.mask.tolist() == [[True, True]]
    aromatic = CHEMICAL_SITE_TYPE_TO_INDEX["aromatic"]
    donor = CHEMICAL_SITE_TYPE_TO_INDEX["donor"]
    assert sites.types[0, 0, aromatic] == 1.0
    assert sites.types[0, 1, donor] == 1.0
    assert not np.allclose(sites.coords[0, 0], sites.coords[0, 1])
    assert sites.orientation_mask.tolist() == [[True, True]]


def test_collate_pads_legacy_residue_features():
    batch = [
        {
            "sample_id": "p_1",
            "sequence": "AC",
            "aa_ids": torch.tensor([2, 3]),
            "group_ids": torch.tensor([0, 1]),
            "residue_features": torch.zeros(2, 15),
            "coords": torch.zeros(2, 3),
            "plddt": torch.ones(2),
            "label": 1,
        }
    ]
    collated = collate_peptides(batch)
    assert collated["residue_features"].shape[-1] == RESIDUE_FEATURE_DIM
    assert collated["structure_features"].shape[-1] == STRUCTURE_FEATURE_DIM
    assert collated["backbone_coords"].shape == (1, 2, 5, 3)
    assert collated["backbone_mask"].shape == (1, 2, 5)
    assert collated["backbone_mask"][0, :, 1].all()
    assert collated["functional_group_coords"].shape == (1, 2, 3)
    assert not collated["functional_group_mask"].any()
    assert collated["chemical_site_coords"].shape == (1, 2, 2, 3)
    assert collated["chemical_site_types"].shape == (1, 2, 2, 8)
    assert not collated["chemical_site_mask"].any()


def test_chemical_site_branch_is_zero_initialized_and_rotation_invariant():
    branch = ChemicalSiteInteractionBranch(
        hidden_dim=16,
        site_hidden_dim=8,
        num_layers=1,
        raw_rbf_bins=8,
        normalized_rbf_bins=4,
        dropout=0.0,
    ).eval()
    site_coords = torch.tensor(
        [[[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], [[2.0, 1.0, 0.0], [2.0, 2.0, 0.0]], [[4.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]]
    )
    site_types = torch.zeros(1, 3, 2, 8)
    site_types[0, 0, 0, CHEMICAL_SITE_TYPE_TO_INDEX["positive"]] = 1.0
    site_types[0, 1, 0, CHEMICAL_SITE_TYPE_TO_INDEX["negative"]] = 1.0
    site_types[0, 1, 1, CHEMICAL_SITE_TYPE_TO_INDEX["aromatic"]] = 1.0
    site_types[0, 2, 0, CHEMICAL_SITE_TYPE_TO_INDEX["hydrophobic"]] = 1.0
    site_mask = torch.tensor([[[True, False], [True, True], [True, False]]])
    orientations = torch.zeros_like(site_coords)
    orientations[..., 1] = 1.0
    orientation_mask = site_mask.clone()
    residue_coords = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    residue_mask = torch.ones(1, 3, dtype=torch.bool)
    plddt = torch.full((1, 3), 0.9)
    args = (
        site_coords,
        site_types,
        orientations,
        orientation_mask,
        site_mask,
        residue_coords,
        residue_mask,
        plddt,
    )
    initial, diagnostics = branch(*args)
    assert torch.count_nonzero(initial) == 0
    assert diagnostics["chemical_edge_count"].item() > 0
    with torch.no_grad():
        torch.nn.init.normal_(branch.residual_projection.weight, std=0.05)
    output, _ = branch(*args)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated_args = (
        site_coords @ rotation.T,
        site_types,
        orientations @ rotation.T,
        orientation_mask,
        site_mask,
        residue_coords @ rotation.T,
        residue_mask,
        plddt,
    )
    rotated, _ = branch(*rotated_args)
    assert torch.allclose(output, rotated, atol=1e-5)


def test_chemical_site_ablation_switches_construct_and_forward():
    branch = ChemicalSiteInteractionBranch(
        hidden_dim=8,
        site_hidden_dim=8,
        num_layers=1,
        raw_rbf_bins=4,
        normalized_rbf_bins=4,
        use_normalized_rbf=False,
        use_orientation=False,
        use_interaction_types=False,
        use_plddt=False,
        use_hydrophobic_sites=False,
        max_site_slots=1,
        dropout=0.0,
    )
    site_coords = torch.zeros(1, 2, 2, 3)
    site_coords[0, 1, 0, 0] = 3.0
    site_types = torch.zeros(1, 2, 2, 8)
    site_types[..., CHEMICAL_SITE_TYPE_TO_INDEX["positive"]] = 1.0
    site_types[0, 0, 0, CHEMICAL_SITE_TYPE_TO_INDEX["hydrophobic"]] = 1.0
    site_mask = torch.ones(1, 2, 2, dtype=torch.bool)
    output, diagnostics = branch(
        site_coords,
        site_types,
        torch.zeros_like(site_coords),
        torch.zeros(1, 2, 2, dtype=torch.bool),
        site_mask,
        torch.zeros(1, 2, 3),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2),
    )
    assert output.shape == (1, 2, 8)
    assert diagnostics["chemical_site_count"].item() == 2


def test_chemical_site_report_summarizes_and_audits_frozen_base(tmp_path):
    control_dir = tmp_path / "control_fold0"
    candidate_dir = tmp_path / "candidate_fold0"
    control_dir.mkdir()
    candidate_dir.mkdir()
    metrics = {
        "balanced_accuracy": 0.80,
        "f1": 0.75,
        "mcc": 0.60,
        "auroc": 0.90,
        "auprc": 0.85,
    }
    torch.save(
        {
            "model_state": {"shared.weight": torch.tensor([1.0])},
            "epoch": 2,
            "val_metrics": metrics,
        },
        control_dir / "best_model.pt",
    )
    candidate_metrics = dict(metrics)
    candidate_metrics["mcc"] = 0.62
    candidate_metrics["auprc"] = 0.87
    torch.save(
        {
            "model_state": {
                "shared.weight": torch.tensor([1.0]),
                "chemical_site_branch.weight": torch.tensor([2.0]),
            },
            "epoch": 3,
            "val_metrics": candidate_metrics,
        },
        candidate_dir / "best_model.pt",
    )
    summary = summarize_chemical_site(
        str(tmp_path / "control_fold{fold}" / "best_model.pt"),
        str(tmp_path / "candidate_fold{fold}" / "best_model.pt"),
        [0],
    )
    assert summary["unweighted_fold_mean"]["candidate_minus_control"]["mcc"] == pytest.approx(0.02)
    assert summary["unweighted_fold_mean"]["candidate_minus_control"]["auprc"] == pytest.approx(0.02)
    assert summary["frozen_base_audit"]["all_nonchemical_tensors_unchanged"]


def test_sequence_only_collate_skips_unused_structure_tensors():
    record = {
        "sample_id": "sequence_only",
        "sequence": "AC",
        "aa_ids": torch.tensor([2, 3]),
        "group_ids": torch.tensor([0, 1]),
        "residue_features": torch.zeros(2, RESIDUE_FEATURE_DIM),
        "plm_features": torch.zeros(2, 6),
        "coords": torch.zeros(2, 3),
        "plddt": torch.ones(2),
        "label": 1,
    }
    collated = collate_peptides([record], include_structure=False, include_atom=False)
    assert "structure_features" not in collated
    assert "coords" not in collated
    assert collated["plm_features"].shape == (1, 2, 6)


def test_model_constructs():
    config = {
        "model": {
            "aa_embedding_dim": 8,
            "group_embedding_dim": 4,
            "hidden_dim": 32,
            "num_sequence_layers": 1,
            "num_egnn_layers": 1,
            "num_attention_heads": 4,
            "dropout": 0.1,
            "graph_mode": "hybrid",
            "spatial_top_k": 4,
            "rbf_bins": 8,
            "rbf_max_distance": 12.0,
        }
    }
    assert GHXToxModel(config) is not None


def test_model_forwards_with_plm_features():
    batch = [
        {
            "sample_id": "p_1",
            "sequence": "AC",
            "aa_ids": torch.tensor([2, 3]),
            "group_ids": torch.tensor([0, 1]),
            "residue_features": torch.zeros(2, RESIDUE_FEATURE_DIM),
            "plm_features": torch.zeros(2, 6),
            "coords": torch.zeros(2, 3),
            "plddt": torch.ones(2),
            "label": 1,
        },
        {
            "sample_id": "p_2",
            "sequence": "ACD",
            "aa_ids": torch.tensor([2, 3, 4]),
            "group_ids": torch.tensor([0, 1, 2]),
            "residue_features": torch.zeros(3, RESIDUE_FEATURE_DIM),
            "plm_features": torch.ones(3, 6),
            "coords": torch.zeros(3, 3),
            "plddt": torch.ones(3),
            "label": 0,
        },
    ]
    collated = collate_peptides(batch)
    assert collated["plm_features"].shape == (2, 3, 6)
    assert collated["structure_features"].shape == (2, 3, STRUCTURE_FEATURE_DIM)

    config = {
        "model": {
            "modality": "sequence_only",
            "structure_feature_dim": STRUCTURE_FEATURE_DIM,
            "plm_embedding_dim": 6,
            "aa_embedding_dim": 8,
            "group_embedding_dim": 4,
            "hidden_dim": 32,
            "num_sequence_layers": 1,
            "num_egnn_layers": 1,
            "num_attention_heads": 4,
            "dropout": 0.1,
            "graph_mode": "hybrid",
            "spatial_top_k": 4,
            "rbf_bins": 8,
            "rbf_max_distance": 12.0,
        }
    }
    output = GHXToxModel(config)(collated)
    assert output["logits"].shape == (2,)

def test_evaluate_parser_constructs():
    args = build_evaluate_arg_parser().parse_args([])
    assert args.checkpoint == "runs/3d_v2_default/best_model.pt"
    assert args.processed == "data/processed/test1_chemical_sites_final_esm2.pt"
    assert args.threshold is None


def test_default_inference_falls_back_without_chemical_sites(tmp_path):
    fallback_path = tmp_path / "fallback.pt"
    primary_path = tmp_path / "primary.pt"
    torch.save(
        {
            "config": {"model": {}, "inference": {"default_threshold": 0.85}},
            "model_state": {},
        },
        fallback_path,
    )
    torch.save(
        {
            "config": {
                "model": {"chemical_site_branch": True},
                "inference": {
                    "default_threshold": 0.67,
                    "fallback_checkpoint": str(fallback_path),
                    "fallback_threshold": 0.85,
                },
            },
            "model_state": {},
        },
        primary_path,
    )
    checkpoint, selected, threshold, fallback_used = resolve_inference_checkpoint(
        primary_path,
        [{"sequence": "AC"}],
        torch.device("cpu"),
    )
    assert checkpoint["config"]["model"] == {}
    assert selected == str(fallback_path)
    assert threshold == 0.85
    assert fallback_used


def test_default_inference_uses_v2_when_sites_are_attached(tmp_path):
    primary_path = tmp_path / "primary.pt"
    torch.save(
        {
            "config": {
                "model": {"chemical_site_branch": True},
                "inference": {"default_threshold": 0.677819},
            },
            "model_state": {},
        },
        primary_path,
    )
    checkpoint, selected, threshold, fallback_used = resolve_inference_checkpoint(
        primary_path,
        [{"chemical_site_mask": torch.ones(2, 2, dtype=torch.bool)}],
        torch.device("cpu"),
    )
    assert checkpoint["config"]["model"]["chemical_site_branch"]
    assert selected == str(primary_path)
    assert threshold == pytest.approx(0.677819)
    assert not fallback_used


def test_esmfold_predict_parser_defaults_to_cuda():
    args = build_esmfold_predict_arg_parser().parse_args([])
    assert args.device == "cuda"
    assert args.fasta == "dataset/train_data or benchmark_data.fasta"


def test_learned_confidence_gate_constructs():
    batch = [
        {
            "sample_id": "p_1",
            "sequence": "ACDE",
            "aa_ids": torch.tensor([2, 3, 4, 5]),
            "group_ids": torch.tensor([0, 1, 2, 3]),
            "residue_features": torch.zeros(4, RESIDUE_FEATURE_DIM),
            "plm_features": torch.zeros(4, 6),
            "coords": torch.randn(4, 3),
            "plddt": torch.tensor([0.95, 0.8, 0.45, 0.3]),
            "label": 1,
        }
    ]
    collated = collate_peptides(batch)
    config = {
        "model": {
            "fusion_gate": "learned_confidence",
            "structure_feature_dim": STRUCTURE_FEATURE_DIM,
            "plm_embedding_dim": 6,
            "aa_embedding_dim": 8,
            "group_embedding_dim": 4,
            "hidden_dim": 32,
            "num_sequence_layers": 1,
            "num_egnn_layers": 1,
            "num_attention_heads": 4,
            "dropout": 0.1,
            "graph_mode": "hybrid",
            "spatial_top_k": 4,
            "rbf_bins": 8,
            "rbf_max_distance": 12.0,
        }
    }
    model = GHXToxModel(config)
    output = model(collated)
    assert output["logits"].shape == (1,)
    assert output["global_gate"].shape == (1,)
    assert any("confidence" in name for name, _ in model.fusion.named_parameters())


def test_spatial_branch_can_exclude_plm_without_removing_sequence_plm():
    config = {
        "model": {
            "plm_embedding_dim": 6,
            "spatial_plm_embedding_dim": 0,
            "structure_feature_dim": STRUCTURE_FEATURE_DIM,
            "hidden_dim": 32,
            "aa_embedding_dim": 8,
            "group_embedding_dim": 4,
            "num_sequence_layers": 1,
            "num_egnn_layers": 1,
            "num_attention_heads": 4,
            "dropout": 0.1,
        }
    }
    model = GHXToxModel(config)
    assert model.sequence_branch.plm_projection is not None
    assert model.spatial_branch.plm_projection is None


def test_atom_graph_and_atom_only_forward():
    records = []
    for index, sequence in enumerate(("ACDE", "KWR")):
        graph = peptide_atom_graph(sequence)
        assert graph["atom_features"].shape[1] == ATOM_FEATURE_DIM
        assert graph["atom_edge_features"].shape[1] == EDGE_FEATURE_DIM
        records.append(
            {
                "sample_id": f"atom_{index}",
                "sequence": sequence,
                "aa_ids": torch.arange(len(sequence)),
                "group_ids": torch.zeros(len(sequence), dtype=torch.long),
                "residue_features": torch.zeros(len(sequence), RESIDUE_FEATURE_DIM),
                "coords": torch.zeros(len(sequence), 3),
                "plddt": torch.ones(len(sequence)),
                "label": index % 2,
                **graph,
            }
        )
    batch = collate_peptides(records)
    config = {
        "model": {
            "modality": "atom_only",
            "hidden_dim": 32,
            "num_atom_layers": 2,
            "num_attention_heads": 4,
            "num_sequence_layers": 1,
            "num_egnn_layers": 1,
            "dropout": 0.1,
        }
    }
    output = GHXToxModel(config)(batch)
    assert output["logits"].shape == (2,)


def test_multiscale_1d_sequence_forward():
    records = []
    for index, sequence in enumerate(("ACDEFG", "KWRH")):
        records.append(
            {
                "sample_id": f"sequence_1d_{index}",
                "sequence": sequence,
                "aa_ids": torch.arange(len(sequence)),
                "group_ids": torch.zeros(len(sequence), dtype=torch.long),
                "residue_features": torch.zeros(len(sequence), RESIDUE_FEATURE_DIM),
                "plm_features": torch.randn(len(sequence), 6),
                "coords": torch.zeros(len(sequence), 3),
                "plddt": torch.ones(len(sequence)),
                "label": index % 2,
            }
        )
    batch = collate_peptides(records)
    config = {
        "model": {
            "modality": "sequence_only",
            "sequence_architecture": "multiscale_1d",
            "sequence_kernels": [3, 5, 7],
            "sequence_use_multiscale": True,
            "sequence_use_bilstm": True,
            "sequence_use_residual": True,
            "hidden_dim": 32,
            "plm_embedding_dim": 6,
            "aa_embedding_dim": 8,
            "group_embedding_dim": 4,
            "dropout": 0.1,
            "global_features": True,
        }
    }
    output = GHXToxModel(config)(batch)
    assert output["logits"].shape == (2,)


def test_fold_fallback_and_manifest_loading(tmp_path):
    groups, protocol = assign_reference_groups(
        ["ACDEFG", "ACDEFA", "WWWWWW"], ["ACDEFG", "WWWWWW"], threshold=0.8
    )
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]
    assert protocol["method"] == "biopython_reference_alignment_fallback"
    manifest = tmp_path / "folds.csv"
    manifest.write_text(
        "source_index,sample_id,label,sequence,group_id,fold\n"
        "0,a,1,ACD,g0,0\n1,b,0,EFG,g1,1\n2,c,0,HIK,g2,1\n",
        encoding="utf-8",
    )
    train_indices, validation_indices = load_fold_indices(manifest, 0, 3)
    assert train_indices == [1, 2]
    assert validation_indices == [0]


def test_oof_threshold_selection_improves_mcc():
    probabilities = torch.tensor([0.10, 0.20, 0.30, 0.40])
    logits = torch.logit(probabilities)
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    threshold, metrics = _best_threshold(logits, labels, "mcc")
    assert 0.2 < threshold <= 0.3
    assert metrics["mcc"] == 1.0


def test_sequence_atom_cross_attention_forward():
    records = []
    for index, sequence in enumerate(("ACDE", "KWR")):
        records.append(
            {
                "sample_id": f"cross_{index}",
                "sequence": sequence,
                "aa_ids": torch.arange(len(sequence)),
                "group_ids": torch.zeros(len(sequence), dtype=torch.long),
                "residue_features": torch.zeros(len(sequence), RESIDUE_FEATURE_DIM),
                "plm_features": torch.randn(len(sequence), 6),
                "coords": torch.zeros(len(sequence), 3),
                "plddt": torch.ones(len(sequence)),
                "label": index % 2,
                **peptide_atom_graph(sequence),
            }
        )
    batch = collate_peptides(records)
    config = {
        "model": {
            "modality": "sequence_atom",
            "hidden_dim": 32,
            "plm_embedding_dim": 6,
            "num_atom_layers": 2,
            "num_attention_heads": 4,
            "num_sequence_layers": 1,
            "num_egnn_layers": 1,
            "dropout": 0.1,
        }
    }
    output = GHXToxModel(config)(batch)
    assert output["logits"].shape == (2,)


def test_multiscale_sequence_atom_forward():
    sequence = "ACWY"
    record = {
        "sample_id": "multiscale_0",
        "sequence": sequence,
        "aa_ids": torch.arange(len(sequence)),
        "group_ids": torch.zeros(len(sequence), dtype=torch.long),
        "residue_features": torch.zeros(len(sequence), RESIDUE_FEATURE_DIM),
        "plm_features": torch.randn(len(sequence), 6),
        "coords": torch.zeros(len(sequence), 3),
        "plddt": torch.ones(len(sequence)),
        "label": 1,
        **peptide_atom_graph(sequence, feature_set="extended"),
    }
    batch = collate_peptides([record])
    config = {
        "model": {
            "modality": "sequence_atom",
            "hidden_dim": 32,
            "plm_embedding_dim": 6,
            "num_atom_layers": 4,
            "atom_multiscale": True,
            "atom_feature_dim": EXTENDED_ATOM_FEATURE_DIM,
            "num_attention_heads": 4,
            "num_sequence_layers": 1,
            "num_egnn_layers": 1,
            "dropout": 0.1,
        }
    }
    model = GHXToxModel(config)
    output = model(batch)
    assert output["logits"].shape == (1,)
    assert model.atom_branch.scale_fusion is not None
    assert batch["atom_features"].shape[-1] == EXTENDED_ATOM_FEATURE_DIM


def test_smoothed_bce_softens_binary_targets():
    loss = SmoothedBCEWithLogitsLoss(smoothing=0.1)
    logits = torch.tensor([2.0, -2.0])
    labels = torch.tensor([1.0, 0.0])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([0.9, 0.1])
    )
    assert torch.allclose(loss(logits, labels), expected)


def test_atom_residual_fusion_starts_with_small_weight():
    sequence = "ACDE"
    record = {
        "sample_id": "residual_0",
        "sequence": sequence,
        "aa_ids": torch.arange(len(sequence)),
        "group_ids": torch.zeros(len(sequence), dtype=torch.long),
        "residue_features": torch.zeros(len(sequence), RESIDUE_FEATURE_DIM),
        "plm_features": torch.randn(len(sequence), 6),
        "coords": torch.randn(len(sequence), 3),
        "plddt": torch.ones(len(sequence)),
        "label": 1,
        **peptide_atom_graph(sequence),
    }
    batch = collate_peptides([record])
    config = {
        "model": {
            "modality": "fusion_atom_residual",
            "hidden_dim": 32,
            "plm_embedding_dim": 6,
            "num_atom_layers": 2,
            "num_attention_heads": 4,
            "num_sequence_layers": 1,
            "num_egnn_layers": 1,
            "structure_feature_dim": STRUCTURE_FEATURE_DIM,
            "dropout": 0.1,
            "atom_residual_initial_weight": 0.05,
        }
    }
    output = GHXToxModel(config)(batch)
    assert output["logits"].shape == (1,)
    assert torch.allclose(output["atom_residual_weight"], torch.tensor(0.05), atol=1e-5)


def test_residual_experts_start_from_sequence_base():
    sequence = "ACDE"
    record = {
        "sample_id": "residual_experts_0",
        "sequence": sequence,
        "aa_ids": torch.arange(len(sequence)),
        "group_ids": torch.zeros(len(sequence), dtype=torch.long),
        "residue_features": torch.zeros(len(sequence), RESIDUE_FEATURE_DIM),
        "plm_features": torch.randn(len(sequence), 6),
        "coords": torch.randn(len(sequence), 3),
        "plddt": torch.tensor([0.9, 0.8, 0.7, 0.6]),
        "label": 1,
        **peptide_atom_graph(sequence),
    }
    batch = collate_peptides([record])
    config = {
        "model": {
            "modality": "residual_experts",
            "sequence_architecture": "multiscale_1d",
            "sequence_use_multiscale": False,
            "sequence_use_bilstm": True,
            "sequence_use_residual": False,
            "hidden_dim": 32,
            "plm_embedding_dim": 6,
            "num_atom_layers": 2,
            "num_attention_heads": 4,
            "num_egnn_layers": 1,
            "structure_feature_dim": STRUCTURE_FEATURE_DIM,
            "dropout": 0.0,
            "residual_atom_initial_weight": 0.1,
            "residual_spatial_initial_weight": 0.1,
        }
    }
    output = GHXToxModel(config)(batch)
    assert output["logits"].shape == (1,)
    assert torch.allclose(output["logits"], output["base_logits"], atol=1e-6)
    assert torch.allclose(output["atom_delta"], torch.zeros(1), atol=1e-6)
    assert torch.allclose(output["spatial_delta"], torch.zeros(1), atol=1e-6)


def test_local_frames_are_rotation_equivariant():
    layer = PLDDTAwareEGNNLayer(16, 8, 0.0, local_frame_edge_features=True)
    coords = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 0.4, 0.3], [3.0, 0.1, 0.5]]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    frames = layer._local_frames(coords, mask)
    rotated_frames = layer._local_frames(coords @ rotation.T, mask)
    assert torch.allclose(rotated_frames, rotation @ frames, atol=1e-5)
    relative = frames.transpose(-1, -2).unsqueeze(2) @ frames.unsqueeze(1)
    rotated_relative = rotated_frames.transpose(-1, -2).unsqueeze(2) @ rotated_frames.unsqueeze(1)
    assert torch.allclose(relative, rotated_relative, atol=1e-5)


def test_full_backbone_geometry_is_rotation_invariant():
    layer = PLDDTAwareEGNNLayer(16, 8, 0.0, backbone_geometry_edge_features=True)
    ca = torch.tensor([[[0.0, 0.0, 0.0], [3.8, 0.4, 0.2], [7.5, 1.0, 0.5]]])
    offsets = torch.tensor(
        [[-1.2, 0.4, 0.1], [0.0, 0.0, 0.0], [1.3, 0.3, 0.2], [2.0, -0.2, 0.4], [0.1, 1.5, 0.2]]
    )
    backbone = ca.unsqueeze(2) + offsets.view(1, 1, 5, 3)
    backbone_mask = torch.ones(1, 3, 5, dtype=torch.bool)
    mask = torch.ones(1, 3, dtype=torch.bool)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    frames = layer._backbone_frames(ca, mask, backbone, backbone_mask)
    rotated_frames = layer._backbone_frames(ca @ rotation.T, mask, backbone @ rotation.T, backbone_mask)
    assert torch.allclose(rotated_frames, rotation @ frames, atol=1e-5)
    distances = torch.cdist(ca, ca)
    features = layer._backbone_pair_features(distances, backbone, backbone_mask)
    rotated_features = layer._backbone_pair_features(
        torch.cdist(ca @ rotation.T, ca @ rotation.T), backbone @ rotation.T, backbone_mask
    )
    assert torch.allclose(features, rotated_features, atol=1e-5)


def test_coordinate_noise_preserves_intra_residue_backbone_geometry():
    backbone = torch.randn(2, 4, 5, 3)
    functional_group_coords = backbone[:, :, 1] + torch.tensor([0.0, 2.0, 0.0])
    chemical_site_coords = backbone[:, :, 1].unsqueeze(2) + torch.tensor(
        [[0.0, 2.0, 0.0], [1.0, 0.0, 0.0]]
    ).view(1, 1, 2, 3)
    batch = {
        "coords": backbone[:, :, 1].clone(),
        "backbone_coords": backbone.clone(),
        "backbone_mask": torch.ones(2, 4, 5, dtype=torch.bool),
        "functional_group_coords": functional_group_coords.clone(),
        "functional_group_mask": torch.ones(2, 4, dtype=torch.bool),
        "chemical_site_coords": chemical_site_coords.clone(),
        "mask": torch.ones(2, 4, dtype=torch.bool),
        "plddt": torch.full((2, 4), 0.8),
    }
    augmented = _apply_coordinate_noise(batch, 0.2, 2.0)
    before = backbone[:, :, 0] - backbone[:, :, 1]
    after = augmented["backbone_coords"][:, :, 0] - augmented["backbone_coords"][:, :, 1]
    assert torch.allclose(before, after, atol=1e-6)
    assert torch.allclose(augmented["coords"], augmented["backbone_coords"][:, :, 1])
    assert torch.allclose(
        augmented["functional_group_coords"] - augmented["coords"],
        functional_group_coords - backbone[:, :, 1],
        atol=1e-6,
    )
    assert torch.allclose(
        augmented["chemical_site_coords"] - augmented["coords"].unsqueeze(2),
        chemical_site_coords - backbone[:, :, 1].unsqueeze(2),
        atol=1e-6,
    )


def test_bootstrap_confidence_intervals_are_reproducible(tmp_path):
    paths = []
    for run_index, probabilities in enumerate(((0.1, 0.8, 0.3, 0.9), (0.2, 0.7, 0.4, 0.95))):
        path = tmp_path / f"predictions_{run_index}.csv"
        path.write_text(
            "sample_id,label,toxicity_probability\n"
            + "\n".join(
                f"sample_{index},{label},{probability}"
                for index, (label, probability) in enumerate(zip((0, 1, 0, 1), probabilities))
            )
            + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    first = bootstrap_confidence_intervals(paths, threshold=0.5, iterations=100, seed=7)
    second = bootstrap_confidence_intervals(paths, threshold=0.5, iterations=100, seed=7)
    assert first == second
    assert first["protocol"]["num_positive"] == 2
    assert first["aggregate"]["point_estimate"]["mcc"] == 1.0


def test_prediction_csv_preserves_float32_probability(tmp_path):
    probability = float(torch.tensor(0.99999994, dtype=torch.float32))
    output = tmp_path / "predictions.csv"
    _write_predictions(
        [
            {
                "sample_id": "sample_1",
                "sequence": "AC",
                "label": 1,
                "toxicity_probability": probability,
                "global_3d_gate": 0.123456789,
            }
        ],
        output,
        threshold=0.85,
    )
    saved_probability = float(output.read_text(encoding="utf-8").splitlines()[1].split(",")[3])
    assert float(torch.tensor(saved_probability, dtype=torch.float32)) == probability


def test_sequence_similarity_audit_finds_high_identity_neighbor():
    from ghxtox.fasta import FastaRecord

    train = [FastaRecord("train", "ACDEFGHIKLMN", 1, "train|1")]
    query = [FastaRecord("query", "ACDEFGHIKLMA", 1, "query|1")]
    assert alignment_identity(train[0].sequence, query[0].sequence) == 11 / 12
    result = audit_similarity(train, query, thresholds=(0.9, 0.8))
    assert result["summary"]["thresholds"]["0.9"]["high_similarity"] == 1
    assert result["rows"][0]["nearest_label_match"] is True


def test_sequence_similarity_candidate_filter_keeps_short_080_pair():
    from ghxtox.fasta import FastaRecord

    train = [FastaRecord("train", "AAXAA", 0, "train|0")]
    query = [FastaRecord("query", "AAYAA", 1, "query|1")]
    result = audit_similarity(train, query, thresholds=(0.8,))
    assert result["rows"][0]["max_identity"] == 0.8
    summary = result["summary"]["thresholds"]["0.8"]
    assert summary["high_similarity"] == 1
    assert summary["high_similarity_label_conflict"] == 1


def test_similarity_subset_eval_excludes_high_identity_samples(tmp_path):
    audit_path = tmp_path / "audit.csv"
    audit_path.write_text(
        "query_id,query_sequence,query_label,max_identity\n"
        "duplicate,AA,1,1.0\nduplicate,CC,1,0.7\nduplicate,DD,0,0.6\n",
        encoding="utf-8",
    )
    prediction_path = tmp_path / "predictions.csv"
    prediction_path.write_text(
        "sample_id,sequence,label,toxicity_probability\n"
        "first_1,AA,1,0.9\nsecond_2,CC,1,0.8\nthird_3,DD,0,0.1\n",
        encoding="utf-8",
    )
    result = evaluate_similarity_subsets(
        audit_path,
        [prediction_path, prediction_path],
        thresholds=(0.8,),
        decision_threshold=0.5,
    )
    subset = result["aggregate"]["0.8"]
    assert subset["num_samples"] == 2
    assert subset["num_positive"] == 1
    assert subset["metrics"]["mcc"]["mean"] == 1.0


def test_prepare_cdhit_fasta_assigns_unique_ids(tmp_path):
    source = tmp_path / "source.fasta"
    source.write_text(">duplicate|1\nACD\n>duplicate|0\nEFG\n", encoding="utf-8")
    output = tmp_path / "output.fasta"
    manifest = tmp_path / "manifest.csv"
    summary = prepare_cdhit_fasta(source, output, manifest, "train")
    text = output.read_text(encoding="utf-8")
    assert ">train_000001|1" in text
    assert ">train_000002|0" in text
    assert summary == {"total": 2, "positive": 1, "negative": 1}


def test_cdhit_combined_clusters_extract_strict_test_subset(tmp_path):
    clusters = tmp_path / "combined.clstr"
    clusters.write_text(
        ">Cluster 0\n0 12aa, >train_000001|1... *\n1 13aa, >test1_000001|1... at 92.31%\n"
        ">Cluster 1\n0 11aa, >test1_000002|0... *\n",
        encoding="utf-8",
    )
    test_fasta = tmp_path / "test.fasta"
    test_fasta.write_text(
        ">test1_000001|1\nACDEFGHIKLMNP\n>test1_000002|0\nAAAAAAAAAAA\n",
        encoding="utf-8",
    )
    output = tmp_path / "strict.fasta"
    summary = extract_strict_subset(clusters, test_fasta, output, "test1_")
    assert len(parse_clusters(clusters)) == 2
    assert summary["num_retained"] == 1
    assert summary["num_excluded"] == 1
    assert read_fasta(output)[0].header == "test1_000002|0"


def test_native_cdhit_audit_preparation_and_comparison(tmp_path):
    reference_manifest = tmp_path / "reference.csv"
    reference_manifest.write_text(
        "cdhit_id,label,sequence\nreference_1,0,AAAA\nreference_2,1,CCCC\n",
        encoding="utf-8",
    )
    positive = tmp_path / "positive.csv"
    positive.write_text("CCCC\nGGGG\n", encoding="utf-8")
    negative = tmp_path / "negative.csv"
    negative.write_text("TTTT\n", encoding="utf-8")
    prepared = prepare_inputs([reference_manifest], positive, negative, tmp_path / "prepared")
    assert prepared["reference_unique"] == 2
    assert prepared["candidate_after_exact_dedup"] == 2

    fallback = tmp_path / "fallback.csv"
    fallback.write_text(
        "sample_id,label,sequence,source_split\nfallback_1,1,GGGG,test\n",
        encoding="utf-8",
    )
    native = tmp_path / "native.fasta"
    native.write_text(">toxinpred3_candidate_0002|0\nTTTT\n", encoding="utf-8")
    summary = summarize_audit(
        prepared["candidate_manifest"],
        fallback,
        {"standard": native},
        tmp_path / "audit",
    )
    assert summary["native_variants"]["standard"]["retained"] == 1
    assert summary["native_variants"]["standard"]["native_only"] == 1
    assert summary["native_variants"]["standard"]["fallback_only"] == 1
    assert summary["num_any_disagreement"] == 2


def test_native_cdhit_length_buckets_and_combine(tmp_path):
    reference = tmp_path / "reference.fasta"
    reference.write_text(">short\nAAAA\n>long\nAAAAAAAAAA\n", encoding="utf-8")
    candidates = tmp_path / "candidates.fasta"
    candidates.write_text(">first|1\nCCCC\n>second|0\nCCCCCCCC\n", encoding="utf-8")
    prepared = prepare_length_buckets(reference, candidates, tmp_path / "buckets", 0.8)
    assert prepared["num_candidate_sequences"] == 2
    assert prepared["buckets"][0]["num_references"] == 1
    assert prepared["buckets"][1]["num_references"] == 1
    for bucket in prepared["buckets"]:
        Path(bucket["native_output"]).write_text(
            Path(bucket["candidate_fasta"]).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    combined = tmp_path / "combined.fasta"
    result = combine_bucket_outputs(prepared["manifest_path"], combined)
    assert result == {"input_candidates": 2, "retained": 2}
    assert len(read_fasta(combined)) == 2


def test_subset_processed_uses_manifest_source_index(tmp_path):
    processed = tmp_path / "processed.pt"
    torch.save(
        {"records": [{"sequence": "AAA", "label": 0}, {"sequence": "CCC", "label": 1}]},
        processed,
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "cdhit_id,source_index,label,sequence\ntrain_000001|0,0,0,AAA\ntrain_000002|1,1,1,CCC\n",
        encoding="utf-8",
    )
    retained = tmp_path / "retained.fasta"
    retained.write_text(">train_000002|1\nCCC\n", encoding="utf-8")
    output = tmp_path / "subset.pt"
    summary = subset_processed(processed, manifest, retained, output)
    assert summary == {"total": 1, "positive": 1, "negative": 0}
    assert torch.load(output, weights_only=False)["records"][0]["sequence"] == "CCC"


def test_report_figure_generation(tmp_path):
    paths = []
    for name in ("test1", "test2"):
        path = tmp_path / f"{name}.csv"
        path.write_text(
            "sample_id,label,toxicity_probability\n"
            "negative,0,0.1\npositive,1,0.9\n",
            encoding="utf-8",
        )
        paths.append(path)
    result = generate_evaluation_figure(paths[0], paths[1], tmp_path / "figures", threshold=0.5)
    assert result["test1"]["confusion_matrix"] == [[1, 0], [0, 1]]
    assert (tmp_path / "figures" / "default_model_evaluation.png").exists()
    assert (tmp_path / "figures" / "figure_metadata.json").exists()

