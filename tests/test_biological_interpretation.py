import torch

from ghxtox.biological_interpretation import (
    _fixed_context_occlusion_batch,
    aggregate_attributions,
    ca_contact_features,
    plm_integrated_gradients,
    residue_occlusion,
)


class _SumPLMModel(torch.nn.Module):
    def forward(self, batch):
        logits = batch["plm_features"].sum(dim=(1, 2))
        return {"logits": logits}


def _single_batch():
    return {
        "sample_id": ["sample"],
        "sequence": ["ACD"],
        "aa_ids": torch.tensor([[2, 3, 4]]),
        "group_ids": torch.tensor([[0, 1, 2]]),
        "residue_features": torch.ones(1, 3, 2),
        "plm_features": torch.tensor([[[1.0, 2.0], [3.0, 4.0], [-1.0, 2.0]]]),
        "mask": torch.ones(1, 3, dtype=torch.bool),
        "labels": torch.tensor([1.0]),
    }


def test_fixed_context_occlusion_masks_only_selected_sequence_evidence():
    batch = _single_batch()
    occluded = _fixed_context_occlusion_batch(batch, [0, 2])
    assert occluded["aa_ids"].shape == (2, 3)
    assert occluded["aa_ids"][0, 0].item() == 1
    assert occluded["aa_ids"][1, 2].item() == 1
    assert torch.equal(occluded["plm_features"][0, 1], batch["plm_features"][0, 1])
    assert torch.count_nonzero(occluded["residue_features"][1, 2]) == 0


def test_occlusion_and_integrated_gradients_recover_linear_contributions():
    model = _SumPLMModel()
    batch = _single_batch()
    expected = batch["plm_features"].sum(dim=-1)[0]
    occlusion = residue_occlusion(model, batch, length=3, chunk_size=2)
    attribution, delta, residual = plm_integrated_gradients(
        model, batch, length=3, steps=8, baseline_mode="zero"
    )
    assert torch.allclose(occlusion, expected)
    assert torch.allclose(attribution, expected)
    assert abs(delta - float(expected.sum())) < 1e-6
    assert abs(residual) < 1e-6


def test_contact_features_separate_long_range_contacts():
    coords = torch.tensor([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0], [0.0, 4.0, 0.0]])
    contacts, long_range = ca_contact_features(coords, cutoff=5.0, minimum_sequence_separation=3)
    assert contacts.tolist() == [2, 2, 1, 1]
    assert long_range.tolist() == [1, 0, 0, 1]


def test_seed_aggregation_marks_cross_method_direction_agreement():
    template = {
        "probability": 0.8,
        "global_gate": 0.4,
        "node_gate": torch.tensor([0.3, 0.6]),
        "attention_received": torch.tensor([0.4, 0.6]),
        "ig_delta_logit": 1.0,
        "ig_completeness_residual": 0.0,
    }
    first = dict(template, occlusion=torch.tensor([2.0, -1.0]), ig=torch.tensor([1.0, -2.0]))
    second = dict(template, occlusion=torch.tensor([1.0, -3.0]), ig=torch.tensor([2.0, -1.0]))
    aggregated = aggregate_attributions([first, second])
    assert aggregated["same_direction"].tolist() == [True, True]
    assert aggregated["occlusion_sign_agreement"].tolist() == [1.0, 1.0]
    assert aggregated["robust_direction"].tolist() == [1.0, -1.0]
