[pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

Article

ToxPLTC: Peptide Toxicity Prediction by Integrating Pretrained T5

Protein Language Model and Text Convolutional Neural Network

[Yunyun](https://pubs.acs.org/action/doSearch?field1=Contrib&text1=) [Liang](https://pubs.acs.org/action/doSearch?field1=Contrib&text1=)\* [and](https://pubs.acs.org/action/doSearch?field1=Contrib&text1=) [Chenxia](https://pubs.acs.org/action/doSearch?field1=Contrib&text1=) [Wang](https://pubs.acs.org/action/doSearch?field1=Contrib&text1=)

Cite This: [J.](https://pubs.acs.org/action/showCitFormats?doi=10.1021/acs.jcim.5c02745&ref=pdf) [Chem.](https://pubs.acs.org/action/showCitFormats?doi=10.1021/acs.jcim.5c02745&ref=pdf) [Inf.](https://pubs.acs.org/action/showCitFormats?doi=10.1021/acs.jcim.5c02745&ref=pdf) [Model.](https://pubs.acs.org/action/showCitFormats?doi=10.1021/acs.jcim.5c02745&ref=pdf) [2026,](https://pubs.acs.org/action/showCitFormats?doi=10.1021/acs.jcim.5c02745&ref=pdf) [66,](https://pubs.acs.org/action/showCitFormats?doi=10.1021/acs.jcim.5c02745&ref=pdf) [4058−4074](https://pubs.acs.org/action/showCitFormats?doi=10.1021/acs.jcim.5c02745&ref=pdf)

[Read](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02745?ref=pdf) [Online](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02745?ref=pdf)

ACCESS

[Metrics](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02745?goto=articleMetrics&ref=pdf) [&](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02745?goto=articleMetrics&ref=pdf) [More](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02745?goto=articleMetrics&ref=pdf)

[Article](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02745?goto=recommendations&?ref=pdf) [Recommendations](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02745?goto=recommendations&?ref=pdf)

ABSTRACT: Peptide-based therapeutics show promising potential in treating a range of diseases, such as diabetes, cancer, and

chronic pain. However, critical challenges, including peptide toxicity, immunogenicity, and stability deﬁciencies of peptides, have

become major obstacles to their direct clinical application. Traditional toxicity testing methods based on a wet lab are not only costly

but also time-consuming. In contrast, the classiﬁcation model based on deep learning provides a new technical path for the eﬃcient

identiﬁcation of peptide toxicity. In this study, we propose a deep learning framework called ToxPLTC, which employs the ProtT5

protein language model based on the Transformer architecture for pretraining peptide sequences, adopts the borderline SMOTE

algorithm to handle an imbalanced training set data, and utilizes a text convolutional neural network combined with a fully

connected layer for classiﬁcation. Additionally, visualization analysis, motif analysis, and mutation-scan analysis are performed to

understand the function of each module and enhance the interpretability of our model. The applicability domain is constructed

based on the K-NN strategy to deﬁne the eﬀective prediction range of our model to ensure the reliability of model predictions. The

ToxPLTC model achieves a balanced accuracy of 93.02% on independent test set 1 and 88.04% on independent test set 2.

Experimental results demonstrate that our model outperforms existing models on independent test sets and has good generalization

ability. The ToxPLTC model possesses great potential as a valuable and robust tool for peptide-based drug development. The source

data sets and codes can be available at the following GitHub repository: <https://github.com/yunyunliang88/ToxPLTC>.

1

. INTRODUCTION

> With the rapid increase in the number of potential

therapeutic peptides, traditional wet-lab toxicity testing has

been constrained by high costs and long experimental cycles,

motivating the use of computational approaches for toxicity

prediction.<sup>12</sup> Existing approaches primarily fall into two

categories: similarity-based methods and machine learning-

based methods. The former relies on sequence alignment or

homology inference, such as BLAST,<sup>13</sup> but are highly

dependent on homologous toxic peptides, show degraded

performance on large-scale data sets, and are sensitive to

parameter settings. The latter leverages machine learning to

Over the past few decades, peptide-based therapeutics have

remained a prominent research focus in the ﬁeld of

biomedicine.

1

−3

Their therapeutic potential has garnered

unprecedented attention in the past decade, demonstrating

robust and rapid development momentum. Peptides are

multifaceted organic molecules composed of amino acid

4

,5

sequences linked by peptide bonds. And peptides are usually

short chains containing no more than 50 amino acids, which

can regulate a variety of biological functions. They

demonstrate signiﬁcant application potential in treating various

<sub>diseases, including diabetes, cancer, and chronic pain.</sub>6

,7

extract toxicity features from positive and negative samples,

> Compared to small molecules, peptides exhibit higher

biological activity, speciﬁcity, and permeability, while also

such as ClanTox<sup>14</sup> and ToxinPred,

15

enhancing predictive

8

−10

being easier to produce.

These advantages make peptides a

capabilities. However, these methods often rely on statistical

features, making it challenging to fully capture sequence-order-

and position-dependent information. Overall, eﬀectively

highly promising class of therapeutic drugs. However, the

peptide toxicity, immunogenicity, and stability deﬁciencies of

peptides represent core scientiﬁc challenges hindering their

1

1

direct translation into therapeutics. To gain deeper insights

Received: November 9, 2025

and overcome these limitations, current research frontiers

focus on developing novel peptide toxicity identiﬁcation and

assessment technologies. These eﬀorts aim to establish more

precise predictive models, laying the theoretical foundation for

rationally designing low-toxicity, high-eﬃcacy peptide ther-

apeutics.

Revised:

March 6, 2026

Accepted: March 16, 2026

Published: March 23, 2026

©

2026 American Chemical Society

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074

4

058

Journal of Chemical Information and Modeling

[pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

Article

4

059

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074

Journal of Chemical Information and Modeling

[pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

Article

Figure 1. continued

Figure 1. Overall ﬂowchart of the ToxPLTC model for peptide toxicity prediction. (A) Data set preparation. (B) Features extraction using ProT5

protein language model. (C) Imbalanced data processing by borderline SMOTE on the training set. (D) Features adjustment aimed to prepare

well-conditioned features for TextCNN. (E) TextCNN module. (F) Classiﬁcation using the fully connected layer.

integrating deeper sequence features remains crucial for

improving the peptide toxicity prediction performance.

peptide sequences. Finally, the features extracted by TextCNN

are fed into a fully connected layer for classiﬁcation decisions.

Experimental results demonstrate that the ToxPLTC model

exhibits an eﬀective and reliable classiﬁcation performance.

This provides a relatively eﬀective solution with a promising

generalization potential for peptide toxicity prediction. The

framework of the ToxPLTC model is presented in Figure 1A−

F.

> In recent years, deep neural network-based methods have

achieved remarkable achievements in the ﬁeld of bioinfor-

matics and have achieved signiﬁcant progress.<sup>16</sup> In 2021, Wei

et al. proposed the ATSE model, which integrated graph neural

<sub>networks (GNN) with an attention mechanism.1</sub><sup>7</sup> In 2022,

Wei et al. further developed a novel deep learning framework

called ToxIBTL to predict peptide and protein toxicity by

integrating the information bottleneck principle with transfer

2

> . MATERIALS AND METHODS
>
> .1. Data Sets

In order to compare fairly, we employ the same data set as the

1

8

learning. Zhao et al. employed both sequence encoding and

variational information bottleneck strategies to obtain richer

2

1

9

features . In 2023, Morozov et al. established the CSM-Toxin

2

1

CAPTP model constructed by Jiao et al. for peptide toxicity

Web server for predicting protein and peptide toxicity based

on the deep learning model ProteinBert.<sup>20</sup> In 2024, Jiao et al.

developed the CAPTP model, which integrated a convolu-

tional neural network (CNN) with a self-attention encoder to

enable end-to-end automatic learning of toxicity-related

<sub>features from peptide sequences.2</sub><sup>1</sup> Rathore et al. introduced

ToxinPred 3.0, an improved variant based on machine learning

and deep learning techniques, which signiﬁcantly enhanced

prediction accuracy.<sup>22</sup> In 2025, Guan et al. proposed ToxiPep,

which combined sequence-based contextual information with

atomic-level structural features for peptide toxicity predic-

prediction. First, they integrated peptide sequences from CSM-

20 12 17

Toxin, ToxinPred2, and ATSE, and deleted sequences with

lengths greater than 50. Next, manually reviewed nontoxic peptides

were retrieved using the keywords “NOT KW-0800 AND NOT KW-

2

5

0020 AND reviewed: true” in the UniProt database. Then, they

discarded these sequences containing B, J, O, U, X, and Z amino acids

and exceeding 50 amino acids in length. Subsequently, these nontoxic

peptides were merged with the previously obtained data set. During

merging, peptide sequences with inconsistent labels and duplicate

peptide sequences were deleted. Hence, 2491 toxic peptides (positive

samples) and 7653 nontoxic peptides (negative samples) were

obtained. Then, redundant sequences were eliminated using CD-

2

3

26

tion. Zhang et al. proposed the ToxMSRC model, which

HIT software with a cutoﬀ of 0.9, resulting in 2138 positive samples

and 5375 negative samples (total 7513 samples). The 7513 samples

were split into the training set and independent test set 1 with an

employed the positive sample augmentation strategy to address

data imbalance and combined a multiscale convolutional

neural network with bidirectional long short-term memory

8

5:15 ratio. Finally, the training set contained 1818 positive samples

and 4569 negative samples, while the independent test set 1

comprised 320 positive samples and 806 negative samples.

> In order to ensure the fairness of comparative evaluation, Jiao et

2

4

(BiLSTM).

> Although the peptide toxicity prediction has made some

progress, there remains room for further improvement. On the

one hand, the application of natural language processing

models in this task is still insuﬃcient; on the other hand, the

inherent class imbalance in peptide toxicity data sets has not

been systematically addressed, which may aﬀect the prediction

performance and generalization ability for toxic peptides.

Overall, existing deep-learning-based peptide toxicity predic-

tion models still exhibit limitations in capturing key features

and local structural information, and their prediction accuracy

and generalization capability remain to be further improved.

> In this study, we propose a novel peptide toxicity prediction

model named ToxPLTC. This model ﬁrst utilizes ProtT5, a

protein language model based on the Transformer architecture,

to generate embeddings from peptide sequences. Through

unsupervised pretrained on large-scale protein sequences,

ProtT5 captures evolutionary information and structural-

functional features embeddings within amino acid sequences,

thereby eﬀectively characterizing the biological properties of

peptide sequences. To address the issue of imbalanced class

distribution in the data set, borderline SMOTE is employed for

data rebalancing. This technique identiﬁes and augments

minority class samples near the classiﬁcation boundary,

generating representative synthetic samples to enhance the

ability of the model to identify critical samples. Features

adjustment serves as input to the text convolutional neural

network (TextCNN), further capturing features within the

2

1

al. removed any samples from the independent test set 1 that were

18

present in the ToxIBTL training data set due to the potential

overlap between the independent test set 1 and the training set of the

ToxIBTL. Hence, the independent test set 2 was obtained and

contained 46 positive samples and 536 negative samples, which was

introduced to reassess the model’s generalization capability. The

compositional details of the three data sets are listed in Table 1.

Table 1. Distribution of Training Set and Independent Test

Sets

> data set

training set

independent test set 1

independent test set 2

positive

negative

total

1818

320

46

4569

806

536

6387

1126

582

> Figure 2 illustrates the probability density distributions of peptide

sequence lengths for toxic and nontoxic peptides in the training set

and two independent test sets. The distribution of positive and

negative samples in the training set and independent test set 1 is

relatively consistent, while independent test set 2 exhibits a certain

shift in the overall distribution due to sample screening. Overall,

substantial overlap is observed between the length distributions of

toxic and nontoxic peptides across all data sets, indicating that the

classiﬁcation task cannot be accomplished based solely on sequence

length. Therefore, it is necessary to learn more discriminative

sequence features. Due to the distribution shift in independent test

4

060

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074

Journal of Chemical Information and Modeling

[pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

Article

Figure 2. Probability density distribution of peptide sequence length on the training set and independent test sets.

set 2, its generalization capability is theoretically weaker than that of

independent test set 1.

feature vector. This procedure is unaﬀected by padding or the

sequence length variability.

2

.2. Feature Representation

2.3. Imbalanced Data Processing

ProtT5 is developed based on Text-To-Text Transfer Transformer

(T5) architecture, along with ProtBert and ProtXLNet, originates

from ProtTrans, a suite of pretrained protein language models

The borderline SMOTE was introduced by Han et al. in 2005;<sup>33</sup> it

generates synthetic samples only for boundary minority samples

rather than sampling the entire minority class, thereby strengthening

the decision boundary and mitigating overgeneralization. The steps of

the algorithm are as follows:

2

7

initiated by Elnaggar et al. ProtT5 employs an encoder-decoder

2

8,29

architecture based on the Transformer framework,

which is

trained via a denoising objective, and the core process involves

randomly masking spans of amino acids in the input protein sequence.

> In this study, we adopt a pretrained model called prot_t5_xl_hal-

Step 1: Boundary samples need to be identiﬁed. For each minority

<sup>sample, x</sup>i<sup>, ﬁnd its k nearest neighbors in the training set. Let m be the</sup>

number of majority class neighbors among these k. If <sup>\< m \< k, x</sup>i <sup>is</sup>

k

3

0

f_uniref50-enc for extracting context-aware embeddings of protein

2

sequences, which is a protein-speciﬁc, extra-large T5-based encoder

model utilizing half-precision arithmetic, pretrained on the big

considered borderline, and these samples are collected into a

dangerous class set. If m = k, then x<sub>i</sub> is considered noise and is

3

1

32

fantastic database

and ﬁne-tuned on UniRef50.

This model

k

2

ignored. If m

<sup>, x</sup>i <sup>is considered safe and is ignored.</sup>

architecture consists of a 24-layer Transformer encoder with 32

attention heads for each encoder, totaling approximately 3 billion

(3B) parameters, and operates using the ﬂoat16 format. The main

structure of the transformer encoder is as follows

Step 2: Targeted synthetic sample generation is required. For each

> nearest neighbors from

sample p in the dangerous class set, ﬁnd its m

′

the minority class only, and then generate a new synthetic sample s<sub>j</sub>

for n randomly selected neighbors q from these m′:

j

transformer encoder = FFN(MHA(X))

\(1\)

s

=

p

\+

×

(

q

p\)

j

j

\(2\)

where MHA stands for multihead self-attention, FFN stands for

feedforward neural network, and X stands for token embedding

combined with relative positional embedding. For a protein sequence

of length L, ProtT5 generates an embedding representation in the

form of an L × 1024 matrix, where 1024 denotes the embedding

dimension for each amino acid residue. During batch processing,

sequences are padded to the same length, and attention masks are

applied to ensure that padded positions do not contribute to feature

computation; sequences exceeding a preset threshold are processed

separately during feature extraction to avoid truncation. To obtain a

ﬁxed-length sequence representation, the embeddings of valid amino

acid residues are averaged along the sequence length dimension L,

thereby representing each protein sequence as a 1024-dimensional

where δ is a random number in the range \[0, 1\].

> Step 3: Merge the original training set with all newly generated

synthetic samples to form a category-balanced new training set for

subsequent model training.

> In this study, the number of toxic peptides is 1818, and the number

of nontoxic peptides is 4569. The disparity between these two ﬁgures

is substantial. If the data set is not subjected to imbalance correction,

it may lead to the model’s inadequate identiﬁcation capability for

positive samples, thereby causing a decline in model performance. To

avoid this issue, we employ borderline SMOTE to perform imbalance

correction on the training set, thereby enhancing model performance.

4

061

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074

Journal of Chemical Information and Modeling

> .4. TextCNN
>
> [pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

z = w z + b

Article

> \(7\)

2

2

2

1

2

3

4

TextCNN was ﬁrst proposed by Kim, and its core idea is to treat

text as a one-dimensional image using convolutional ﬁlters to extract

N-gram local features from word sequences, which are then utilized

for classiﬁcation. The TextCNN used in this study consists of the

following three parts:

<sub>e</sub>z<sub>2i</sub>

<sup>Soft max(z</sup>2i<sup>) =</sup>

K

j=1

<sub>e</sub>z<sub>2j</sub>

\(8\)

where z represents the output of the ﬁrst fully connected layer, z

1

. Features adjustment layer: Features obtained through ProtT5

> encoding and borderline SMOTE are processed for preparing
>
> well-conditioned feature representations with suitable dimen-
>
> sion, stable distribution, and nonlinear activation for the
>
> subsequent module as follows:

1

2

represents the output of the second fully connected layer, and z

2i

represents the predicted probability of the ith class from the second

fully connected layer. w and w denote the weights of the fully

1

2

connected layers, while b and b denote their biases. The output

1

2

probabilities from the Softmax function lie within the range \[0, 1\].

F = ReLU(BN(Linear(F<sub>ProtT5</sub>)))

\(3\)

2.6. Applicability Domain

To ensure the reliability of model predictions, this study establishes an

applicability domain (AD) to deﬁne its eﬀective prediction range of

> The motivation of “Linear + BatchNorm + ReLU” includes

\(1\) the linear layer compressing ProtT5 features from 1024 to

3

5,36

the model within the descriptor space.

In QSAR and bioactivity

1

28 dimensions, signiﬁcantly reducing the number of

prediction studies, models typically exhibit good predictive capability

only within the descriptor space covered by their training data.

Therefore, clearly deﬁning the model’s applicability domain is a

critical step in model validation and application. To this end, this

study employs a distance-based method using k-nearest neighbors (k-

NN)<sup>37</sup> to construct the model’s applicability domain.

> parameters in the subsequent TextCNN convolutional layers,
>
> as the kernels now slide only over 128 dimensions. During this
>
> feature compression, feature transformation is performed by
>
> learning the importance of features through the weight matrix,
>
> automatically selecting the most relevant features. (2) The
>
> BatchNorm layer standardizes each feature dimension of every
>
> batch to zero mean and unit variance, reducing the internal
>
> covariate shift, stabilizing training, and making the data
>
> distribution more consistent, which allows for larger learning
>
> rates to accelerate training. Additionally, the introduction of
>
> noise through mini-batch statistics provides a slight regulariza-
>
> tion eﬀect. (3) The ReLU layer enables the model to learn
>
> complex patterns and alleviates the vanishing gradient
>
> problem, as the gradient in the positive region is always 1.

. Multiscale convolutional layer: Create three 1D convolutional

> layers of diﬀerent sizes; each convolutional layer generates n
>
> feature maps. For instance, kernel size = 3, 4, 5, and n = 64, this
>
> layer captures 64-dimensional 3-g, 4-g, and 5-g features,
>
> respectively. Small kernels capture local detail features, while
>
> large kernels capture broader contextual features. Then, global
>
> maximum pooling is applied to the outputs of diﬀerent kernel
>
> sizes, automatically adapting to the varying lengths of
>
> convolutional outputs and ensuring that each feature map is
>
> pooled into a single value. The speciﬁc operation of multiscale
>
> convolutional is as follows:

1

. Deﬁnition of Descriptor Space

> The feature space of the model is composed of ﬁxed-length

embedding vectors obtained by encoding protein sequences by

using the protein language model ProtT5. The embedding

vectors of all samples collectively form the descriptor space of

the model, which serves as the foundation for the subsequent

applicability domain analysis.

2

2\. Calculation of the AD Threshold

> For each sample in the training set, the average Euclidean

distance to its k nearest neighbors within the training set is

calculated and denoted as d . The average distances of all

i

training samples constitute a set {d , d , ···, and d }, where N is

1

2

N

the number of training samples. The mean μ and standard

deviation σ of this set are computed, and the AD threshold T is

deﬁned as

T = + 2

\(9\)

In this study, k is set to 5.

F<sub>Size</sub> = max pooling(Conv(F))), Size = k, m, n

\(4\)

3

. AD Assessment for Independent Test Sets

> where k, m, and n represent convolution kernels of diﬀerent
>
> scales.

. Features fusion Layer: Concatenate features from three

> diﬀerent scales to obtain more comprehensive features as
>
> follows:
>
> For each sample in the independent test set, the average

Euclidean distance to its ﬁve nearest neighbors in the training

≤

T, the sample is

3

set is calculated and denoted as d . If d

> test test

considered to be inside the AD; otherwise, it is considered to

be outside the AD.

> To evaluate the coverage of the model’s applicable domain

on test data, the AD coverage metric is introduced and deﬁned

as

F

> Output

= Concat(F , F , F )

k

m

n

\(5\)

> The advantages of TextCNN include the following: (1) It can

simultaneously capture local details and broader contextual

information, enhancing the model’s ability to understand multi-

granularity language patterns. (2) CNN lacks the recursive structure,

allowing for highly parallelized computation, which generally results in

faster training and prediction speeds.

> N

N<sub>total</sub>

Coverage =

AD

× 100%

\(10\)

where N

represents the number of test samples located

> AD

inside the applicability domain, and N

is the total number of

total

2

.5. Classiﬁcation Module

samples in the independent test set. This metric measures the

proportion of samples the model can cover in actual

predictions, thereby reﬂecting the practical scope of the

model’s applicable domain.

The classiﬁcation module in this study primarily consists of fully

connected layers and a Softmax layer. Feature vectors extracted by

TextCNN are ﬁrst fed into the ﬁrst fully connected layer, which

incorporates dropout with a probability of 0.2 and the Leaky Rectiﬁed

Linear Unit (LeakyReLU) activation function for regularization and

nonlinear transform. Then, the output passes through the second fully

connected layer and the Softmax layer. The speciﬁc formula is as

follows:

2

.7. Model Performance Metrics

To comprehensively evaluate the performance of the ToxPLTC

model, this study selected six metrics: accuracy (ACC), balanced

accuracy (BACC), sensitivity (Sn), speciﬁcity (Sp), the area under the

ROC curve (auROC), and Matthew’s correlation coeﬃcient

z = LeakyReLU(dropout(w x + b ))

\(6\)

> 38−40

(MCC).

Their calculation formulas are as follows:

1

1

1

4

062

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074

Journal of Chemical Information and Modeling

[pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

Article

> TP + TN

TP + TN + FP + FN

four key hyperparameters: combination of diﬀerent convolu-

tional kernel size (window size) is empirically set within the

range of (4,5,6), (4,6,8), (4,7,11), (4,8,12), (3,5,7), and

(3,6,9); the number of output channels for convolutional

kernel (the number of ﬁlters, denoted n_ﬁlters) is selected

from 16, 32, 48, and 64; the dropout rate is searched between

0.2 and 0.4 with a step size of 0.1; and the learning rate is

explored from 0.0001 to 0.0006 with increments of 0.0001.

The detailed settings and corresponding optimal values for all

hyperparameters are summarized in Table 2. Figure 3 presents

the analysis results of hyperparameter optimization.

ACC =

× 100%

1

i

> TP

TP + FN

TN

y

j

z

j

z

j

z

BACC =

\+

× 100%

2

TN + FP

> TP

TP + FN

Sn =

× 100%

> TN
>
> TN + FP

TP × TN

Sp =

× 100%

FP × FN

MCC =

(TP + FN)(TP + FP)(TN + FP)(TN + FN)

\(11\)

Table 2. Description of Hyperparameter Optimization

where TP and TN are the counts of correct predictions that align with

the actual categories, whereas FP and FN represent the number of

samples whose prediction is inconsistent with the real category.

Speciﬁcally, ACC reﬂects the model’s overall discrimination

capability; BACC and MCC address class imbalance issues; Sn

indicates the model’s ability to identify positive samples, while Sp

reﬂects its ability to identify negative samples. Additionally, this study

employs the receiver operating characteristic (ROC) curve and the

precision-recall (PR) curve for visualization analysis. The ROC curve

evaluates the overall model discrimination ability using the area under

the ROC curve values, where values closer to 1 indicate good

performance. The PR curve illustrates the relationship between

precision and recall, with the area under the PR curve (auPRC) being

particularly suitable for data scenarios with scarce positive samples.

The shape and position of the PR curve itself also hold signiﬁcant

reference value.

hyperparameter

search ranges

optimum

window sizes

\[(4,5,6), (4,6,8), (4,7,11), (4,8,12), (3,5,7)

(4,5,6)

,

(3,6,9)\]

n_ﬁlters

dropout rate

learning rate

\[16,32,48,64\]

\[0.2, 0.3, 0.4\]

64

0.2

0.0005

\[0.0001, 0.0002, 0.0003, 0.0004, 0.0005,

0

.0006\]

> As shown in Figure 3A, the optimal performance reaches a

stable plateau after the third trial among 30 trials, as the

number of trials increases. Subsequent trials fail to signiﬁcantly

improve model performance, and later trial points cluster

around the current optimal solution, indicating that the

optimization process has fully converged. The hyperparameter

importance analysis in Figure 3B reveals that window sizes are

the decisive factor for model performance, accounting for 69%

of the importance. This demonstrates that constructing the

combination of diﬀerent convolutional kernel sizes is crucial

for the model. In contrast, n_ﬁlters and dropout rate have

relatively minor inﬂuences, indicating that the model is

insensitive to changes in these two parameters. Figure 3C

further reveals the speciﬁc inﬂuence mechanisms of each

parameter: model performance is highly dependent on the

selection of the window sizes, which is a prerequisite for

achieving peak performance; the learning rate is identiﬁed as a

highly sensitive parameter, with its optimal values concentrated

around 0.0005; the model performs best with n_ﬁlters set to

4

1,42

The K-fold cross-validation and independent test

are applied

for evaluating a model’s robustness and generalization performance.

In this study, we employ 5-fold cross-validation to evaluate models on

the training set and validate them using an independent test set. The

speciﬁc procedure involves randomly partitioning the data set into ﬁve

equally sized subsets. Four of these subsets are sequentially selected as

the training set, while the remaining subset serves as the validation set

to assess model performance. This process is repeated ﬁve times,

ensuring that each subset is used exactly once as the validation set.

Finally, the performance metrics obtained from the ﬁve tests are

averaged, and the resulting calculation serves as the ﬁnal evaluation

metric for the model’s performance. This method, through multiple

cycles of training and validation, makes more eﬃcient use of limited

data and provides a more robust and reliable performance estimate

than a single data set partition. The independent test uses a data set

that is completely excluded from model training, aiming to evaluate

the model’s generalization ability when faced with entirely new and

unknown data.

6

4; while the dropout rate shows no clear trend within the

range of 0.2 to 0.4, conﬁrming its limited regularizing eﬀect in

this context. Based on the above analysis, the ﬁnal optimal

conﬁguration is determined as follows: window sizes are

(4,5,6), n_ﬁlters are 64, dropout rate is 0.2, and learning rate is

3

. RESULTS AND DISCUSSION

0

.0005. This conﬁguration ensures model performance while

3

.1. Experimental Setup and Hyperparameter

reﬂecting an eﬀective balance between the structural design

Optimization

and optimization strategy.

3

.2. Prediction Performance of the ToxPLTC Model

The deep learning framework is operated in Python 3.9 and

PyTorch 2.8.0 + CPU under PyCharm, and the operating

system is 64-bit Windows 11. The hardware environment is

Intel(R) Core(TM) i5−10210U CPU @ 1.60 GHz with 2.11

GHz, and RAM is 8.0 GB.

The predictive performance of the model on the training set

and independent test sets is shown in Table 3. To evaluate the

stability of the model performance, we employ 5-fold cross-

validation on the training set and assessed six metrics: ACC,

BACC, Sn, Sp, MCC, and auROC. The mean values of 5-fold

and standard deviations are calculated, ACC, BACC, Sn, Sp,

> This study employs the Optuna hyperparameter optimiza-

tion framework to automatically search for optimal parameter

4

3

±

±

combinations. Optuna dynamically adjusts search strategies

MCC, and auROC reach 96.92 0.36%, 96.92 0.37%, 98.51

based on historical trial results, evaluates parameter perform-

ance through objective functions, and intelligently selects the

next set of candidate parameters using the Tree-structured

± 0.53%, 95.33 ± 0.88%, 0.9390 ± 0.0070, and 0.9881 ±

0.0034. The independent test set 1 shares the same

distribution as the training set, evaluating the model’s

generalization ability under ideal conditions. The independent

test set 2 exhibits a distribution shift relative to the training set,

assessing the model’s robustness and generalization ability in

4

4

Parzen Estimator (TPE) sampling algorithm.

We set the

batch size to 32 and epochs to 150 in advance, and employ

ACC as objective value and 30 as number of trials to optimize

4

063

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074

Journal of Chemical Information and Modeling

[pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

Article

Figure 3. Analysis of experimental results for hyperparameter optimization. (A) Process of hyperparameter optimization. (B) Relative importance

of hyperparameters to the model. (C) Inﬂuence of diﬀerent hyperparameter values on model performance.

Table 3. Performance of the ToxPLTC Model On the Training Set and Independent Test Sets

> data set

training set

evaluation

ACC (%)

BACC (%)

Sn (%)

Sp (%)

MCC

> auROC

0.9920

5-fold CV

1

97.39

97.36

99.15

95.56

0.9484

2

97.15

97.20

98.65

95.74

0.9434

0.9911

3

96.66

96.66

98.13

95.19

0.9335

0.9827

4

97.04

97.05

97.68

96.41

0.9409

0.9863

5

96.38

96.92 ± 0.36

93.78

96.34

96.92 ± 0.37

93.02

98.92

98.51 ± 0.53

91.25

93.76

95.33 ± 0.88

94.79

0.9287

0.9390 ± 0.0070

0.8496

0.9884

0.9881 ± 0.0034

0.9709

mean

independent

independent

independent test set 1

independent test set 2

92.61

88.04

82.61

93.47

0.6197

0.9295

real-world complex scenarios. As shown in Table 3, the ACC,

BACC, Sn, Sp, MCC and auROC reach 93.78%, 93.02%,

of BACC and Sn, which is attributed to the distribution shift

between test set 2 and the training set.

> Figure 4 displays the ROC curve and PR curve for the

training set and independent test sets. The ROC curve is used

to comprehensively evaluate the overall performance of the

model under diﬀerent classiﬁcation thresholds, while the PR

curve focuses more on the model’s identiﬁcation performance

on positive samples, particularly suitable for assessing data sets

with imbalanced class distributions. Figure 4A intuitively

9

1.25%, 94.79%, 0.8496 and 0.9709, respectively, for

independent test set 1, and the ACC, BACC, Sn, Sp, MCC

and auROC reach 92.61%, 88.04%, 82.61%, 93.47%, 0.6197

and 0.9295, respectively, for independent test set 2. The

generalization performance on independent test set 2 is clearly

inferior to that on independent test set 1, particularly in terms

4

064

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074

Journal of Chemical Information and Modeling

[pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

Article

Figure 4. ROC and PR curves of the ToxPLTC model. (A) ROC and PR curves for 5-fold cross-validation on the training set. (B) ROC and PR

curves on the independent test set 1. (C) ROC and PR curves on the independent test set 2.

displays 5-fold and mean ROC and PR curves on the training

set, which indicates that the model exhibits good discriminative

ability and stability. Figure 4B,C displays the ROC and PR

curves for independent test set 1 and independent test set 2,

4

065

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074

Journal of Chemical Information and Modeling

[pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

Article

Table 4. Ablation Experiments of the ToxPLTC on the Independent Test Sets

data set

mdel

ACC (%)

BACC (%)

Sn (%)

Sp (%)

MCC

auROC

independent test set 1

ProtT5 + BSMOTE + TextCNN

ProtT5 + TextCNN

93.78

91.92

92.63

92.36

92.72

92.10

91.92

92.90

92.61

91.58

91.24

92.10

92.27

91.75

91.07

91.92

93.02

90.40

91.55

90.71

91.24

91.18

90.87

91.65

88.04

83.51

83.32

85.77

85.87

84.60

85.21

84.69

91.25

86.88

89.06

86.88

87.81

89.06

88.44

88.75

82.61

73.91

73.91

78.26

78.26

76.09

78.26

76.09

94.79

93.92

94.04

94.54

94.67

93.30

93.30

94.54

93.47

93.10

92.72

93.28

93.47

93.10

92.16

93.28

0.8496

0.8028

0.8213

0.8126

0.8217

0.8098

0.8050

0.8268

0.6197

0.5524

0.5428

0.5863

0.5913

0.5669

0.5577

0.5719

0.9709

0.9638

0.9646

0.9616

0.9708

0.9693

0.9634

0.9667

0.9295

0.9242

0.9280

0.9142

0.9229

0.9241

0.9087

0.9254

ProtT5 + BSMOTE + (4,5)

ProtT5 + BSMOTE + (5,6)

ProtT5 + BSMOTE + (4,6)

ProtT5 + BSMOTE + (4)

ProtT5 + BSMOTE + (5)

ProtT5 + BSMOTE + (6)

ProtT5 + BSMOTE + TextCNN

ProtT5 + TextCNN

ProtT5 + BSMOTE + (4,5)

ProtT5 + BSMOTE + (5,6)

ProtT5 + BSMOTE + (4,6)

ProtT5 + BSMOTE + (4)

ProtT5 + BSMOTE + (5)

ProtT5 + BSMOTE + (6)

independent test set 2

Figure 5. Heatmap for ablation experiments of the ToxPLTC on the independent test sets.

respectively. As shown in Figure 4B, the ROC curve for the

independent test set 1 is close to the upper-left corner,

indicating the model’s good overall classiﬁcation performance.

The PR curve for the independent test set 1 is close to the

upper-right corner, indicating that the model can eﬀectively

distinguish toxic peptides. As shown in Figure 4C, the ROC

curve for the independent test set 2 is also close to the upper-

left corner; however, the PR curve deviates from the upper-

right corner, with an auPRC of 0.7353, which is signiﬁcantly

lower than the auPRC of 0.9426 on independent test set 1.

This indicates that the model’s ability to identify toxic peptides

is weaker on independent test set 2, which is related to the

distribution shift between test set 2 and the training set.

results indicate that the ToxPLTC model maintains relatively

high and stable performance across all evaluated metrics on

both test sets, whereas models without the oversampling

strategy or with a reduced combination of convolutional kernel

sizes exhibit lower performance levels compared with the

complete model.

> To facilitate an intuitive comparison of the overall

performance of diﬀerent model conﬁgurations across evalua-

tion metrics, the results in Table 4 are visualized using a

heatmap, as shown in Figure 5. The heatmap illustrates the

distribution of metric values for diﬀerent models. As observed,

the performance across metrics varies among model conﬁg-

urations. The ToxPLTC model is generally associated with

higher-value regions across the evaluated metrics, while other

model variants show more dispersed distributions. This

visualization is consistent with the quantitative results reported

in Table 4 and further illustrates the inﬂuence of diﬀerent

model settings on predictive performance.

3

.3. Ablation Experiments

To evaluate the impact of each module on the overall

performance of the ToxPLTC model, ablation experiments are

conducted on two independent test sets. Speciﬁcally, these

ablation experiments included: Removing the borderline

SMOTE (BSMOTE) oversampling strategy; Adjusting the

convolutional kernel sizes in TextCNN by using single-scale

kernels (4), (5), and (6), as well as multiscale combinations

(4,5), (4,6), and (5,6), followed by comparisons with the

complete model (ProtT5 + BSMOTE + TextCNN). Table 4

presents the performance metrics of the diﬀerent model

variants for the two independent test sets. The experimental

3

.4. Comparative Experiments

3.4.1. Comparison of Diﬀerent Protein Language

Models. Protein language models can extract key information

from amino acid sequences and convert it into mathematical

representations.<sup>4</sup>

5,46

These representations reveal the struc-

tural, functional, and evolutionary relationships of proteins,

serving as important features for supporting downstream tasks.

4

066

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074

Journal of Chemical Information and Modeling

[pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

Article

Table 5. Performance Comparison of Diﬀerent Protein Language Models

> data set

training set

model

ACC (%)

BACC (%)

Sn (%)

Sp (%)

MCC

auROC

ESM-2-t33

ESM-2-t30

ESM-2-t12

ProtXLNet

ProtBert

96.52

96.48

95.77

94.51

94.05

96.92

92.63

91.30

89.96

90.32

87.21

93.78

91.58

91.24

88.14

91.41

86.77

92.61

96.52

96.48

95.77

94.51

94.04

96.92

91.65

89.02

89.32

88.06

84.00

93.02

87.48

82.33

80.65

80.43

69.97

88.04

98.45

98.49

97.89

96.62

96.21

98.51

89.38

83.75

87.81

82.81

76.56

91.25

82.61

71.74

71.74

67.39

50.00

82.61

94.59

94.47

93.64

92.39

91.87

95.33

93.92

94.29

90.82

93.30

91.44

94.79

92.35

92.91

89.55

93.47

89.93

93.47

0.9311

0.9304

0.9163

0.8909

0.8818

0.9390

0.8218

0.7849

0.7633

0.7618

0.6839

0.8496

0.5905

0.5330

0.4595

0.5179

0.3179

0.6197

0.9882

0.9891

0.9847

0.9780

0.9750

0.9881

0.9691

0.9598

0.9601

0.9364

0.9243

0.9709

0.9213

0.9118

0.9073

0.8838

0.8140

0.9295

ProtT5

independent test set 1

independent test set 2

ESM-2-t33

ESM-2-t30

ESM-2-t12

ProtXLNet

ProtBert

ProtT5

ESM-2-t33

ESM-2-t30

ESM-2-t12

ProtXLNet

ProtBert

ProtT5

Table 6. Performance Comparison between Borderline SMOTE and SMOTE

> data set

training set

> model

SMOTE

borderline SMOTE

SMOTE

borderline SMOTE

SMOTE

borderline SMOTE

ACC (%)

BACC (%)

Sn (%)

Sp (%)

MCC

auROC

96.45

96.92

91.30

93.78

91.07

92.61

96.47

96.92

89.97

93.02

80.25

88.04

97.36

98.51

86.88

91.25

67.39

82.61

95.57

95.33

93.05

94.79

93.10

93.47

0.9293

0.9390

0.7892

0.8496

0.5080

0.6197

0.9881

0.9881

0.9629

0.9709

0.9124

0.9295

independent test set 1

independent test set 2

To evaluate the feature extraction capabilities of diﬀerent

pretraining strategies, this study selects 6 representative models

from the ProtTrans and ESM-2 frameworks, including ProtT5,

ProtBert, ProtXLNet, ESM-2-t12, ESM-2-t30, and ESM-2-t33.

ESM-2 is a protein language model based on the Transformer

and 0.0002 (p \< 0.05) for the training set, independent test set

1, and independent test set 2, respectively. Further Bootstrap

analysis shows that the 95% conﬁdence intervals for the overall

performance improvement of ProtT5 relative to all comparison

models are \[1.060, 2.077\], \[3.360, 6.186\], and \[4.605, 10.559\]

on the training set, independent test set 1, and independent

test set 2, respectively. None of these intervals includes zero,

suggesting that its performance advantage remains statistically

stable.

4

7

framework.

Its technical core lies in pretraining using a

masked language model, which learns the deep evolutionary

constraints and rules governing protein folding and function in

an unsupervised manner by predicting randomly masked

amino acid residues within vast data sets of known peptide

sequences. ESM-2 has multiple versions, including ESM-2-t33,

ESM-2-t30, and ESM-2-t12, which are distinguished by their

parameter counts and number of Transformer layers.

> 3.4.2. Comparison between Borderline SMOTE and

SMOTE. To evaluate the impact of diﬀerent oversampling

strategies on model performance, we compare the model

performance after feature processing using classic SMOTE<sup>50</sup>

and borderline SMOTE. As shown in Table 6, on both the

training set and the two independent test sets, the model

employing the borderline SMOTE strategy consistently

outperforms the model using SMOTE across all six evaluation

metrices. Speciﬁcally, on independent test set 1, borderline

SMOTE improves ACC from 91.30 to 93.78% and BACC

from 89.97 to 93.02%. The MCC shows a particularly notable

increase from 0.7892 to 0.8496, representing an improvement

of 6.04%, indicating a substantial enhancement in the overall

classiﬁcation quality. On independent test set 2, borderline

SMOTE similarly demonstrates comprehensive advantages,

with improvements of 7.79% in BACC and 11.17% in MCC.

These comparative results indicate that borderline SMOTE, by

targeting and augmenting minority class samples near the

classiﬁcation boundary, more eﬀectively optimizes class

distribution balance compared to SMOTE’s uniform sampling

mechanism. This approach enables the model to focus on

learning more challenging classiﬁcation regions, thereby

> Table 5 presents the performance comparison of diﬀerent

models on the training set and two independent test sets.

Overall, the pretraining method based on ProtT5 demonstrates

relatively stable classiﬁcation performance across various

evaluation metrics compared to other protein language models,

while maintaining good generalization capability on the

independent test sets. On the independent test set 1, the

ProtT5-based model improves ACC and BACC by at least

1

.15 and 1.37%, respectively, with an MCC increase of no less

than 2.78%. On independent test set 2, it raises ACC and

BACC by at least 1.03 and 0.56%, respectively, and improves

MCC by more than 2.92%.

To compare the overall performance diﬀerences among the

4

8

protein language models, Friedman tests

and Bootstrap

conﬁdence interval analyses<sup>49</sup> are conducted on the training set

and the two independent test sets. The Friedman test results

indicate statistically signiﬁcant diﬀerences among the models

across all three data sets, with p-values of 0.00004, 0.00009,

4

067

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074

Journal of Chemical Information and Modeling

[pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

Article

Figure 6. Comparison of diﬀerent classiﬁers on independent test sets.

Figure 7. T-SNE visualization of features representation derived from diﬀerent modules on the training set. (A) ProtT5 features. (B) Features after

borderline SMOTE. (C) Features learned using TextCNN. (D) Features learned using fully connected layer.

constructing a decision boundary with good discriminative

power and better generalization performance.

classiﬁers: TextCNN,<sup>34</sup>

CNN,<sup>51</sup>

> and bidirectional gated

recurrent unit (BiGRU).<sup>52</sup> As shown in Figure 6, upon

comparison of the bar charts of the three models across the

same evaluation metrics, it is evident that TextCNN

consistently outperforms both CNN and BiGRU models on

the two independent test sets. The bar lengths corresponding

to each metric for TextCNN are noticeably longer than those

3

.4.3. Comparison of Diﬀerent Classiﬁers. As the

decision-making core of machine learning models, classiﬁers

function by constructing decision boundaries to compress

input features and map them into category probability

distributions. This study compares three representative

4

068

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074

Journal of Chemical Information and Modeling

[pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

Article

Figure 8. Visualization of original features and generated features by borderline SMOTE.

of the other two models. Speciﬁcally, on independent test set 1,

TextCNN achieves a BACC of 93.02%, which is 2.59% and

Figure 7B shows the features distribution after borderline

SMOTE processing, and positive and negative samples remain

highly mixed with blurred class boundaries. This indicates that

the initial feature representation lacks suﬃcient discriminative

power. Subsequently, after the features undergo processing

through TextCNN, the visualization results in Figure 7C

exhibit distinct diﬀerentiation. The positive and negative

samples begin to form independent clusters with overlapping

regions signiﬁcantly reduced. This indicates that the convolu-

tional layers eﬀectively captured the local discriminative

patterns within the sequences. Figure 7D shows the visual-

ization results after passing through the fully connected layer,

and positive and negative samples each form tightly clustered

groups with minimal overlap between classes. The visualization

results demonstrate that the model progressively learned more

discriminative features across layers, eﬀectively improving

classiﬁcation performance. This feature evolution path

provides an intuitive explanation for the model’s classiﬁcation

process.

> To compare the distribution between original features and

generated features by the borderline SMOTE, Figure 8 is

plotted using t-SNE. In Figure 8, the light-colored points

represent the original features, while the dark-colored points

represent the generated minority class features. The positions

of the newly generated features for the minority class in the

feature space are surrounded by the features of the original

samples and ﬁll the “gaps” of the original features, especially

the “gaps” between the minority class and the majority class in

the boundary region. All points collectively form a more

complete data distribution that is closer to the underlying real-

world distribution.

2

.8% higher than CNN and BiGRU, respectively. On

independent test set 2, its BACC further increases to

8

8.04%, representing improvements of 5.81% and 4.53% over

CNN and BiGRU, respectively. Additionally, TextCNN

maintains stable leads in other key metrics such as Sn, Sp,

and MCC.

> This advantage can likely be attributed to TextCNN’s

unique multiscale convolutional architecture. By employing

parallel convolutional kernels of varying sizes, the model is

capable of capturing multilevel local patterns within protein

sequences simultaneously. This multigranularity feature

extraction ability enables the model to construct more

discriminative sequence representations. In contrast, the

single-scale receptive ﬁeld of CNN may limit its feature

perception range, while BiGRU, although eﬀective in modeling

sequential dependencies, exhibits a relatively limited capability

in identifying local key patterns. Therefore, the experimental

results suﬃciently demonstrate that for the task studied in this

paper, TextCNN leverages its structural advantages to exhibit

stronger feature extraction capability and model applicability.

3

.5. Visualization Analysis

To better understand protein representation and deep learning

model performance, we conduct a systematic visualization of

the output features from each module. Speciﬁcally, we use t-

distributed stochastic neighbor embedding (t-SNE)<sup>53</sup> to

project the high-dimensional features learned by each layer

into a two-dimensional plane, thereby clearly illustrating the

evolutionary path of feature learning.

> Figure 7 shows the visualization results based on the training

set, demonstrating a clear feature learning progression. Figure

3

.6. Interpretability Analysis

7

A shows the feature distribution of pretraining ProtT5, and

3.6.1. Motif Analysis. Motifs in peptide sequences are

the positive and negative samples are mixed together. The

number of negative sample points is higher than that of

positive sample points due to the imbalance in the data set.

evolutionarily conserved, short peptide patterns that carry a

speciﬁc functional or structural signiﬁcance. They act as

“keywords” for deciphering the language of protein function,

4

069

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074

Journal of Chemical Information and Modeling

[pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

Article

Figure 9. Discovered conserved motifs for toxic peptides in the training set.

bridging linear amino acid sequences with complex 3D

structures and biological roles. Additionally, they serve as

crucial starting points for both bioinformatics analysis and

<sub>experimental research. In this study, we use MEME 5.5.95</sub><sup>4</sup> to

identify the top ﬁve motifs in the training set of toxic peptides,

as illustrated in Figure 9. The lengths of motifs 1, 2, 3, 4, and 5

are 15, 11, 14, 15, and 15, respectively. Toxic peptides

containing these motifs account for 48.4% of the positive

samples, highlighting the importance of recognizing these

motifs for improving the prediction accuracy of toxic peptides.

Among the ﬁve motifs, cysteine (C) is the most conserved

amino acid, followed by glycine (G), which is consistent with

<sub>the conclusion from Figure 10A generated by kpLogo5</sub><sup>5</sup> using

toxic peptide samples truncated to the ﬁrst 38 amino acid

samples truncated to the ﬁrst 38 amino acid residues in the

training set. Figure 10C,D shows heatmaps of mutation

matrices for two samples from the training set and independent

test set in ISM experiments, respectively, which are generated

by systematically mutating the amino acid at each position in

the sequence to the other 19 amino acids, followed by

repredicting the sequence after each mutation to obtain the

probability of it being classiﬁed as a positive sample. As shown

in Figure 10C, when the amino acids at positions 21, 30, 32,

and 36 are mutated to leucine (L), the predicted probability of

the sequence is signiﬁcantly reduced; that is, it is predicted to

be the negative sample. This is because L at these positions

occurs much more frequently in negative samples (Figure 10B)

than in positive samples (Figure 10A). As shown in Figure

> 0D, when the amino acids at positions 6, 13, 14, 15, 34, 35,

and 36 are mutated to Arginine (R), the predicted probability

of the sequence is signiﬁcantly reduced; that is, it is predicted

to be the negative sample. This is because R at these positions

occurs much more frequently in negative samples than in

positive samples. Similar conclusions also apply to L at

positions 6 and 13. In summary, if a positive sample is mutated

to R or L at some critical positions, it is highly likely to be

predicted as a negative sample. This indicates that our model

eﬀectively captures the important distinguishing features

between positive and negative samples.

1

residues in the training set that C and G exhibit high frequency

at multiple positions. In other words, the high-frequency C and

G in Figure 10A mainly come from conserved motifs. These

2

2

conclusions align with the ﬁndings in ToxiPep , where amino

acids such as C and G exhibit signiﬁcantly higher attention

weights at speciﬁc positions in toxic peptide sequences. Since

toxic peptides shorter than 11 amino acids lack motifs, the

small convolutional kernel in TextCNN can capture sequence

features without motifs, while the large convolutional kernel

can capture sequence features with motifs. This enables our

model to improve the prediction accuracy of toxic peptides by

leveraging these conserved regions.

3

.7. Comparison with diﬀerent existing models

3

.6.2. Mutation-Scan Analysis. In order to enhance the

To evaluate the reliability and practical value of the ToxPLTC

model, this study systematically compares it with current state-

of-the-art models, with results summarized in Table 7. To

address the class imbalance present in the data, we adopt

BACC, which serves as the primary evaluation metric for a

interpretability of our model, we conducted in silico muta-

genesis (ISM) experiments on the sequences of toxic peptides

for mutation-scan analysis. Figure 10A,B is generated by

5

5

kpLogo using toxic peptide samples and nontoxic peptide

4

070

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074

Journal of Chemical Information and Modeling

[pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

Article

Figure 10. Mutation-scan analysis. (A, B) Frequency of occurrence of each amino acid residue at each position for toxic peptides and nontoxic

peptides in the training set, respectively. (C, D) Heatmap of mutation matrices for two samples from the training set and independent test set in

ISM experiments, respectively.

Table 7. Performance Comparison between ToxPLTC and Other State-of-the-Art Models on the Independent Test Sets

data set

method

BACC (%)

Sn (%)

Sp (%)

MCC

auROC

independent test set 1

CSM-Toxin \[19\]

ToxinPred2 \[12\]

ToxIBTL \[17\]

CAPTP \[20\]

ToxMSRC \[23\]

ToxPLTC

ToxIBTL \[17\]

CAPTP \[20\]

ToxMSRC \[23\]

ToxPLTC

47.81

64.22

91.56

91.59

92.17

93.02

78.47

82.95

86.89

88.04

4.06

91.56

32.51

91.56

92.56

96.53

94.79

89.55

91.98

95.52

93.47

−0.076

0.299

0.803

0.811

0.852

0.850

0.431

0.525

0.655

0.620

0.400

0.874

0.916

0.959

0.965

0.971

0.785

0.901

0.943

0.930

95.94

91.56

90.63

87.81

91.25

67.39

73.91

78.26

82.61

independent test set 2

comprehensive assessment of model performance. In the

independent test set 1, ToxPLTC achieves the BACC of

set 1, the Friedman test yields a p-value of 0.0036 (p \< 0.05),

indicating signiﬁcant diﬀerences among the models. The

Bootstrap conﬁdence interval analysis further reveals that the

95% conﬁdence interval for ToxPLTC compared to all models

is \[2.689, 9.274\], which does not include zero, conﬁrming the

statistical signiﬁcance of its performance superiority. On

independent test set 2, the Friedman test similarly shows

signiﬁcant diﬀerences, with a p-value of 0.0109 (p \< 0.05),

while the Bootstrap analysis returns a 95% conﬁdence interval

of \[8.128, 30.484\], further demonstrating the model’s

signiﬁcant and stable performance advantage in more

challenging data environments.

9

3.02%, outperforming all compared models. On independent

test set 2, ToxPLTC attains the BACC of 88.04%, exceeding

the other models by at least 1.15%, demonstrating its stable

overall classiﬁcation capability in imbalanced data environ-

ments. Furthermore, on this test set, the Sn of ToxPLTC

surpasses that of the other models by at least 4.35%, indicating

its stronger ability to identify positive samples.

> To further statistically validate the performance advantage of

our model, Friedman tests and Bootstrap conﬁdence interval

analyses are conducted on both test sets. On independent test

4

071

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074

Journal of Chemical Information and Modeling

[pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

Article

> In summary, both experimental results and statistical tests

consistently indicate that the ToxPLTC model outperforms

existing comparative methods on both independent test sets,

exhibiting good generalization capability and robustness. This

makes it well-suited for prediction tasks in real-world scenarios

characterized by class imbalance.

> Therefore, future research eﬀorts can focus on three

directions: (1) exploring feature extraction methods that

better capture long-range dependencies in sequences to

improve the model’s ability to recognize complex sequence

patterns; (2) by reﬁning the model architecture or training

strategy, the applicability of the model can be extended to

cover a broader range of peptide sequence lengths,

encompassing both shorter and longer peptide sequence; (3)

establishing a user-friendly online web server to facilitate

communication and exchange within the research team. These

improvements will further enhance the model’s predictive

performance and provide a more reliable computational tool

for peptide drug discovery.

3

.8. Applicability Domain Analysis

To further evaluate the reliability and applicability of the

model predictions, this study conducts an applicability domain

analysis on two independent test sets. Based on the

distribution of k-NN average distances in the training set

feature space, the AD threshold of the model is calculated as T

=

1.9492.

Independent test set 1 contains a total of 1126 samples, of

4

. CONCLUSIONS

which 1097 are located within the AD, corresponding to a

coverage rate of 97.42%. This indicates that the vast majority

of test samples exhibit high similarity to the training data in the

feature space, and thus, the prediction results of our model are

considered reliable. For these samples within the AD, the

Peptide-based therapeutics show great promise in the treat-

ment of diseases, such as diabetes, cancer, and chronic pain.

Accurate peptide toxicity prediction is crucial for peptide-based

drug development. In this study, we propose a deep learning

model named ToxPLTC. The model ﬁrst utilizes the protein

language model ProtT5 to generate embedded representations

of peptide sequences rich in semantic information. Sub-

sequently, to address the issue of class imbalance in the data

set, the borderline SMOTE method is introduced to rebalance

the data of the training set, thereby enhancing the model’s

ability to recognize minority classes. During the feature

extraction, the embedded vectors are dimensionally reduced

and fed into TextCNN to further capture local semantic

features within the peptide sequences. Experimental results

demonstrate that the model performs well in predicting

peptide toxicity, showing good potential for guiding the

development of safer and more eﬃcient peptide-based

therapeutics and oﬀering considerable scientiﬁc value.

5

6

model achieves a Brier score of 0.0589 \< 0.1 and an expected

<sub>calibration error (ECE)5</sub><sup>7</sup> of 0.0557 \< 0.1, both below 0.1,

demonstrating that the model not only maintains good

predictive performance but also preserves reasonable calibra-

tion of probability outputs within the applicability domain.

> Independent test set 2 contains a total of 582 samples, of

which 559 are located within the AD, yielding a coverage rate

of 96.05%, which also reﬂects a high AD coverage. For these

samples within the AD, the Brier score and ECE are 0.0702 \<

0

.1 and 0.0736 \< 0.1, respectively, both remaining at low levels.

This further conﬁrms the reliability and stability of the model’s

probability predictions.

> Overall, both independent test sets exhibit high AD

coverage, indicating that the test samples generally reside

within the feature space deﬁned by the training data.

Meanwhile, the Brier score and ECE for samples inside the

AD remain consistently low, suggesting that the model’s

probability predictions are well-calibrated. These results

demonstrate that the applicability domain constructed in this

study eﬀectively delineates the sample space where model

predictions are reliable, thereby enhancing the credibility and

stability of the model in practical applications.

■

ASSOCIATED CONTENT

Data Availability Statement

The source data sets, codes in Python, and pretrained models

are publicly available at [https://github.com/yunyunliang88/](https://github.com/yunyunliang88/ToxPLTC)

[ToxPLTC](https://github.com/yunyunliang88/ToxPLTC).

■

AUTHOR INFORMATION

3

.9. Discussion

Corresponding Author

Experimental results indicate that the ToxPLTC model

demonstrates good performance in predicting peptide toxicity,

and the model eﬀectively distinguishes toxic peptides from

nontoxic peptides, forming clear intraclass clustering and

interclass separation. This suggests that the model has learned

key sequence features associated with toxicity. However,

despite achieving satisfactory performance on BACC, the

model still has room for improvement on certain evaluation

metrics. This indicates that the model’s predictive capabilities

in some aspects require further enhancement.

Yunyun Liang − School of Science, Xi’an Polytechnic

University, Xi’an 710048, P. R. China;

[orcid.org/0000-](https://orcid.org/0000-0002-1749-564X)

[0](https://orcid.org/0000-0002-1749-564X)

[002-1749-564X](https://orcid.org/0000-0002-1749-564X); Email: <yunyunliang88@163.com>

Author

Chenxia Wang − School of Science, Xi’an Polytechnic

University, Xi’an 710048, P. R. China

Complete contact information is available at:

[https://pubs.acs.org/10.1021/acs.jcim.5c02745](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02745?ref=pdf)

> A major limitation of the current model lies in its restrictions

on the applicability of peptide sequence lengths. The model

performs well only on peptide sequences with lengths between

Author Contributions

Y.L. performed the implementation, model creation, and

methodology. C.W., carried out computation, model valida-

tion, model visualization, and model interpretation. Both

authors were involved in drafting and revising the manuscript.

1

1 and 50 amino acids, and its predictive performance may

decline for sequences outside this range. Additionally, although

TextCNN achieves satisfactory classiﬁcation results, it still has

certain limitations in eﬀectively capturing long-distance

dependencies across the entire sequence due to constraints

in sequence length and network depth.

Notes

The authors declare no competing ﬁnancial interest.

4

072

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074

Journal of Chemical Information and Modeling

[pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

Article

■

ACKNOWLEDGMENTS

> \(21\) Jiao, S. H.; Ye, X. C.; Sakurai, T.; et al. [Integrated](https://doi.org/10.1093/bioinformatics/btae297) [convolution](https://doi.org/10.1093/bioinformatics/btae297)

[and](https://doi.org/10.1093/bioinformatics/btae297) [self-attention](https://doi.org/10.1093/bioinformatics/btae297) [for](https://doi.org/10.1093/bioinformatics/btae297) [improving](https://doi.org/10.1093/bioinformatics/btae297) [peptide](https://doi.org/10.1093/bioinformatics/btae297) [toxicity](https://doi.org/10.1093/bioinformatics/btae297) [prediction.](https://doi.org/10.1093/bioinformatics/btae297)

Bioinformatics 2024, 40 (5), No. btae297.

This work was supported by the National Natural Science

Foundation of China (no. 12101480).

> \(22\) Rathore, A. S.; Choudhury, S.; Arora, A.; et al. [ToxinPred](https://doi.org/10.1016/j.compbiomed.2024.108926) [3.0:](https://doi.org/10.1016/j.compbiomed.2024.108926)

[An](https://doi.org/10.1016/j.compbiomed.2024.108926) [improved](https://doi.org/10.1016/j.compbiomed.2024.108926) [method](https://doi.org/10.1016/j.compbiomed.2024.108926) [for](https://doi.org/10.1016/j.compbiomed.2024.108926) [predicting](https://doi.org/10.1016/j.compbiomed.2024.108926) [the](https://doi.org/10.1016/j.compbiomed.2024.108926) [toxicity](https://doi.org/10.1016/j.compbiomed.2024.108926) [of](https://doi.org/10.1016/j.compbiomed.2024.108926) [peptides.](https://doi.org/10.1016/j.compbiomed.2024.108926)

Computers in Biology and Medicine 2024, 179, No. 108926.

> \(23\) Guan, J. H.; Xie, P. L.; Meng, D.; et al. [ToxiPep:](https://doi.org/10.1016/j.csbj.2025.05.039) [Peptide](https://doi.org/10.1016/j.csbj.2025.05.039)

[toxicity](https://doi.org/10.1016/j.csbj.2025.05.039) [prediction](https://doi.org/10.1016/j.csbj.2025.05.039) [via](https://doi.org/10.1016/j.csbj.2025.05.039) [fusion](https://doi.org/10.1016/j.csbj.2025.05.039) [of](https://doi.org/10.1016/j.csbj.2025.05.039) [context-aware](https://doi.org/10.1016/j.csbj.2025.05.039) [representation](https://doi.org/10.1016/j.csbj.2025.05.039) [and](https://doi.org/10.1016/j.csbj.2025.05.039)

[atomic-level](https://doi.org/10.1016/j.csbj.2025.05.039) [graph.](https://doi.org/10.1016/j.csbj.2025.05.039) Computational and Structural Biotechnology Journal

■

REFERENCES

> \(1\) Larché, M.; Wraith, D. C. [Peptide-based](https://doi.org/10.1038/nm1226) [therapeutic](https://doi.org/10.1038/nm1226) [vaccines](https://doi.org/10.1038/nm1226) [for](https://doi.org/10.1038/nm1226)

[allergic](https://doi.org/10.1038/nm1226) [and](https://doi.org/10.1038/nm1226) [autoimmune](https://doi.org/10.1038/nm1226) [diseases.](https://doi.org/10.1038/nm1226) Nat. Med. 2005, 11 (Suppl 4),

S69−S76.

2

025, 27, 2347−2358.

> \(2\) Baig, M. H.; Ahmad, K.; Saeed, M.; et al. [Peptide](https://doi.org/10.1016/j.biopha.2018.04.025) [based](https://doi.org/10.1016/j.biopha.2018.04.025)

[therapeutics](https://doi.org/10.1016/j.biopha.2018.04.025) [and](https://doi.org/10.1016/j.biopha.2018.04.025) [their](https://doi.org/10.1016/j.biopha.2018.04.025) [use](https://doi.org/10.1016/j.biopha.2018.04.025) [for](https://doi.org/10.1016/j.biopha.2018.04.025) [the](https://doi.org/10.1016/j.biopha.2018.04.025) [treatment](https://doi.org/10.1016/j.biopha.2018.04.025) [of](https://doi.org/10.1016/j.biopha.2018.04.025) [neurodegenerative](https://doi.org/10.1016/j.biopha.2018.04.025) [and](https://doi.org/10.1016/j.biopha.2018.04.025)

[other](https://doi.org/10.1016/j.biopha.2018.04.025) [diseases.](https://doi.org/10.1016/j.biopha.2018.04.025) Biomedicine & Pharmacotherapy 2018, 103, 574−581.

> \(3\) Liu, W. S.; Tang, H. C.; Li, L. F.; et al. [Peptide-based](https://doi.org/10.1111/cpr.13025) [therapeutic](https://doi.org/10.1111/cpr.13025)

[cancer](https://doi.org/10.1111/cpr.13025) [vaccine:](https://doi.org/10.1111/cpr.13025) [current](https://doi.org/10.1111/cpr.13025) [trends](https://doi.org/10.1111/cpr.13025) [in](https://doi.org/10.1111/cpr.13025) [clinical](https://doi.org/10.1111/cpr.13025) [application.](https://doi.org/10.1111/cpr.13025) Cell Proliferation

> \(24\) Zhang, S.; Ren, J.; Liang, Y.; Martelli, P. L. [An](https://doi.org/10.1093/bioinformatics/btaf462) [innovative](https://doi.org/10.1093/bioinformatics/btaf462)

[peptide](https://doi.org/10.1093/bioinformatics/btaf462) [toxicity](https://doi.org/10.1093/bioinformatics/btaf462) [prediction](https://doi.org/10.1093/bioinformatics/btaf462) [model](https://doi.org/10.1093/bioinformatics/btaf462) [based](https://doi.org/10.1093/bioinformatics/btaf462) [on](https://doi.org/10.1093/bioinformatics/btaf462) [multi-scale](https://doi.org/10.1093/bioinformatics/btaf462) [convolutional](https://doi.org/10.1093/bioinformatics/btaf462)

[neural](https://doi.org/10.1093/bioinformatics/btaf462) [network](https://doi.org/10.1093/bioinformatics/btaf462) [and](https://doi.org/10.1093/bioinformatics/btaf462) [residual](https://doi.org/10.1093/bioinformatics/btaf462) [connection.](https://doi.org/10.1093/bioinformatics/btaf462) Bioinformatics 2025, 41,

No. btaf462.

> \(25\) UniProt Consortium. [UniProt:](https://doi.org/10.1093/nar/gky1049) [a](https://doi.org/10.1093/nar/gky1049) [worldwide](https://doi.org/10.1093/nar/gky1049) [hub](https://doi.org/10.1093/nar/gky1049) [of](https://doi.org/10.1093/nar/gky1049) [protein](https://doi.org/10.1093/nar/gky1049)

[knowledge.](https://doi.org/10.1093/nar/gky1049) Nucleic Acids Res. 2019, 47 (D1), D506−D515.

> \(26\) Fu, L. M.; Niu, B. F.; Zhu, Z. W.; et al. [CD-HIT:](https://doi.org/10.1093/bioinformatics/bts565) [accelerated](https://doi.org/10.1093/bioinformatics/bts565) [for](https://doi.org/10.1093/bioinformatics/bts565)

[clustering](https://doi.org/10.1093/bioinformatics/bts565) [the](https://doi.org/10.1093/bioinformatics/bts565) [next-generation](https://doi.org/10.1093/bioinformatics/bts565) [sequencing](https://doi.org/10.1093/bioinformatics/bts565) [data.](https://doi.org/10.1093/bioinformatics/bts565) Bioinformatics 2012,

2

021, 54 (5), No. e13025.

\(4\) Apostolopoulos, V.; Bojarska, J.; Chai, T. T.; et al. [A](https://doi.org/10.3390/molecules26020430) [global](https://doi.org/10.3390/molecules26020430)

[review](https://doi.org/10.3390/molecules26020430) [on](https://doi.org/10.3390/molecules26020430) [short](https://doi.org/10.3390/molecules26020430) [peptides:](https://doi.org/10.3390/molecules26020430) [frontiers](https://doi.org/10.3390/molecules26020430) [and](https://doi.org/10.3390/molecules26020430) [perspectives.](https://doi.org/10.3390/molecules26020430) Molecules 2021,

2

6 (2), 430.

2

8 (23), 3150−3152.

\(5\) Lien, S.; Lowman, H. B. [Therapeutic](https://doi.org/10.1016/j.tibtech.2003.10.005) [peptides.](https://doi.org/10.1016/j.tibtech.2003.10.005) Trends Biotechnol.

> \(27\) Elnaggar, A.; Heinzinger, M.; Dallago, C.; et al. [ProtTrans:](https://doi.org/10.1109/TPAMI.2021.3095381)

[toward](https://doi.org/10.1109/TPAMI.2021.3095381) [understanding](https://doi.org/10.1109/TPAMI.2021.3095381) [the](https://doi.org/10.1109/TPAMI.2021.3095381) [language](https://doi.org/10.1109/TPAMI.2021.3095381) [of](https://doi.org/10.1109/TPAMI.2021.3095381) [life](https://doi.org/10.1109/TPAMI.2021.3095381) [through](https://doi.org/10.1109/TPAMI.2021.3095381) [self-supervised](https://doi.org/10.1109/TPAMI.2021.3095381)

[learning.](https://doi.org/10.1109/TPAMI.2021.3095381) IEEE Transactions on Pattern Analysis and Machine

Intelligence 2022, 44 (10), 7112−7127.

2

003, 21 (12), 556−562.

> \(6\) Muttenthaler, M.; King, G. F.; Adams, D. J.; et al. [Trends](https://doi.org/10.1038/s41573-020-00135-8) [in](https://doi.org/10.1038/s41573-020-00135-8)

[peptide](https://doi.org/10.1038/s41573-020-00135-8) [drug](https://doi.org/10.1038/s41573-020-00135-8) [discovery.](https://doi.org/10.1038/s41573-020-00135-8) Nat. Rev. Drug Discovery 2021, 20 (4), 309−

3

25\.

> \(28\) Thumuluri, V.; Almagro Armenteros, J. J.; Johansen, A. R.; et al.

[DeepLoc](https://doi.org/10.1093/nar/gkac278) [2.0:](https://doi.org/10.1093/nar/gkac278) [multi-label](https://doi.org/10.1093/nar/gkac278) [subcellular](https://doi.org/10.1093/nar/gkac278) [localization](https://doi.org/10.1093/nar/gkac278) [prediction](https://doi.org/10.1093/nar/gkac278) [using](https://doi.org/10.1093/nar/gkac278)

[protein](https://doi.org/10.1093/nar/gkac278) [language](https://doi.org/10.1093/nar/gkac278) [models.](https://doi.org/10.1093/nar/gkac278) Nucleic Acids Res. 2022, 50 (W1), W228−

W234.

> \(29\) Fang, Y. T.; Xu, F.; Wei, L. S.; et al. [AFP-MFL:](https://doi.org/10.1093/bib/bbac606) [accurate](https://doi.org/10.1093/bib/bbac606)

[identification](https://doi.org/10.1093/bib/bbac606) [of](https://doi.org/10.1093/bib/bbac606) [antifungal](https://doi.org/10.1093/bib/bbac606) [peptides](https://doi.org/10.1093/bib/bbac606) [using](https://doi.org/10.1093/bib/bbac606) [multi-view](https://doi.org/10.1093/bib/bbac606) [feature](https://doi.org/10.1093/bib/bbac606) [learning.](https://doi.org/10.1093/bib/bbac606)

Briefings Bioinf. 2023, 24 (1), No. bbac606.

> \(30\) Hu, X.; Li, J. Y.; Liu, T. G. [Alg-MFDL:](https://doi.org/10.1016/j.ab.2024.115701) [A](https://doi.org/10.1016/j.ab.2024.115701) [multi-feature](https://doi.org/10.1016/j.ab.2024.115701) [deep](https://doi.org/10.1016/j.ab.2024.115701)

[learning](https://doi.org/10.1016/j.ab.2024.115701) [framework](https://doi.org/10.1016/j.ab.2024.115701) [for](https://doi.org/10.1016/j.ab.2024.115701) [allergenic](https://doi.org/10.1016/j.ab.2024.115701) [proteins](https://doi.org/10.1016/j.ab.2024.115701) [prediction.](https://doi.org/10.1016/j.ab.2024.115701) Anal. Biochem.

> 025, 697, No. 115701.
>
> \(31\) Steinegger, M.; Mirdita, M.; Soding, J. [Protein-level](https://doi.org/10.1038/s41592-019-0437-4) [assembly](https://doi.org/10.1038/s41592-019-0437-4)

[increases](https://doi.org/10.1038/s41592-019-0437-4) [protein](https://doi.org/10.1038/s41592-019-0437-4) [sequence](https://doi.org/10.1038/s41592-019-0437-4) [recovery](https://doi.org/10.1038/s41592-019-0437-4) [from](https://doi.org/10.1038/s41592-019-0437-4) [metagenomic](https://doi.org/10.1038/s41592-019-0437-4) [samples](https://doi.org/10.1038/s41592-019-0437-4)

[manyfold.](https://doi.org/10.1038/s41592-019-0437-4) Nat. Methods 2019, 16 (7), 603−606.

> \(7\) Otvos, L. Jr.; Wade, J. D. [Current](https://doi.org/10.3389/fchem.2014.00062) [challenges](https://doi.org/10.3389/fchem.2014.00062) [in](https://doi.org/10.3389/fchem.2014.00062) [peptide-based](https://doi.org/10.3389/fchem.2014.00062)

[drug](https://doi.org/10.3389/fchem.2014.00062) [discovery.](https://doi.org/10.3389/fchem.2014.00062) Front. Chem. 2014, 2, 62.

> \(8\) Chames, P.; Van Regenmortel, M.; Weiss, E.; et al. [Therapeutic](https://doi.org/10.1111/j.1476-5381.2009.00190.x)

[antibodies:](https://doi.org/10.1111/j.1476-5381.2009.00190.x) [successes,](https://doi.org/10.1111/j.1476-5381.2009.00190.x) [limitations](https://doi.org/10.1111/j.1476-5381.2009.00190.x) [and](https://doi.org/10.1111/j.1476-5381.2009.00190.x) [hopes](https://doi.org/10.1111/j.1476-5381.2009.00190.x) [for](https://doi.org/10.1111/j.1476-5381.2009.00190.x) [the](https://doi.org/10.1111/j.1476-5381.2009.00190.x) [future.](https://doi.org/10.1111/j.1476-5381.2009.00190.x) Br. J.

Pharmacol. 2009, 157 (2), 220−233.

> \(9\) Vlieghe, P.; Lisowski, V.; Martinez, J.; et al. [Synthetic](https://doi.org/10.1016/j.drudis.2009.10.009) [therapeutic](https://doi.org/10.1016/j.drudis.2009.10.009)

[peptides:](https://doi.org/10.1016/j.drudis.2009.10.009) [science](https://doi.org/10.1016/j.drudis.2009.10.009) [and](https://doi.org/10.1016/j.drudis.2009.10.009) [market.](https://doi.org/10.1016/j.drudis.2009.10.009) Drug Discovery Today 2010, 15 (1−2),

4

0−56.

> \(10\) Wang, L.; Wang, N. X.; Zhang, W. P.; et al. [Therapeutic](https://doi.org/10.1038/s41392-022-00904-4)

[peptides:](https://doi.org/10.1038/s41392-022-00904-4) [current](https://doi.org/10.1038/s41392-022-00904-4) [applications](https://doi.org/10.1038/s41392-022-00904-4) [and](https://doi.org/10.1038/s41392-022-00904-4) [future](https://doi.org/10.1038/s41392-022-00904-4) [directions.](https://doi.org/10.1038/s41392-022-00904-4) Signal Trans-

duction Targeted Ther. 2022, 7 (1), 48.

2

> \(11\) Gentilucci, L.; De Marco, R.; Cerisoli, L. [Chemical](https://doi.org/10.2174/138161210793292555)

[modifications](https://doi.org/10.2174/138161210793292555) [designed](https://doi.org/10.2174/138161210793292555) [to](https://doi.org/10.2174/138161210793292555) [improve](https://doi.org/10.2174/138161210793292555) [peptide](https://doi.org/10.2174/138161210793292555) [stability:](https://doi.org/10.2174/138161210793292555) [incorporation](https://doi.org/10.2174/138161210793292555)

[of](https://doi.org/10.2174/138161210793292555) [non-natural](https://doi.org/10.2174/138161210793292555) [amino](https://doi.org/10.2174/138161210793292555) [acids,](https://doi.org/10.2174/138161210793292555) [pseudo-peptide](https://doi.org/10.2174/138161210793292555) [bonds,](https://doi.org/10.2174/138161210793292555) [and](https://doi.org/10.2174/138161210793292555) [cyclization.](https://doi.org/10.2174/138161210793292555)

Curr. Pharm. Des. 2010, 16 (28), 3185−3203.

> \(12\) Sharma, N.; Naorem, L. D.; Jain, S.; et al. [ToxinPred2:](https://doi.org/10.1093/bib/bbac174) [an](https://doi.org/10.1093/bib/bbac174)

[improved](https://doi.org/10.1093/bib/bbac174) [method](https://doi.org/10.1093/bib/bbac174) [for](https://doi.org/10.1093/bib/bbac174) [predicting](https://doi.org/10.1093/bib/bbac174) [toxicity](https://doi.org/10.1093/bib/bbac174) [of](https://doi.org/10.1093/bib/bbac174) [proteins.](https://doi.org/10.1093/bib/bbac174) Briefings Bioinf.

> \(32\) Suzek, B. E.; Wang, Y. Q.; Huang, H. Z.; et al. [UniRef](https://doi.org/10.1093/bioinformatics/btu739) [clusters:](https://doi.org/10.1093/bioinformatics/btu739)

[a](https://doi.org/10.1093/bioinformatics/btu739) [comprehensive](https://doi.org/10.1093/bioinformatics/btu739) [and](https://doi.org/10.1093/bioinformatics/btu739) [scalable](https://doi.org/10.1093/bioinformatics/btu739) [alternative](https://doi.org/10.1093/bioinformatics/btu739) [for](https://doi.org/10.1093/bioinformatics/btu739) [improving](https://doi.org/10.1093/bioinformatics/btu739) [sequence](https://doi.org/10.1093/bioinformatics/btu739)

[similarity](https://doi.org/10.1093/bioinformatics/btu739) [searches.](https://doi.org/10.1093/bioinformatics/btu739) Bioinformatics 2015, 31 (6), 926−932.

> \(33\) Han, H, Wang, W Y, Mao, B H Borderline-SMOTE: a new

over-sampling method in imbalanced data sets learning. In Interna-

tional conference on intelligent computing; Springer Berlin Heidelberg:

Berlin, Heidelberg, 2005: 878−887.

2

022, 23 (5), No. bbac174.

> \(13\) Altschul, S. F.; Madden, T. L.; Schäffer, A. A.; et al. [Gapped](https://doi.org/10.1093/nar/25.17.3389)

[BLAST](https://doi.org/10.1093/nar/25.17.3389) [and](https://doi.org/10.1093/nar/25.17.3389) [PSI-BLAST:](https://doi.org/10.1093/nar/25.17.3389) [a](https://doi.org/10.1093/nar/25.17.3389) [new](https://doi.org/10.1093/nar/25.17.3389) [generation](https://doi.org/10.1093/nar/25.17.3389) [of](https://doi.org/10.1093/nar/25.17.3389) [protein](https://doi.org/10.1093/nar/25.17.3389) [database](https://doi.org/10.1093/nar/25.17.3389) [search](https://doi.org/10.1093/nar/25.17.3389)

[programs.](https://doi.org/10.1093/nar/25.17.3389) Nucleic Acids Res. 1997, 25 (17), 3389−3402.

> \(14\) Naamati, G.; Askenazi, M.; Linial, M. [ClanTox:](https://doi.org/10.1093/nar/gkp299) [a](https://doi.org/10.1093/nar/gkp299) [classifier](https://doi.org/10.1093/nar/gkp299) [of](https://doi.org/10.1093/nar/gkp299)

[short](https://doi.org/10.1093/nar/gkp299) [animal](https://doi.org/10.1093/nar/gkp299) [toxins.](https://doi.org/10.1093/nar/gkp299) Nucleic Acids Res. 2009, 37 (suppl_2), W363−

W368.

> \(34\) Kim, Y.Convolutional neural networks for sentence classi-

ﬁcation. In Proceedings of the 2014 conference on empirical methods in

natural language processing (EMNLP); Doha, Qatar, 25−29 October

2

014, pp. 1746-1751.

> \(35\) Hanser, T.; Barber, C.; Marchaland, J. F.; et al. [Applicability](https://doi.org/10.1080/1062936X.2016.1250229)

[domain:](https://doi.org/10.1080/1062936X.2016.1250229) [towards](https://doi.org/10.1080/1062936X.2016.1250229) [a](https://doi.org/10.1080/1062936X.2016.1250229) [more](https://doi.org/10.1080/1062936X.2016.1250229) [formal](https://doi.org/10.1080/1062936X.2016.1250229) [definition.](https://doi.org/10.1080/1062936X.2016.1250229) SAR and QSAR in

Environmental Research 2016, 27 (11), 865−881.

> \(15\) Gupta, S.; Kapoor, P.; Chaudhary, K.; et al. [In](https://doi.org/10.1371/journal.pone.0073957) [silico](https://doi.org/10.1371/journal.pone.0073957) [approach](https://doi.org/10.1371/journal.pone.0073957)

[for](https://doi.org/10.1371/journal.pone.0073957) [predicting](https://doi.org/10.1371/journal.pone.0073957) [toxicity](https://doi.org/10.1371/journal.pone.0073957) [of](https://doi.org/10.1371/journal.pone.0073957) [peptides](https://doi.org/10.1371/journal.pone.0073957) [and](https://doi.org/10.1371/journal.pone.0073957) [proteins.](https://doi.org/10.1371/journal.pone.0073957) PloS One 2013, 8 (9),

No. e73957.

> \(16\) Le, N. Q. K.; Yapp, E. K. Y.; Nagasundaram, N.; et al.

[Computational](https://doi.org/10.1016/j.csbj.2019.09.005) [identification](https://doi.org/10.1016/j.csbj.2019.09.005) [of](https://doi.org/10.1016/j.csbj.2019.09.005) [vesicular](https://doi.org/10.1016/j.csbj.2019.09.005) [transport](https://doi.org/10.1016/j.csbj.2019.09.005) [proteins](https://doi.org/10.1016/j.csbj.2019.09.005) [from](https://doi.org/10.1016/j.csbj.2019.09.005)

[sequences](https://doi.org/10.1016/j.csbj.2019.09.005) [using](https://doi.org/10.1016/j.csbj.2019.09.005) [deep](https://doi.org/10.1016/j.csbj.2019.09.005) [gated](https://doi.org/10.1016/j.csbj.2019.09.005) [recurrent](https://doi.org/10.1016/j.csbj.2019.09.005) [units](https://doi.org/10.1016/j.csbj.2019.09.005) [architecture.](https://doi.org/10.1016/j.csbj.2019.09.005) Computa-

tional and Structural Biotechnology Journal 2019, 17, 1245.

> \(17\) Wei, L. S.; Ye, X. C.; Xue, Y. Y.; et al. [ATSE:](https://doi.org/10.1093/bib/bbab041) [a](https://doi.org/10.1093/bib/bbab041) [peptide](https://doi.org/10.1093/bib/bbab041) [toxicity](https://doi.org/10.1093/bib/bbab041)

[predictor](https://doi.org/10.1093/bib/bbab041) [by](https://doi.org/10.1093/bib/bbab041) [exploiting](https://doi.org/10.1093/bib/bbab041) [structural](https://doi.org/10.1093/bib/bbab041) [and](https://doi.org/10.1093/bib/bbab041) [evolutionary](https://doi.org/10.1093/bib/bbab041) [information](https://doi.org/10.1093/bib/bbab041) [based](https://doi.org/10.1093/bib/bbab041)

[on](https://doi.org/10.1093/bib/bbab041) [graph](https://doi.org/10.1093/bib/bbab041) [neural](https://doi.org/10.1093/bib/bbab041) [network](https://doi.org/10.1093/bib/bbab041) [and](https://doi.org/10.1093/bib/bbab041) [attention](https://doi.org/10.1093/bib/bbab041) [mechanism.](https://doi.org/10.1093/bib/bbab041) Briefings Bioinf.

> \(36\) Kar, S., Roy, K., Leszczynski, J. Applicability domain: a step

toward conﬁdent predictions and decidability for QSAR modeling. In

Computational Toxicology: Methods and Protocols; Springer New York:

New York, NY, 2018: 141

−

169\.

\(37\) Cover, T. M.; Hart, P. E. [Nearest](https://doi.org/10.1109/TIT.1967.1053964) [neighbor](https://doi.org/10.1109/TIT.1967.1053964) [pattern](https://doi.org/10.1109/TIT.1967.1053964)

[classification.](https://doi.org/10.1109/TIT.1967.1053964) IEEE Transactions on Information Theory 1967, 13

(1), 21

−

> 27\.

2

021, 22 (5), No. bbab041.

> \(38\) Pan, X. Y.; Zuallaert, J.; Wang, X.; et al. [ToxDL:](https://doi.org/10.1093/bioinformatics/btaa656) [deep](https://doi.org/10.1093/bioinformatics/btaa656) [learning](https://doi.org/10.1093/bioinformatics/btaa656)

[using](https://doi.org/10.1093/bioinformatics/btaa656) [primary](https://doi.org/10.1093/bioinformatics/btaa656) [structure](https://doi.org/10.1093/bioinformatics/btaa656) [and](https://doi.org/10.1093/bioinformatics/btaa656) [domain](https://doi.org/10.1093/bioinformatics/btaa656) [embeddings](https://doi.org/10.1093/bioinformatics/btaa656) [for](https://doi.org/10.1093/bioinformatics/btaa656) [assessing](https://doi.org/10.1093/bioinformatics/btaa656) [protein](https://doi.org/10.1093/bioinformatics/btaa656)

[toxicity.](https://doi.org/10.1093/bioinformatics/btaa656) Bioinformatics 2021, 36 (21), 5159

> \(18\) Wei, L. S.; Ye, X. C.; Sakurai, T.; et al. [ToxIBTL:](https://doi.org/10.1093/bioinformatics/btac006) [prediction](https://doi.org/10.1093/bioinformatics/btac006) [of](https://doi.org/10.1093/bioinformatics/btac006)

[peptide](https://doi.org/10.1093/bioinformatics/btac006) [toxicity](https://doi.org/10.1093/bioinformatics/btac006) [based](https://doi.org/10.1093/bioinformatics/btac006) [on](https://doi.org/10.1093/bioinformatics/btac006) [information](https://doi.org/10.1093/bioinformatics/btac006) [bottleneck](https://doi.org/10.1093/bioinformatics/btac006) [and](https://doi.org/10.1093/bioinformatics/btac006) [transfer](https://doi.org/10.1093/bioinformatics/btac006)

[learning.](https://doi.org/10.1093/bioinformatics/btac006) Bioinformatics 2022, 38 (6), 1514−1524.

−

5168\.

> \(39\) Yan, K.; Lv, H.; Guo, Y.; et al. [sAMPpred-GAT:](https://doi.org/10.1093/bioinformatics/btac715) [prediction](https://doi.org/10.1093/bioinformatics/btac715) [of](https://doi.org/10.1093/bioinformatics/btac715)

[antimicrobial](https://doi.org/10.1093/bioinformatics/btac715) [peptide](https://doi.org/10.1093/bioinformatics/btac715) [by](https://doi.org/10.1093/bioinformatics/btac715) [graph](https://doi.org/10.1093/bioinformatics/btac715) [attention](https://doi.org/10.1093/bioinformatics/btac715) [network](https://doi.org/10.1093/bioinformatics/btac715) [and](https://doi.org/10.1093/bioinformatics/btac715) [predicted](https://doi.org/10.1093/bioinformatics/btac715)

[peptide](https://doi.org/10.1093/bioinformatics/btac715) [structure.](https://doi.org/10.1093/bioinformatics/btac715) Bioinformatics 2023, 39 (1), No. btac715.

> \(40\) Chicco, D.; Jurman, G. [The](https://doi.org/10.1186/s12864-019-6413-7) [advantages](https://doi.org/10.1186/s12864-019-6413-7) [of](https://doi.org/10.1186/s12864-019-6413-7) [the](https://doi.org/10.1186/s12864-019-6413-7) [Matthews](https://doi.org/10.1186/s12864-019-6413-7)

[correlation](https://doi.org/10.1186/s12864-019-6413-7) [coefficient](https://doi.org/10.1186/s12864-019-6413-7) [(MCC)](https://doi.org/10.1186/s12864-019-6413-7) [over](https://doi.org/10.1186/s12864-019-6413-7) [F1](https://doi.org/10.1186/s12864-019-6413-7) [score](https://doi.org/10.1186/s12864-019-6413-7) [and](https://doi.org/10.1186/s12864-019-6413-7) [accuracy](https://doi.org/10.1186/s12864-019-6413-7) [in](https://doi.org/10.1186/s12864-019-6413-7) [binary](https://doi.org/10.1186/s12864-019-6413-7)

[classification](https://doi.org/10.1186/s12864-019-6413-7) [evaluation.](https://doi.org/10.1186/s12864-019-6413-7) BMC Genomics 2020, 21 (1), 6.

> \(19\) Zhao, Z. Y.; Gui, J. Y.; Yao, A. Q.; et al. [Improved](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [prediction](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

[model](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [of](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [protein](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [and](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [peptide](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [toxicity](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [by](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [integrating](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [channel](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [attention](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

[into](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [a](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [convolutional](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [neural](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [network](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [and](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [gated](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [recurrent](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) [units.](https://doi.org/10.1021/acsomega.2c05881?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) ACS

Omega 2022, 7 (44), 40569−40577.

> \(20\) Morozov, V.; Rodrigues, C. H. M.; Ascher, D. B. [CSM-toxin:](https://doi.org/10.3390/pharmaceutics15020431) [a](https://doi.org/10.3390/pharmaceutics15020431)

[web-server](https://doi.org/10.3390/pharmaceutics15020431) [for](https://doi.org/10.3390/pharmaceutics15020431) [predicting](https://doi.org/10.3390/pharmaceutics15020431) [protein](https://doi.org/10.3390/pharmaceutics15020431) [toxicity.](https://doi.org/10.3390/pharmaceutics15020431) Pharmaceutics 2023, 15 (2),

> \(41\) Fushiki, T. [Estimation](https://doi.org/10.1007/s11222-009-9153-8) [of](https://doi.org/10.1007/s11222-009-9153-8) [prediction](https://doi.org/10.1007/s11222-009-9153-8) [error](https://doi.org/10.1007/s11222-009-9153-8) [by](https://doi.org/10.1007/s11222-009-9153-8) [using](https://doi.org/10.1007/s11222-009-9153-8) [K-fold](https://doi.org/10.1007/s11222-009-9153-8) [cross-](https://doi.org/10.1007/s11222-009-9153-8)

[validation.](https://doi.org/10.1007/s11222-009-9153-8) Statistics and Computing 2011, 21 (2), 137−146.

4

31\.

4

073

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074

Journal of Chemical Information and Modeling

[pubs.acs.org/jcim](pubs.acs.org/jcim?ref=pdf)

Article

> \(42\) Yadav, S., Shukla, S. [Analysis](https://doi.org/10.1109/IACC.2016.25) [of](https://doi.org/10.1109/IACC.2016.25) [k-fold](https://doi.org/10.1109/IACC.2016.25) [cross-validation](https://doi.org/10.1109/IACC.2016.25) [over](https://doi.org/10.1109/IACC.2016.25)

[hold-out](https://doi.org/10.1109/IACC.2016.25) [validation](https://doi.org/10.1109/IACC.2016.25) [on](https://doi.org/10.1109/IACC.2016.25) [colossal](https://doi.org/10.1109/IACC.2016.25) [datasets](https://doi.org/10.1109/IACC.2016.25) [for](https://doi.org/10.1109/IACC.2016.25) [quality](https://doi.org/10.1109/IACC.2016.25) [classiﬁcation.](https://doi.org/10.1109/IACC.2016.25) In

2

016 IEEE 6th International conference on advanced computing (IACC);

IEEE, 2016: 78−83.

> \(43\) Akiba, T., Sano, S., Yanase, T.et al. Optuna: A next-generation

hyperparameter optimization framework. In Proceedings of the 25th

ACM SIGKDD international conference on knowledge discovery & data

mining. 2019: 2623−2631.

> \(44\) Bergstra, J., Bardenet, R., Bengio, Y.et al.Algorithms for hyper-

parameter optimization. Advances in Neural Information Processing

Systems, 2011, 24.

> \(45\) Madani, A.; Krause, B.; Greene, E. R.; et al. [Large](https://doi.org/10.1038/s41587-022-01618-2) [language](https://doi.org/10.1038/s41587-022-01618-2)

[models](https://doi.org/10.1038/s41587-022-01618-2) [generate](https://doi.org/10.1038/s41587-022-01618-2) [functional](https://doi.org/10.1038/s41587-022-01618-2) [protein](https://doi.org/10.1038/s41587-022-01618-2) [sequences](https://doi.org/10.1038/s41587-022-01618-2) [across](https://doi.org/10.1038/s41587-022-01618-2) [diverse](https://doi.org/10.1038/s41587-022-01618-2) [families.](https://doi.org/10.1038/s41587-022-01618-2)

Nat. Biotechnol. 2023, 41 (8), 1099−1106.

> \(46\) Ferruz, N.; Höcker, B. [Controllable](https://doi.org/10.1038/s42256-022-00499-z) [protein](https://doi.org/10.1038/s42256-022-00499-z) [design](https://doi.org/10.1038/s42256-022-00499-z) [with](https://doi.org/10.1038/s42256-022-00499-z)

[language](https://doi.org/10.1038/s42256-022-00499-z) [models.](https://doi.org/10.1038/s42256-022-00499-z) Nature Machine Intelligence 2022, 4 (6), 521−532.

> \(47\) Lin, Z. M.; Akin, H.; Rao, R.; et al. [Evolutionary-scale](https://doi.org/10.1126/science.ade2574)

[prediction](https://doi.org/10.1126/science.ade2574) [of](https://doi.org/10.1126/science.ade2574) [atomic-level](https://doi.org/10.1126/science.ade2574) [protein](https://doi.org/10.1126/science.ade2574) [structure](https://doi.org/10.1126/science.ade2574) [with](https://doi.org/10.1126/science.ade2574) [a](https://doi.org/10.1126/science.ade2574) [language](https://doi.org/10.1126/science.ade2574) [model.](https://doi.org/10.1126/science.ade2574)

Science 2023, 379 (6637), 1123−1130.

> \(48\) Friedman, M. [The](https://doi.org/10.1080/01621459.1937.10503522) [use](https://doi.org/10.1080/01621459.1937.10503522) [of](https://doi.org/10.1080/01621459.1937.10503522) [ranks](https://doi.org/10.1080/01621459.1937.10503522) [to](https://doi.org/10.1080/01621459.1937.10503522) [avoid](https://doi.org/10.1080/01621459.1937.10503522) [the](https://doi.org/10.1080/01621459.1937.10503522) [assumption](https://doi.org/10.1080/01621459.1937.10503522) [of](https://doi.org/10.1080/01621459.1937.10503522)

[normality](https://doi.org/10.1080/01621459.1937.10503522) [implicit](https://doi.org/10.1080/01621459.1937.10503522) [in](https://doi.org/10.1080/01621459.1937.10503522) [the](https://doi.org/10.1080/01621459.1937.10503522) [analysis](https://doi.org/10.1080/01621459.1937.10503522) [of](https://doi.org/10.1080/01621459.1937.10503522) [variance.](https://doi.org/10.1080/01621459.1937.10503522) Journal of the American

Statistical Association 1937, 32 (200), 675−701.

> \(49\) Efron, B.Bootstrap methods: another look at the jackknife.

Breakthroughs in Statistics. In Springer Series in Statistics; Springer:

New York, NY, 1992: 569−593.

> \(50\) Chawla, N. V.; Bowyer, K. W.; Hall, L. O.; et al. [SMOTE:](https://doi.org/10.1613/jair.953)

[synthetic](https://doi.org/10.1613/jair.953) [minority](https://doi.org/10.1613/jair.953) [over-sampling](https://doi.org/10.1613/jair.953) [technique.](https://doi.org/10.1613/jair.953) Journal of Artificial

Intelligence Research 2002, 16, 321−357.

> \(51\) LeCun, Y.; Bottou, L.; Bengio, Y.; et al. [Gradient-based](https://doi.org/10.1109/5.726791) [learning](https://doi.org/10.1109/5.726791)

[applied](https://doi.org/10.1109/5.726791) [to](https://doi.org/10.1109/5.726791) [document](https://doi.org/10.1109/5.726791) [recognition.](https://doi.org/10.1109/5.726791) Proc. IEEE 2002, 86 (11), 2278−

2

324\.

> \(52\) Cho, K., Van Merriënboer, B., Gulcehre, C.et al. Learning

phrase representations using RNN encoder-decoder for statistical

machine translation. In Proceedings of the 2014 conference on empirical

methods in natural language processing (EMNLP); Statistics, 2014: pp

1

724-1734.

\(53\) Der Maaten, L. V.; Hinton, G. E. [Visualizing](https://doi.org/10.48550/arXiv.2108.01301) [data](https://doi.org/10.48550/arXiv.2108.01301) [using](https://doi.org/10.48550/arXiv.2108.01301) [t-SNE.](https://doi.org/10.48550/arXiv.2108.01301)

J. Mach. Learn. Res. 2008, 9, 2579−2605.

> \(54\) Bailey, T. L.; Elkan, C. Fitting a mixture model by expectation

maximization to discover motifs in biopolymers. Proceedings. Interna-

tional Conference on Intelligent Systems for Molecular Biology 1994, 2,

2

8−36.

> \(55\) Wu, X.; Bartel, D. P. [kpLogo:](https://doi.org/10.1093/nar/gkx323) [positional](https://doi.org/10.1093/nar/gkx323) [k-mer](https://doi.org/10.1093/nar/gkx323) [analysis](https://doi.org/10.1093/nar/gkx323) [reveals](https://doi.org/10.1093/nar/gkx323)

[hidden](https://doi.org/10.1093/nar/gkx323) [specificity](https://doi.org/10.1093/nar/gkx323) [in](https://doi.org/10.1093/nar/gkx323) [biological](https://doi.org/10.1093/nar/gkx323) [sequences.](https://doi.org/10.1093/nar/gkx323) Nucleic Acids Res. 2017, 45

(W1), W534−W538.

> \(56\) BRIER, G. W. [Verification](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2) [of](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2) [forecasts](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2) [expressed](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2) [in](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2) [terms](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2) [of](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2)

[probability.](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2) Mon. Weather Rev. 1950, 78 (1), 1−3.

> \(57\) Guo, C, Pleiss, G, Sun, Yet al. On calibration of modern neural

networks. In 34th International Conference on Machine Learning;

PMLR: Sydney, Australia, 2017, 70: 1321−1330.

4

074

> [https://doi.org/10.1021/acs.jcim.5c02745](https://doi.org/10.1021/acs.jcim.5c02745?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

J. Chem. Inf. Model. 2026, 66, 4058−4074
