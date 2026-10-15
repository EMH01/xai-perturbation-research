# Learned Perturbation for Explainable Image Classification

Repository for my **2023 Computer Science Diploma Thesis** on **Explainable Artificial Intelligence (XAI)** for image classifiers:

> *Proposal of a new explainability method to interpret the results of trained models for image classification*

The thesis proposed **VXN (Vision's eXplainer Network)**, a local, model-dependent explanation method in which a learned neural network produces pixel-level relevance information for a trained image classifier. The experimental work used VGG19-BN and Food-101-derived class subsets.

**The research itself was completed in 2023. This repository does not claim subsequent experiments, new scientific results, or a later continuation of the study.**

The thesis PDF is the authoritative record of the original methodology, experiments, results, conclusions, and recommendations. The code in this repository is organized into modules so that the implementation is easier to inspect and run, while preserving the core behavior of the thesis workflow.

## Research idea

~~~mermaid
flowchart LR
    A[Input image] --> B[Learned explainer]
    B --> C[Pixel relevance mask]
    A --> D[Element-wise perturbation]
    C --> D
    D --> E[VGG19-BN classifier]
    E --> F[Class prediction]
~~~

For an input image x, the explainer learns a mask m(x) constrained to [0, 1] by a sigmoid output. The perturbed image is:

~~~text
x' = x ⊙ m(x)
~~~

The classifier weights are not updated during explainer training; the optimizer updates the explainer parameters.

## 2023 thesis implementation

The modular code preserves the central implementation described in the thesis and present in the original Colab/Jupyter workflow:

- VGG19-BN as the image-classification backbone.
- Resize to 256, center-crop to 224 × 224, then tensor conversion.
- Reduced Food-101-derived class sets.
- A dense explainer that:
  - flattens 3 × 224 × 224 image data;
  - applies a fully connected hidden layer;
  - generates a sigmoid pixel mask;
  - perturbs the image by element-wise multiplication.
- A cascade in which the perturbed image is passed to the trained classifier.
- Cross-entropy loss.
- SGD with learning rate 0.001 and momentum 0.9 as the default training configuration represented here.
- Experimental configurations with different class counts, batch sizes, epoch counts, hidden-layer sizes, and perturbation choices.
- The filtered-data condition in which only samples already classified correctly by the classifier are retained for the explainer experiments.

The thesis PDF is preserved in [docs/Proposal of a new explainability method.pdf](docs/Proposal%20of%20a%20new%20explainability%20method.pdf).

## Repository organization

The original thesis workflow was developed in a Colab/Jupyter-style environment. In this repository, the same implementation is separated by responsibility:

~~~text
.
├── scripts/
│   ├── train_classifier.py
│   └── train_explainer.py
├── src/xai_perturbation/
│   ├── checkpoints.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── explanations.py
│   ├── models.py
│   └── training.py
├── tests/
├── docs/
│   └── Proposal of a new explainability method.pdf
├── pyproject.toml
└── README.md
~~~

This organization is a **software refactoring of the thesis implementation, not a scientific update of the research**.

## Technical refactoring notes

A small number of engineering changes make the 2023 implementation easier to run with a conventional Python project structure. They are not presented as new experiments or research findings.

### Current TorchVision syntax

The notebook used:

~~~python
models.vgg19_bn(pretrained=True)
~~~

The modular code uses the current equivalent weights API:

~~~python
weights = VGG19_BN_Weights.DEFAULT
model = vgg19_bn(weights=weights)
~~~

### Checkpoint format

The notebook saved complete PyTorch model objects. The modular project stores state_dict checkpoints plus metadata, which is more portable across environments.

### Dataset splitting

The thesis workflow used a 60/20/20 random_split without an explicit seed. The modular scripts preserve that behavior by default. A seed may be supplied explicitly when a deterministic rerun is desired.

### Correctly classified samples

The thesis workflow identified samples already classified correctly and used the resulting filtered data in the explainer experiments. The modular version represents the same selected samples as a PyTorch Subset instead of rewriting the image files.

### Classifier parameters during explainer training

As in the thesis workflow, the explainer optimizer does not update classifier weights. The modular version marks those parameters as not requiring gradients while leaving the combined network's train/eval mode under normal PyTorch module control.

## Setup

Requirements: Python 3.11+.

~~~bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows

pip install -e ".[dev]"
~~~

## Train the classifier stage

The dataset is expected in TorchVision ImageFolder layout:

~~~text
dataset/
├── class_a/
├── class_b/
└── ...
~~~

Then:

~~~bash
python scripts/train_classifier.py dataset/ \
  --output checkpoints/classifier.pt \
  --epochs 5
~~~

To make the split deterministic for a rerun, an optional seed can be supplied:

~~~bash
python scripts/train_classifier.py dataset/ --seed 42
~~~

## Train the explainer stage

~~~bash
python scripts/train_explainer.py \
  dataset/ \
  checkpoints/classifier.pt \
  --output checkpoints/explainer.pt \
  --epochs 4
~~~

For the filtered-data condition used in the thesis experiments:

~~~bash
python scripts/train_explainer.py \
  dataset/ \
  checkpoints/classifier.pt \
  --correct-only
~~~

The defaults exposed by the script correspond to one of the thesis configurations; the thesis itself documents the different experimental settings that were tested.

## Quality checks

~~~bash
ruff check .
pytest -q
~~~

The tests are software checks for the modular repository. They are **not additional thesis experiments** and do not alter the scientific conclusions of the 2023 work.

## Original 2023 conclusions and recommendations

The thesis concluded that:

- a new local, model-dependent explanation method, VXN, had been developed;
- the explainer and VGG19 classifier were implemented as a cascade;
- results were obtained across different experimental scenarios;
- changes to some model parameters appeared to have a positive effect on the quality of the explanations.

The work explicitly described the results as appropriate for an **experimental stage** and identified further study as necessary.

The recommendations recorded in the thesis included:

- performing a hyperparameter search to improve training of the explainer;
- evaluating the effect of the different perturbation functions;
- studying image transformations that increase brightness and contrast.

These are **recommendations from the 2023 thesis**, not work claimed as completed after the thesis.

## Academic record

The original thesis document is preserved at:

[docs/Proposal of a new explainability method.pdf](docs/Proposal%20of%20a%20new%20explainability%20method.pdf)

It remains the primary academic source for the project.
