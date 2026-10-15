# Learned Perturbation for Explainable Image Classification

A reproducible modernization of my **Computer Science Diploma Thesis (December 2023)** on **Explainable Artificial Intelligence (XAI)** for image classifiers.

The original work investigated whether a learnable network placed before a trained image classifier could produce a **pixel-level relevance mask** while preserving the classifier's ability to predict the correct class. The experiments used VGG19-based image classification and Food-101-derived subsets.

> This repository modernization preserves the research method. It improves reproducibility, code structure, checkpointing, and experimental controls without presenting new results as if they belonged to the original study.


## Historical context and research continuity

This project originated as my **Computer Science Diploma Thesis**, completed in **December 2023**, under the title *Proposal of a new explainability method to interpret the results of trained models for image classification*.

The original thesis must be read in its **2023 methodological and technological context**. The field of explainable AI has continued to evolve since then, including the development and refinement of perturbation-based methods, attribution techniques, quantitative faithfulness metrics, robustness criteria, and broader evaluation frameworks.

The present repository has two distinct purposes:

1. **Preserve the original research artifact and method.** The thesis, experimental design, original conclusions, and historical implementation remain identifiable as the 2023 work.
2. **Modernize the implementation for reproducibility.** The current codebase improves project structure, experiment controls, checkpointing, dataset handling, and testing. These engineering changes are not presented as new scientific validation of the original method.

### What the 2023 thesis concluded

The thesis introduced **VXN (Vision's eXplainer Network)**, a local, model-dependent explanation approach in which a learned neural network generates pixel-level relevance information for a trained image classifier.

The original experiments reported:

- a first experimental demonstration that the explainer could learn non-null relevance maps;
- greater visual detail in the reduced two-class case study than in the broader class configurations;
- evidence that reducing the number of classes and changing architectural/training parameters could improve the resulting explanations;
- sensitivity to the dataset and substantial computational cost;
- the need for further experimentation before making stronger claims about explanation quality.

These findings should therefore be interpreted as **experimental-stage results**, not as evidence that VXN is superior to established or current XAI methods.

### A meaningful continuation today

A contemporary continuation should not simply reproduce the 2023 experiments. It should **re-evaluate the central hypothesis under current XAI methodology**. A rigorous next research phase would include:

- comparison with established and contemporary XAI baselines;
- quantitative faithfulness and explanation-quality metrics in addition to visual inspection;
- robustness and sensitivity analysis;
- evaluation across multiple image models and datasets;
- explicit treatment of selection effects introduced by evaluating only correctly classified samples;
- systematic hyperparameter and perturbation-function studies;
- multiple random seeds and uncertainty reporting;
- computational-cost and scalability analysis.

This distinction is intentional: the repository preserves a genuine undergraduate research contribution while making clear which claims belong to the **original 2023 thesis** and which questions remain open for a future research continuation.

## Research idea

```mermaid
flowchart LR
    A[Input image] --> B[Learned explainer]
    B --> C[Pixel relevance mask]
    A --> D[Element-wise perturbation]
    C --> D
    D --> E[Frozen classifier]
    E --> F[Class prediction]
```

For an input image `x`, the explainer learns a mask `m(x)` constrained to `[0, 1]` by a sigmoid output. The classifier receives the perturbed image:

```text
x' = x ⊙ m(x)
```

Training updates the **explainer only**. The classifier is kept frozen.

## What is preserved from the original research

- VGG19-BN as the image-classification backbone.
- A learned pre-classifier perturbation stage.
- The original dense pixel-mask explainer:
  - flatten `3 × 224 × 224`
  - fully connected hidden layer
  - sigmoid relevance mask
  - element-wise image perturbation
- Experiments on reduced Food-101-derived class sets.
- The option to train the explainer only on samples already classified correctly.

The original thesis PDF is preserved in [`docs/Proposal of a new explainability method.pdf`](docs/Proposal%20of%20a%20new%20explainability%20method.pdf).

## What is modernized

### Reproducible project structure

The original implementation mixed dataset preparation, training, filtering, visualization, Google Drive paths, and model serialization in one Colab-derived Python file. These responsibilities are now separated.

```text
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
├── pyproject.toml
└── README.md
```

### Current TorchVision model API

The classifier uses the weights API rather than the deprecated `pretrained=True` argument:

```python
weights = VGG19_BN_Weights.DEFAULT
model = vgg19_bn(weights=weights)
```

### Portable checkpoints

Instead of pickling complete model objects, checkpoints contain a `state_dict` and experiment metadata. Loading uses `weights_only=True`.

### Deterministic dataset splitting

Train/evaluation/test splits accept an explicit random seed rather than depending on the state of a Colab session.

### Non-destructive filtering

The original workflow physically removed or exported dataset examples during some preparation steps. The new implementation represents correctly-classified examples as a `Subset`, leaving the underlying dataset unchanged.

### Frozen classifier behavior

The classifier is not only excluded from the optimizer: its parameters have `requires_grad=False` and it remains in evaluation mode while the explainer trains. This prevents classifier BatchNorm/Dropout state from drifting during explanation training.

## Important methodological note: correctly classified subsets

The original study included experiments where the explainer was trained/evaluated on samples the classifier already predicted correctly. This can be useful when asking:

> “What information does this already-correct classifier rely on?”

However, selecting data based on classifier correctness changes the evaluated population and can introduce **selection bias**. The modernization therefore makes this behavior explicit:

```bash
python scripts/train_explainer.py DATASET CLASSIFIER.pt --correct-only
```

It is **off by default** and should be reported whenever used.

## Preprocessing

Two preprocessing modes are intentionally kept distinct:

- `legacy` — resize, center crop, tensor conversion; reproduces the original experimental code.
- `vgg` — preprocessing associated with the pretrained VGG19-BN weights.

For strict comparison with the original experiments, use `legacy`. For new experiments, preprocessing must be declared as part of the experimental configuration.

## Setup

Requirements: Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows

pip install -e ".[dev]"
```

## Train a classifier

The dataset is expected in TorchVision `ImageFolder` layout:

```text
dataset/
├── class_a/
├── class_b/
└── ...
```

Then:

```bash
python scripts/train_classifier.py dataset/ \
  --output checkpoints/classifier.pt \
  --epochs 5 \
  --preprocessing legacy
```

## Train the explainer

```bash
python scripts/train_explainer.py \
  dataset/ \
  checkpoints/classifier.pt \
  --output checkpoints/explainer.pt \
  --epochs 4
```

To reproduce the correctly-classified-only experimental condition:

```bash
python scripts/train_explainer.py \
  dataset/ \
  checkpoints/classifier.pt \
  --correct-only
```

## Quality checks

```bash
ruff check .
pytest -q
```

Tests use tiny synthetic tensors and do **not** download VGG19 weights or Food-101.

## Research status

The **2023 thesis** reported experimental-stage consistency in the generated relevance maps and identified concrete directions for improving explanation quality. Its final conclusions describe VXN as a local, model-dependent neural explanation approach, sensitive to the data and computationally expensive, with the strongest qualitative results appearing in the reduced two-class experiment.

The thesis itself recommended additional hyperparameter search, evaluation of alternative perturbation functions, image-transformation experiments, and application of the method to other image-classification models.

The modernization does **not** fabricate, upgrade, or reinterpret those historical results as contemporary evidence. The next research pass should add:

- machine-readable experiment configurations;
- reconstruction of the exact class subsets used in the thesis;
- saved original metrics where they can be verified from source material;
- quantitative explanation metrics (for example deletion/insertion-style evaluation);
- comparison against established and current XAI baselines;
- representative relevance-map figures;
- multiple random seeds and uncertainty reporting;
- robustness and sensitivity evaluation;
- hardware/runtime and computational-cost documentation.

## Original research artifact

The thesis PDF, **Proposal of a new explainability method**, is preserved in [`docs/`](docs/) alongside the maintained research implementation.
