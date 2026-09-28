import argparse
from pathlib import Path

from xai_perturbation.checkpoints import load_state_dict, save_state_dict
from xai_perturbation.data import (
    correctly_classified_subset,
    imagefolder_splits,
    legacy_transform,
    make_loader,
    vgg_transform,
)
from xai_perturbation.models import (
    DensePixelMaskExplainer,
    PerturbationClassifier,
    build_vgg19_classifier,
)
from xai_perturbation.training import choose_device, train_explainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the learned perturbation explainer.")
    parser.add_argument("data", type=Path, help="ImageFolder dataset root")
    parser.add_argument("classifier", type=Path, help="Classifier state-dict checkpoint")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/explainer.pt"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--correct-only",
        action="store_true",
        help=(
            "Reproduce the thesis experiment that trains on examples the classifier "
            "already predicts correctly. This introduces selection bias."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device()

    metadata = {}
    checkpoint_metadata = None
    classifier_classes = None

    # Read metadata after constructing a classifier with the class count inferred from the dataset.
    provisional_transform = legacy_transform()
    train_set, _, _ = imagefolder_splits(
        args.data,
        transform=provisional_transform,
        seed=args.seed,
    )
    classifier_classes = train_set.dataset.classes

    classifier = build_vgg19_classifier(len(classifier_classes), pretrained=False)
    checkpoint_metadata = load_state_dict(classifier, args.classifier, map_location=device)
    preprocessing = checkpoint_metadata.get("preprocessing", "legacy")

    if preprocessing == "vgg":
        train_set, _, _ = imagefolder_splits(
            args.data,
            transform=vgg_transform(),
            seed=args.seed,
        )

    classifier.to(device)
    if args.correct_only:
        train_set = correctly_classified_subset(
            classifier,
            train_set,
            batch_size=args.batch_size,
            device=device,
        )

    explainer = DensePixelMaskExplainer(hidden_dim=args.hidden_dim)
    model = PerturbationClassifier(explainer, classifier)
    loader = make_loader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )
    history = train_explainer(
        model,
        loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        device=device,
    )

    metadata.update(
        {
            "architecture": "dense-pixel-mask",
            "hidden_dim": args.hidden_dim,
            "correct_only": args.correct_only,
            "classifier_checkpoint": str(args.classifier),
            "classifier_metadata": checkpoint_metadata,
        }
    )
    save_state_dict(explainer, args.output, metadata=metadata)
    final = history[-1]
    print(
        f"Saved {args.output} | epoch={final.epoch} "
        f"loss={final.loss:.4f} accuracy={final.accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
