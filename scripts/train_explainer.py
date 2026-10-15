import argparse
from pathlib import Path

from xai_perturbation.checkpoints import load_state_dict, save_state_dict
from xai_perturbation.data import (
    correctly_classified_subset,
    imagefolder_splits,
    make_loader,
    thesis_transform,
)
from xai_perturbation.models import (
    DensePixelMaskExplainer,
    PerturbationClassifier,
    build_vgg19_classifier,
)
from xai_perturbation.training import choose_device, train_explainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the learned perturbation explainer stage from the 2023 thesis."
    )
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
            "Use only examples the classifier predicts correctly, reproducing the "
            "filtered-data condition used in the thesis experiments."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional split seed. Omit it to retain the notebook's unseeded random_split "
            "behavior."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device()

    train_set, _, _ = imagefolder_splits(
        args.data,
        transform=thesis_transform(),
        seed=args.seed,
    )
    classifier_classes = train_set.dataset.classes

    classifier = build_vgg19_classifier(len(classifier_classes), pretrained=False)
    checkpoint_metadata = load_state_dict(classifier, args.classifier, map_location=device)
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

    save_state_dict(
        explainer,
        args.output,
        metadata={
            "architecture": "dense-pixel-mask",
            "hidden_dim": args.hidden_dim,
            "correct_only": args.correct_only,
            "classifier_checkpoint": str(args.classifier),
            "classifier_metadata": checkpoint_metadata,
            "split_seed": args.seed,
        },
    )
    final = history[-1]
    print(
        f"Saved {args.output} | epoch={final.epoch} "
        f"loss={final.loss:.4f} accuracy={final.accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
