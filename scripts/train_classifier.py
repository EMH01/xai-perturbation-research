import argparse
from pathlib import Path

from xai_perturbation.checkpoints import save_state_dict
from xai_perturbation.data import imagefolder_splits, make_loader, thesis_transform
from xai_perturbation.models import build_vgg19_classifier
from xai_perturbation.training import choose_device, train_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the VGG19-BN classifier stage used in the 2023 thesis workflow."
    )
    parser.add_argument("data", type=Path, help="ImageFolder dataset root")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/classifier.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
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
    train_set, _, _ = imagefolder_splits(
        args.data,
        transform=thesis_transform(),
        seed=args.seed,
    )
    classes = train_set.dataset.classes
    loader = make_loader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )

    device = choose_device()
    model = build_vgg19_classifier(len(classes))
    history = train_classifier(
        model,
        loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        device=device,
    )
    save_state_dict(
        model,
        args.output,
        metadata={
            "classes": classes,
            "preprocessing": "thesis",
            "epochs": args.epochs,
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
