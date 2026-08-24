"""Train or warm-start the scalable stationary tree model."""

from __future__ import annotations

import argparse

from deep_macrofin_jax import PDETrainer, TrainingConfig, load_checkpoint
from deep_macrofin_jax.models import TreeConfig, TreeModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint", default="checkpoints/tree.npz")
    parser.add_argument("--warm-start", default=None)
    args = parser.parse_args()

    model = TreeModel(TreeConfig(n_trees=args.trees))
    trainer = PDETrainer(
        model,
        TrainingConfig(steps=args.steps, batch_size=args.batch_size),
    )
    params = None
    if args.warm_start:
        params, metadata = load_checkpoint(args.warm_start)
        if metadata.get("model", {}).get("n_trees") != args.trees:
            raise ValueError("warm-start checkpoint has a different number of trees")
    result = trainer.fit(params=params, checkpoint_path=args.checkpoint)
    print(result.history[-1])
    print(f"saved {result.checkpoint_path}")


if __name__ == "__main__":
    main()
