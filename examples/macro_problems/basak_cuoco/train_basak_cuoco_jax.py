"""Train or warm-start the Basak-Cuoco model."""

from __future__ import annotations

import argparse

from deep_macrofin_jax import PDETrainer, TrainingConfig, load_checkpoint
from deep_macrofin_jax.models import BasakCuocoModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint", default="checkpoints/basak_cuoco.npz")
    parser.add_argument("--warm-start", default=None)
    args = parser.parse_args()

    model = BasakCuocoModel()
    trainer = PDETrainer(
        model,
        TrainingConfig(steps=args.steps, batch_size=args.batch_size),
    )
    params = load_checkpoint(args.warm_start)[0] if args.warm_start else None
    result = trainer.fit(params=params, checkpoint_path=args.checkpoint)
    print(result.history[-1])
    print(f"saved {result.checkpoint_path}")


if __name__ == "__main__":
    main()
