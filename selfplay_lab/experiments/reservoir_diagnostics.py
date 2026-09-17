"""How much does the reservoir actually do, and how much does its wiring matter?

Reported alongside the null experiment because a null result needs both halves to
be honest. `connectome_null.py` answers "is there a difference between the
measured wiring and a rewiring of it". This answers "was there a difference
available to find" - how much of the feature variance is nonlinear at all, and
how strongly the features encode which wiring produced them.

Two numbers carry it:

  linear R^2        fraction of feature variance explained by an affine function
                    of the input. Near 1.0 means the reservoir is effectively a
                    linear projection and the recurrence is decorative.
  corr(real, null)  correlation between features from the measured wiring and
                    features from a rewiring of it, on identical inputs. Near
                    1.0 means the features barely distinguish the two, so no
                    downstream learner could either, whatever its capacity.

Together they bound what the null experiment could have detected. A null result
from an operating point where corr(real, rewired) is 0.94 is a weaker statement
than the same null from one where it is 0.5, and the difference should be visible
rather than buried.

Usage:
  python -m selfplay_lab.experiments.reservoir_diagnostics
"""

import argparse
import json

import numpy as np
import torch

from ..flybrain import FlyReservoir, apply_null, load_extract
from ..train import DEFAULT_EXTRACT

# Leduc information states are sparse and mostly binary. Uniform noise would
# flatter the reservoir by driving every neuron at once, so the probe inputs are
# sparse binary at a density close to a real information state's.
PROBE_DENSITY = 0.15


def linear_r2(features, inputs):
    """Fraction of feature variance an affine function of the input explains."""
    design = np.concatenate([inputs, np.ones((len(inputs), 1))], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, features, rcond=None)
    residual = features - design @ coefficients
    centred = features - features.mean(axis=0)
    return 1.0 - float((residual**2).sum() / (centred**2).sum())


def effective_rank(features, tolerance=1e-3):
    spectrum = np.linalg.svd(features - features.mean(axis=0), compute_uv=False)
    return int((spectrum > spectrum[0] * tolerance).sum())


def probe(extract, input_dim, *, steps, gain, seed, samples, null="rewired"):
    generator = torch.Generator().manual_seed(seed)
    x = (torch.rand(samples, input_dim, generator=generator) < PROBE_DENSITY).float()

    def features_for(name):
        weights, _ = apply_null(extract, name, seed=seed)
        reservoir = FlyReservoir(
            weights, input_dim=input_dim, seed=seed, steps=steps, gain=gain
        )
        with torch.no_grad():
            return reservoir(x).numpy()

    real = features_for("real")
    other = features_for(null)
    return {
        "steps": steps,
        "gain": gain,
        "linear_r2": round(linear_r2(real, x.numpy()), 4),
        f"corr_real_vs_{null}": round(
            float(np.corrcoef(real.ravel(), other.ravel())[0, 1]), 4
        ),
        "effective_rank": effective_rank(real),
        "n_neurons": real.shape[1],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extract", default=DEFAULT_EXTRACT)
    p.add_argument("--input-dim", type=int, default=47, help="Leduc 3p information state")
    p.add_argument("--samples", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--null", default="rewired")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    extract = load_extract(args.extract)
    configs = [(1, 4.0), (12, 4.0), (24, 4.0), (12, 8.0), (40, 8.0)]
    rows = [
        probe(extract, args.input_dim, steps=s, gain=g, seed=args.seed,
              samples=args.samples, null=args.null)
        for s, g in configs
    ]

    key = f"corr_real_vs_{args.null}"
    print(f"{'steps':>6}{'gain':>6}{'linear R2':>12}{'corr(real,' + args.null + ')':>24}{'eff.rank':>10}")
    for row in rows:
        marker = "   <- used by connectome_null.py" if (row["steps"], row["gain"]) == (12, 4.0) else ""
        print(f"{row['steps']:>6}{row['gain']:>6.1f}{row['linear_r2']:>12.4f}"
              f"{row[key]:>24.4f}{row['effective_rank']:>10}{marker}")
    print("\nlinear R2 near 1.0: the reservoir is an affine map of its input.")
    print(f"corr near 1.0: the features barely encode WHICH wiring produced them,")
    print("so a null result at that operating point is a weak statement.")
    print("steps=1 applies the recurrent matrix zero times, so every wiring is identical there.")

    if args.out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({"config": vars(args), "probes": rows}, handle, indent=2)
        print(f"\nwrote {args.out}")
    return rows


if __name__ == "__main__":
    main()
