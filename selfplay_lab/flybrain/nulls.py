"""Null models: the controls that decide whether the connectome did anything.

A reservoir of 449 recurrently connected nonlinear units will produce a useful
nonlinear expansion of its input whether or not its wiring came from an animal.
So "the fly-brain agent learned to play" is not a result. The result, if there is
one, is the *difference* between the measured wiring and a null that preserves
everything about it except the thing under test.

Each null holds a different property fixed, so a difference localises:

  rewired         same in/out degree per neuron, same signs, same weight
                  multiset - only which neuron connects to which is destroyed.
                  THIS IS THE PRIMARY CONTROL.
  shuffled_signs  same topology, weights and set of silent (unsigned) neurons;
                  which cells are inhibitory is permuted among the signed ones.
                  Tests whether excitation/inhibition placement matters.
  shuffled_weights same topology and signs; weight values permuted across edges.
                  Tests whether the graded synapse counts matter at all.
  random_matched  same neuron count, edge count and weight distribution, but an
                  Erdos-Renyi topology. The weakest null and the easiest to beat.

If the measured wiring does not beat `rewired`, the honest report is that the
connectome contributed nothing beyond its degree sequence.
"""

import numpy as np

# Double-edge swaps attempted per existing edge. 10 is the usual rule of thumb
# for mixing a configuration-model rewiring; below ~5 the result still correlates
# visibly with the original edge list.
SWAPS_PER_EDGE = 10


def rewire_degree_preserving(extract, rng, *, swaps_per_edge=SWAPS_PER_EDGE):
    """Degree-preserving double-edge swap on the directed edge list.

    Repeatedly picks two edges (a->b) and (c->d) and rewrites them as (a->d) and
    (c->b). Each swap preserves the out-degree of every neuron and the in-degree
    of every neuron exactly. Weights travel with their presynaptic neuron, so
    every neuron's total output strength is preserved too.

    Signs need no special handling: a sign is a property of the presynaptic
    neuron, and a swap never changes which neuron an edge leaves from.
    """
    pre = extract.edges_pre.copy()
    post = extract.edges_post.copy()
    weight = extract.edges_weight.copy()

    present = set(zip(pre.tolist(), post.tolist()))
    n_edges = len(pre)
    attempts = int(swaps_per_edge * n_edges)
    accepted = 0

    first = rng.integers(0, n_edges, size=attempts)
    second = rng.integers(0, n_edges, size=attempts)
    for i, j in zip(first.tolist(), second.tolist()):
        if i == j:
            continue
        a, b = pre[i], post[i]
        c, d = pre[j], post[j]
        if a == d or c == b:
            continue  # would create a self-loop
        if (a, d) in present or (c, b) in present:
            continue  # would create a duplicate edge
        present.discard((a, b))
        present.discard((c, d))
        present.add((a, d))
        present.add((c, b))
        post[i], post[j] = d, b
        accepted += 1

    return pre, post, weight, {"attempted": attempts, "accepted": accepted}


def _matrix_from_edges(n, pre, post, weight, signs, *, normalise=True):
    raw = np.zeros((n, n), dtype=np.float64)
    np.add.at(raw, (post, pre), weight)
    weights = raw * np.nan_to_num(signs, nan=0.0)[np.newaxis, :]
    if normalise:
        row_sums = np.abs(weights).sum(axis=1)
        safe = np.where(row_sums > 0, row_sums, 1.0)
        weights = weights / safe[:, np.newaxis]
    return weights


def apply_null(extract, name, seed, *, normalise=True):
    """Return (weight_matrix, info) for the named null model.

    `name="real"` returns the measured wiring untouched, so that every arm of an
    experiment goes through exactly the same code path and any difference is
    attributable to the matrix rather than to the plumbing.
    """
    rng = np.random.default_rng(seed)
    n = extract.n_neurons
    pre, post, weight = extract.edges_pre, extract.edges_post, extract.edges_weight
    signs = extract.signs
    info = {"null": name, "seed": seed}

    if name == "real":
        pass
    elif name == "rewired":
        pre, post, weight, stats = rewire_degree_preserving(extract, rng)
        info.update(stats)
    elif name == "shuffled_signs":
        # Permute signs only among neurons that HAVE a sign. Permuting the
        # unknowns too would move which neurons are silent, changing the number
        # of nonzero weights and confounding the control with a density change.
        signs = signs.copy()
        known = np.flatnonzero(~np.isnan(signs))
        signs[known] = rng.permutation(signs[known])
    elif name == "shuffled_weights":
        weight = rng.permutation(weight)
    elif name == "random_matched":
        n_edges = len(pre)
        chosen = set()
        new_pre, new_post = [], []
        while len(chosen) < n_edges:
            a = int(rng.integers(0, n))
            b = int(rng.integers(0, n))
            if a == b or (a, b) in chosen:
                continue
            chosen.add((a, b))
            new_pre.append(a)
            new_post.append(b)
        pre = np.array(new_pre, dtype=np.int64)
        post = np.array(new_post, dtype=np.int64)
        weight = rng.permutation(weight)
    else:
        raise ValueError(f"unknown null model {name!r}; known: {sorted(NULL_MODELS)}")

    matrix = _matrix_from_edges(n, pre, post, weight, signs, normalise=normalise)
    return matrix, info


NULL_MODELS = ("real", "rewired", "shuffled_signs", "shuffled_weights", "random_matched")
