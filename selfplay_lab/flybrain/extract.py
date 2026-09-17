"""Read a connectome circuit extract.

The extract format is produced by `aigm brain build` in the ai-gm repository and
is documented in `fixtures/connectome/README.md`. It is deliberately a plain
gzipped JSON document rather than a pickle or a binary array format: the wiring
of a real animal's brain should be readable with `zcat | jq`.

This loader is intentionally a copy rather than a cross-repository import. A
research lab that depends on another repository's package to read a 138 KB data
file has bought a coupling it does not need.
"""

import gzip
import json
from dataclasses import dataclass

import numpy as np

EXTRACT_SCHEMA = "aigm.connectome.extract/1"


@dataclass
class CircuitExtract:
    circuit: str
    body_ids: np.ndarray
    cell_types: list
    signs: np.ndarray          # +1 excitatory, -1 inhibitory, 0 modulatory, NaN unknown
    edges_pre: np.ndarray
    edges_post: np.ndarray
    edges_weight: np.ndarray
    provenance: dict

    @property
    def n_neurons(self):
        return len(self.body_ids)

    @property
    def n_edges(self):
        return len(self.edges_pre)

    def indices_of_type(self, *prefixes):
        return np.array(
            [i for i, t in enumerate(self.cell_types) if any(t.startswith(p) for p in prefixes)],
            dtype=np.int64,
        )


def load_extract(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    schema = payload.get("schema")
    if schema != EXTRACT_SCHEMA:
        raise ValueError(f"unsupported extract schema {schema!r}, expected {EXTRACT_SCHEMA!r}")
    neurons = payload["neurons"]
    signs = np.array(
        [np.nan if n["sign"] is None else float(n["sign"]) for n in neurons],
        dtype=np.float64,
    )
    return CircuitExtract(
        circuit=payload["circuit"],
        body_ids=np.array([n["body_id"] for n in neurons], dtype=np.int64),
        cell_types=[n["cell_type"] for n in neurons],
        signs=signs,
        edges_pre=np.array(payload["edges"]["pre"], dtype=np.int64),
        edges_post=np.array(payload["edges"]["post"], dtype=np.int64),
        edges_weight=np.array(payload["edges"]["weight"], dtype=np.float64),
        provenance=payload.get("provenance", {}),
    )


def signed_matrix(extract, *, normalise=True):
    """Build the signed weight matrix W[post, pre].

    A neuron whose transmitter could not be determined has sign NaN and is
    treated as contributing nothing: we do not know what it would do, and
    assuming excitatory because that is the common case would be inventing data.
    Those neurons stay in the matrix as zero columns so that indices, readouts
    and null models all line up with the extract.

    Row normalisation divides each neuron's input by its own total input
    magnitude. The absolute synapse count a neuron receives depends on its size
    and on reconstruction completeness; the relative weighting of its inputs is
    what the connectome measures cleanly, and that is what survives.
    """
    n = extract.n_neurons
    raw = np.zeros((n, n), dtype=np.float64)
    np.add.at(raw, (extract.edges_post, extract.edges_pre), extract.edges_weight)
    signs = np.nan_to_num(extract.signs, nan=0.0)
    weights = raw * signs[np.newaxis, :]
    if normalise:
        row_sums = np.abs(weights).sum(axis=1)
        safe = np.where(row_sums > 0, row_sums, 1.0)
        weights = weights / safe[:, np.newaxis]
    return weights
