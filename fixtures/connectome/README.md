# Connectome fixtures

## `ring_attractor.hemibrain-v1.2.json.gz`

The central-complex circuit of an adult *Drosophila melanogaster* brain: 449
neurons (EPG, PEN_a, PEN_b, PEG, Delta7, EL, and the ER/ExR ellipsoid-body ring
neurons) and the 63,417 weighted connections between them, with a resolved
fast-synaptic sign per neuron.

This is a **derived work**, not a copy: it is the subgraph induced on those cell
types, plus a sign per neuron resolved from published neurotransmitter
predictions. It is committed - unlike the 92 MB of sources it is built from - so
that the tests run against real measured wiring with no network access and no
optional dependencies.

Rebuild it with:

```bash
aigm brain fetch                  # downloads and verifies the sources into var/
aigm brain build ring_attractor
```

## Attribution

Derived from the **Janelia hemibrain v1.2** connectome, released under
**CC BY 4.0**.

> Scheffer, L. K., Xu, C. S., Januszewski, M., et al. (2020). A connectome and
> analysis of the adult *Drosophila* central brain. *eLife* 9:e57443.
> https://doi.org/10.7554/eLife.57443

Neurotransmitter predictions from the same release:

> Eckstein, N., Bates, A. S., Champion, A., et al. (2024). Neurotransmitter
> classification from electron microscopy images at synaptic sites in
> *Drosophila melanogaster*. *Cell* 187(10):2574-2594.
> https://doi.org/10.1016/j.cell.2024.03.016

Where those predictions are contradicted by directly measured cell identity, the
correction and its citation are recorded in
`backend/app/connectome/neurotransmitters.py`. See `docs/CONNECTOME_GM.md`
section 4 for which populations those are and how the threshold was chosen.
