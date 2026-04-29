# EMI Shielding in Nb-Based MXenes

First-principles electromagnetic interference (EMI) shielding effectiveness of (Nb₀.₈M₀.₂)₄C₃Tₓ monolayers across the complete 3*d* transition-metal series (M = Sc–Zn).

**Associated publication:**
L. Arrieta, M. Gutiérrez, and J. I. Borge, "The Electronic Penalty of Magnetic Doping: Frequency-Resolved Shielding Across the Complete 3*d* Series in Ultrathin Nb-Based MXenes," *J. Mater. Chem. C* (2026).

## Workflow

The computational workflow proceeds in three steps:

1. **DFT** — Ground-state electronic structure with Quantum ESPRESSO (PBE + Hubbard *U*).
2. **KGEC** — Frequency-dependent optical conductivity σ(ω) via the Kubo–Greenwood formula.
3. **Thin-film SE** — Shielding effectiveness from the complex dielectric function using the Schulz relation.

## Repository structure

```
inputs/
  scf/          Quantum ESPRESSO self-consistent field inputs (one per dopant)
  nscf/         Non-self-consistent field inputs for DOS/bands
  kgec/         KGEC optical conductivity inputs
scripts/
  heatmap.py    Heatmap and statistical analysis (Fig. 2)
  compute_se.py σ(ω) → SE conversion
  plot_sigma.jl σ₁(ω) visualization (Julia/CairoMakie)
data/
  se_all_bands.csv       SE (%) and SE (dB) per dopant and microwave band
  conductivity/          Tabulated σ₁(ω) per composition
figures/
  heatmap.png    SE heatmap across all bands and dopants (Fig. 2)
```

## Requirements

- Python ≥ 3.9 (matplotlib, numpy, scipy)
- Julia ≥ 1.9 with CairoMakie (for σ₁ plots)
- Quantum ESPRESSO ≥ 7.2
- KGEC code ([Calderin et al., Comput. Phys. Commun. 2017](https://doi.org/10.1016/j.cpc.2017.01.017))

## License

MIT
