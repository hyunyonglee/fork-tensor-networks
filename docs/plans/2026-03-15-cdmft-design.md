# Cluster DMFT (CDMFT) Design

## Goal

Real-space CDMFT implementation using ForkTensorNetworks. Start with 2-site single-orbital cluster on Bethe lattice, then extend to 2×2 and multi-orbital.

## Approach

Reuse existing FTN structure: cluster sites map to orbital indices (N_orb = N_cluster_sites).
Create **new W tensors** for cluster hopping `t'` (single-particle hopping),
separate from the multi-orbital U'/J channels (two-particle interactions).

## Phase 1: 2-site single-orbital CDMFT (Bethe lattice)

### Model

```
H_cluster = ε₁n₁ + ε₂n₂ + U(n₁↑n₁↓ + n₂↑n₂↓) + t(c†₁c₂ + h.c.)
H_bath = Σₖ εₖ nₖ + Vₖ(c†₀cₖ + h.c.)  per cluster site
```

FTN mapping:
- N_orb = 2 (cluster site 1, cluster site 2)
- Backbone: [site1↑, site1↓, site2↑, site2↓] (Lx=4)
- Each arm: impurity(y=1) + bath(y=2,...,Ly)

### W tensor construction

**New W tensors** for CDMFT cluster hopping, not reusing multi-orbital U'/J tensors.

Cluster hopping `t'` is a single-particle operator:
- `t'·(c†_A↑ c_B↑ + h.c.)` and `t'·(c†_A↓ c_B↓ + h.c.)`
- Distinct from U'/J (two-particle) → requires dedicated backbone bond channels
- Backbone aux_x_idx carries c† and c operators for inter-cluster-site hopping

For 2-site (Lx=4, W_imp1 + W_imp3 only, no W_imp2):
- W_site1↑, W_site1↓: emit c†, c operators onto backbone bond
- W_site2↑, W_site2↓: receive and complete hopping terms
- On-site U on each cluster site (same as single-site AIM)

### Self-consistency

**Single-site DMFT** (current):
```
Δ(ω) = (D²/4) · G(ω)           scalar
Σ(ω) = G₀⁻¹(ω) - G⁻¹(ω)       scalar
```

**2-site CDMFT** (Bethe lattice):
```
G(ω)  → 2×2 matrix: G₁₁, G₁₂, G₂₁, G₂₂
Σ(ω)  → 2×2 matrix
Δ(ω)  → 2×2 matrix (hybridization matrix)
```

Bethe lattice self-consistency for cluster:
```
G⁻¹_cluster(ω) = (ω+μ)I - t_cluster - Σ(ω) - Δ(ω)
Δ_new(ω) = (D²/4) · G(ω)   (matrix version)
```

### Green's function computation

4 TDVP runs for all G_ij matrix elements (see `2026-03-15-cdmft-2site-formulation.md`):
- Run A: evolve |ψ⁺₁⟩ → G>₁₁, G>₂₁
- Run B: evolve |ψ⁺₂⟩ → G>₁₂, G>₂₂
- Run C: evolve |ψ⁻₁⟩ → G<₁₁, G<₂₁ (conj trick)
- Run D: evolve |ψ⁻₂⟩ → G<₁₂, G<₂₂ (conj trick)
- PH symmetric: skip C, D

ESPRIT analytic continuation applied independently to each matrix element (ν₁,ν₂).
Off-diagonal residues are complex-valued but Lehmann representation guarantees
sum-of-exponentials form → ESPRIT is valid.

### Implementation plan

1. **Hamiltonian**: New W tensors for cluster hopping t' (new file or extend AndersonImpurityModel.jl)
2. **Green's function**: 4 TDVP runs, matrix-valued G with cross-site overlaps
3. **ESPRIT**: Per-element 2-pass pipeline (same algorithm, complex residues for off-diagonal)
4. **Self-consistency**: Matrix Bethe self-consistency Δ_ij(ω) = D²/4 · G^R_ij(ω)
5. **Bath discretization**: Δ₁₁, Δ₂₂ → bath params (independent per cluster site)
6. **Validation**: Compare with Kotliar et al., PRL 87, 186401 (2001)

### File structure

All CDMFT work in existing `dmft/` folder (no separate folder):
```
dmft/
├── Bath.jl                (reuse as-is)
├── ESPRIT.jl              (reuse as-is)
├── GreensFunction.jl      (extend: matrix GF support)
├── DMFTLoop.jl            (extend: CDMFT mode)
├── solvers/
│   └── ComplexTimeSolver.jl  (extend: cross-site overlaps)
└── models/
    └── ClusterAIM.jl      (new: CDMFT W tensors with t' hopping)
```

### Config

```toml
[model]
N_orb = 1                # physical orbitals per cluster site
N_cluster = 2            # cluster sites
N_bath = 10              # bath sites per cluster site
U = 4.0
half_bandwidth = 2.0
t_cluster = 1.0          # intra-cluster hopping (configurable)

[cdmft]
off_diagonal_bath = "diagonal_only"  # or "full"
```

## Phase 2: 2×2 plaquette (4 cluster sites)

- N_orb = 4, Lx = 8
- Non-adjacent hopping needed (A↔C skips B on backbone)
- Backbone bond dimension increases for long-range hopping channels
- Self-consistency: 4×4 matrix equations
- Benchmark: Park et al., PRL 101, 186403 (2008)

## Phase 3: Multi-orbital CDMFT

- Cluster hopping t + Kanamori interaction (U, U', J) simultaneously
- Example: 2-site, 2-orbital → N_orb=4, same structure as 2×2 plaquette
- Self-consistency: (2×N_orb) × (2×N_orb) matrix equations

## Phase 4: Square lattice self-consistency

- Replace Bethe lattice Δ = D²/4 · G with k-space integration
- Σ(ω) from impurity solver → lattice G via Dyson equation with k-sum
- More expensive but physically relevant

## Key references

- Bauernfeind, "Fork Tensor-Product States" (Dissertation), Appendix C — W tensor construction
- Kotliar et al., PRL 87, 186401 (2001) — 2-site CDMFT benchmark
- Park et al., PRL 101, 186403 (2008) — 2×2 plaquette CDMFT
- Maier et al., RMP 77, 1027 (2005) — Cluster DMFT review
