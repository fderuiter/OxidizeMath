Preallocated zero-allocation workspace for higher-dimensional cut-and-project quasicrystal generation:
$$\Lambda_{\parallel}(\boldsymbol{\phi}_\perp) = \bigl\{\,\pi_\parallel(\mathbf{n}) \;\big|\; \mathbf{n} \in \mathbb{Z}^D,\; |\pi_\parallel(\mathbf{n})| \le R_\parallel,\; \pi_\perp(\mathbf{n}) - \boldsymbol{\phi}_\perp \in W\,\bigr\}$$

Reference: Shechtman et al. (1984), DOI: `10.1103/PhysRevLett.53.1951`, [`shechtman1984metallic`](../../../papers/references.bib#shechtman1984metallic);
Levine & Steinhardt (1984), DOI: `10.1103/PhysRevLett.53.2477`, [`levine1984quasicrystals`](../../../papers/references.bib#levine1984quasicrystals);
Duneau & Katz (1985), DOI: `10.1103/PhysRevLett.54.2688`, [`duneau1985quasiperiodic`](../../../papers/references.bib#duneau1985quasiperiodic);
[`docs/math_specs/domain_physics.md`](../domain_physics.md).

# Stability
Evaluates orthogonal parallel and perpendicular lattice projections over bounded integer hypercubes and polygonal half-space windows with $10^{-12}$ boundary tolerance, allocating `0` heap bytes during `generate_into`.

# Domain of Validity
Clamps physical aperture radius $R_\parallel \in \[1.0, 6.0\]$, window scale $s_W \in \[0.5, 1.4\]$, and perpendicular phason shifts $\phi_{\perp,i} \in \[-2.0, 2.0\]$ within preallocated `MAX_QUASICRYSTAL_SITES` and `MAX_QUASICRYSTAL_EDGES` capacities.
