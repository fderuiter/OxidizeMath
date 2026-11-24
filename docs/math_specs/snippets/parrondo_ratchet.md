Computes the exact stationary capital drift $g = \sum_{i=0}^{M-1} \pi_i (p_i - q_i)$ of a modular Parrondo Markov chain.

Reference: Harmer & Abbott (1999), DOI: 10.1214/ss/1009212519;
[`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Solves the finite periodic stationary distribution $\boldsymbol{\pi} P = \boldsymbol{\pi}$, $\sum_i \pi_i = 1$ directly via constrained Householder QR rather than noisy Monte Carlo averaging.

# Domain of Validity
Valid for irreducible modular capital games ($M \in \{3, 5, 7, 9\}$) with transition probabilities $p_i \in (0, 1)$.
