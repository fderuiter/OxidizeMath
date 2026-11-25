First-order iterative parameter update interface $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \Delta \boldsymbol{\theta}_t(\mathbf{g}_t)$.

Reference: Robbins & Monro (1951), DOI: 10.1214/aoms/1177729586;
[`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Implementations maintain bounded parameter updates for finite gradients $\mathbf{g}_t$ when the learning rate $\eta$ satisfies the objective's smoothness bound $\eta < 2 / L$.

# Domain of Validity
Requires matching finite parameter and gradient vector dimensions and strictly positive step sizes $\eta > 0$.
