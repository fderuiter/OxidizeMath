First-order Stochastic Gradient Descent optimizer:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{g}_t
$$

Reference: Robbins & Monro (1951), DOI: 10.1214/aoms/1177729586;
[`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Convergent on $L$-smooth convex objectives for step sizes $0 < \eta < 2/L$; parameter updates execute in-place on contiguous slices without per-step heap allocation.

# Domain of Validity
Valid for positive learning rate $\eta > 0$ and finite parameter and gradient vectors or matrices of matching dimensions.

# Examples

```rust
# use pure_math::__doctest_facade as math_explorer;
use math_explorer::nalgebra::DVector;
use math_explorer::pure_math::analysis::optimization::{Optimizer, SGD};

let mut sgd = SGD::new(0.1_f64);
let mut params = DVector::from_vec(vec![2.0_f64, -1.0_f64]);
let grads = DVector::from_vec(vec![0.5_f64, -0.25_f64]);
sgd.update_vector(0_u64, &mut params, &grads).unwrap();
assert!((params[0] - 1.95).abs() < 1e-12);
```
