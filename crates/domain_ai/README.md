# Domain AI

Neural architectures, attention mechanisms, reinforcement learning, and generative modeling.

## 1. Overview & Architecture

The `domain_ai` crate implements foundational Artificial Intelligence and Deep Learning algorithms from mathematical first principles. By leveraging pure Rust linear algebra routines, the crate exposes the transparent mathematical operations of modern AI architectures without reliance on opaque external neural network frameworks.

The crate enables educational exploration, deterministic unit verification, and interactive parameter manipulation of state-of-the-art machine learning models across native and web platforms.

### Module Structure

- **`activations`**: Vectorized activation functions (ReLU, Sigmoid, Softmax) optimized for matrix computation.
- **`transformer`**: Attention mechanisms, multi-head projections, positional encoding, and Transformer encoder/decoder stacks.
- **`optimization`**: First-order numerical optimization algorithms, including Stochastic Gradient Descent (SGD) and Adam.
- **`deep_learning_theory`**: Theoretical mechanics of backpropagation, loss surfaces, and universal approximation dynamics.
- **`reinforcement_learning`**: Decision-making agents implementing Q-learning, policy gradients, and Bellman value iterations.
- **`gaussian_splatting`**: Mathematical foundations of 3D Gaussian Splatting and differentiable rasterization.
- **`sds`**: Score Distillation Sampling bridging 2D diffusion priors with 3D representations.
- **`self_calibration`**: Conformal prediction and uncertainty estimation algorithms.

---

## 2. Developer Instructions

### Applying Neural Activations and Softmax

Process tensor representations using vectorized activation and normalization functions:

```rust
use domain_ai::ai::activations::{relu, softmax_row_wise};
use nalgebra::DMatrix;

// Apply the Rectified Linear Unit (ReLU) element-wise in-place
let mut activations = DMatrix::from_vec(2, 2, vec![-1.5, 2.0, 0.0, -0.5]);
relu(&mut activations);
assert_eq!(activations[(0, 0)], 0.0);
assert_eq!(activations[(1, 0)], 2.0);
assert_eq!(activations[(0, 1)], 0.0);
assert_eq!(activations[(1, 1)], 0.0);

// Compute numerically stable row-wise softmax distributions
let logits = DMatrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
let probabilities = softmax_row_wise(&logits);

let total_probability: f64 = probabilities.iter().sum();
assert!((total_probability - 1.0).abs() < 1e-6);
assert!(probabilities[(0, 2)] > probabilities[(0, 1)]);
assert!(probabilities[(0, 1)] > probabilities[(0, 0)]);
```

---

## 3. Ecosystem Integration

`domain_ai` modules are unified under `math_explorer` to power interactive visualization tabs in `math_explorer_gui`, allowing real-time visualization of attention heads, loss trajectories, and agent policies.
