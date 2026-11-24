use super::calculus::{linear_backward, relu, relu_prime};
use super::linear_algebra::{DenseLayer, Vector};
use crate::ai::optimization::{Optimizer, ParamType};

pub use pure_math::pure_math::analysis::optimization::Trainable;

use verified_engine::Theory;

/// Two-layer Multi-Layer Perceptron evaluating $f(x) = W_2\,\text{ReLU}(W_1 x + b_1) + b_2$ (`@cybenko1989approximation`, `spec:transformer_v1`).
///
/// - **Numerical stability**: Xavier/He-scaled dense initialization bounds pre-activation variance; gradient updates remain stable for learning rates $\eta < 2 / \|W_1\|_2 \|W_2\|_2$.
/// - **Domain of validity**: Requires finite input vectors $x \in \mathbb{R}^{d_{\text{in}}}$ with `layer1` shape $(d_{\text{hid}}, d_{\text{in}})$ and `layer2` shape $(d_{\text{out}}, d_{\text{hid}})$ ($d_{\text{in}}, d_{\text{hid}}, d_{\text{out}} \ge 1$).
#[derive(Theory)]
#[theory(
    description = "A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural network consisting of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function.",
    citation = "cybenko1989approximation"
)]
pub struct TwoLayerMLP {
    #[allow(missing_docs)]
    pub layer1: DenseLayer,
    #[allow(missing_docs)]
    pub layer2: DenseLayer,
}

impl TwoLayerMLP {
    /// Initializes a two-layer network $f(x) = W_2\,\text{ReLU}(W_1 x + b_1) + b_2$ using the deterministic workspace RNG (`@cybenko1989approximation`, `spec:transformer_v1`).
    /// Numerical stability: variance-scaled weight initialization prevents forward explosion. Domain of validity: positive dimensions `input_dim >= 1`, `hidden_dim >= 1`, `output_dim >= 1`.
    #[verified_engine::verified]
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            layer1: DenseLayer::new(input_dim, hidden_dim),
            layer2: DenseLayer::new(hidden_dim, output_dim),
        }
    }

    /// Initializes $f(x) = W_2\,\text{ReLU}(W_1 x + b_1) + b_2$ from caller-supplied `rng` (`@cybenko1989approximation`, `spec:transformer_v1`).
    /// Numerical stability: bounded uniform/Gaussian fan-in scaling preserves $O(1)$ layer activations. Domain of validity: positive dimensions `input_dim >= 1`, `hidden_dim >= 1`, `output_dim >= 1`.
    #[verified_engine::verified]
    pub fn new_with_rng<R: rand::Rng + ?Sized>(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        rng: &mut R,
    ) -> Self {
        Self {
            layer1: DenseLayer::new_with_rng(input_dim, hidden_dim, rng),
            layer2: DenseLayer::new_with_rng(hidden_dim, output_dim, rng),
        }
    }
}

impl Trainable for TwoLayerMLP {
    #[verified_engine::verified]
    fn forward(&self, x: &Vector) -> Vector {
        let z1 = self.layer1.forward(x);
        let a1 = relu(&z1);
        self.layer2.forward(&a1)
    }

    #[verified_engine::verified]
    fn backward_update(
        &mut self,
        x: &Vector,
        loss_grad: &Vector,
        optimizer: &mut dyn Optimizer<f64>,
    ) -> Result<(), crate::ai::optimization::OptimizationError> {
        // --- Forward Re-computation (needed for gradients) ---
        // Ideally, we'd cache these, but re-computing is simpler for this educational example.
        let z1 = self.layer1.forward(x);
        let a1 = relu(&z1);
        // z2 is not needed explicitly since we have loss_grad (dL/dz2) passed in.

        // --- Backward Pass ---
        // Backprop through Layer 2
        // dL/dW2, dL/db2, dL/da1
        // loss_grad corresponds to d_z2
        let (d_a1, d_w2, d_b2) = linear_backward(loss_grad, &a1, &self.layer2.weights);

        // Backprop through ReLU
        // dL/dz1 = dL/da1 * ReLU'(z1)
        let d_z1 = d_a1.component_mul(&relu_prime(&z1));

        // Backprop through Layer 1
        // dL/dW1, dL/db1, dL/dx
        let (_, d_w1, d_b1) = linear_backward(&d_z1, x, &self.layer1.weights);

        // --- Update ---
        // Use legacy update methods provided by the Optimizer facade
        // Layer 2
        optimizer.update_matrix((2, ParamType::Weight), &mut self.layer2.weights, &d_w2)?;
        optimizer.update_vector((2, ParamType::Bias), &mut self.layer2.bias, &d_b2)?;

        // Layer 1
        optimizer.update_matrix((1, ParamType::Weight), &mut self.layer1.weights, &d_w1)?;
        optimizer.update_vector((1, ParamType::Bias), &mut self.layer1.bias, &d_b1)?;

        Ok(())
    }
}
