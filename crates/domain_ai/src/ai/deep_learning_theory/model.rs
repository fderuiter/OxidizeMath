use super::calculus::{linear_backward, relu, relu_prime};
use super::linear_algebra::{DenseLayer, Vector};
use crate::ai::optimization::{Optimizer, ParamType};

pub use pure_math::pure_math::analysis::optimization::Trainable;

use verified_engine::Theory;

/// Two-layer feedforward multi-layer perceptron (MLP) with rectified linear hidden units.
///
/// Evaluates the affine-ReLU-affine map:
///
/// $$\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1, \qquad \mathbf{a}_1 = \max(\mathbf{0}, \mathbf{z}_1), \qquad \mathbf{z}_2 = \mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2$$
///
/// Reference: Rumelhart, Hinton & Williams (1986), DOI: 10.1038/323533a0; see
/// [`docs/math_specs/domain_ai.md`](../../../../../docs/math_specs/domain_ai.md).
///
/// # Stability
/// Backpropagation propagates exact subgradients $\mathbb{I}[\mathbf{z}_1 > 0]$; step sizes must
/// remain bounded ($\eta \ll 2 / \|\mathbf{W}\|_2^2$) to avoid activation saturation or divergence.
///
/// # Domain of Validity
/// Finite real input vectors $\mathbf{x} \in \mathbb{R}^{d_{\mathrm{in}}}$ matching
/// `layer1.weights.ncols()`, with positive layer dimensions $(d_{\mathrm{in}}, d_{\mathrm{hid}}, d_{\mathrm{out}})$.
#[derive(Theory)]
#[theory(
    description = "A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural network consisting of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function.",
    citation = "Learning representations by back-propagating errors (Rumelhart et al., 1986)"
)]
pub struct TwoLayerMLP {
    /// First dense affine layer mapping $\mathbb{R}^{d_{\mathrm{in}}} \to \mathbb{R}^{d_{\mathrm{hid}}}$.
    pub layer1: DenseLayer,
    /// Second dense affine layer mapping $\mathbb{R}^{d_{\mathrm{hid}}} \to \mathbb{R}^{d_{\mathrm{out}}}$.
    pub layer2: DenseLayer,
}

impl TwoLayerMLP {
    /// Constructs a two-layer MLP initialized from the default uniform random generator.
    #[verified_engine::verified]
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            layer1: DenseLayer::new(input_dim, hidden_dim),
            layer2: DenseLayer::new(hidden_dim, output_dim),
        }
    }

    /// Constructs a two-layer MLP initialized deterministically using `rng`.
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
