use super::linear_algebra::{Matrix, Vector};
use crate::ai::activation_kernels as kernel;

/// ReLU (Rectified Linear Unit) activation function: f(z) = max(0, z).
/// Applied element-wise.
#[verified_engine::verified]
pub fn relu(z: &Vector) -> Vector {
    z.map(kernel::relu)
}

/// Derivative of ReLU.
/// f'(z) = 1 if z > 0, else 0.
#[verified_engine::verified]
pub fn relu_prime(z: &Vector) -> Vector {
    z.map(kernel::relu_prime)
}

/// Sigmoid activation function: \sigma(z) = 1 / (1 + e^{-z}).
#[verified_engine::verified]
pub fn sigmoid(z: &Vector) -> Vector {
    z.map(kernel::sigmoid)
}

/// Derivative of Sigmoid: \sigma'(z) = \sigma(z) * (1 - \sigma(z)).
#[verified_engine::verified]
pub fn sigmoid_prime(z: &Vector) -> Vector {
    z.map(kernel::sigmoid_prime)
}

/// Computes the gradients for a linear layer during backpropagation.
///
/// Given the gradient of the loss with respect to the output z (\frac{\partial L}{\partial z}),
/// compute the gradients with respect to inputs x, weights W, and bias b.
///
/// Returns tuple: (grad_x, grad_W, grad_b)
///
/// Derivations:
/// z = Wx + b
/// \frac{\partial L}{\partial x} = W^T \cdot \frac{\partial L}{\partial z}
/// \frac{\partial L}{\partial W} = \frac{\partial L}{\partial z} \cdot x^T
/// \frac{\partial L}{\partial b} = \frac{\partial L}{\partial z}
#[verified_engine::verified]
pub fn linear_backward(grad_z: &Vector, x: &Vector, w: &Matrix) -> (Vector, Matrix, Vector) {
    // grad_z is (output_dim)
    // x is (input_dim)
    // W is (output_dim, input_dim)

    // dL/dx = W^T * dL/dz
    let grad_x = w.transpose() * grad_z;

    // dL/dW = dL/dz * x^T (outer product)
    // In nalgebra, we can do this via gemm or explicit multiplication
    // grad_z is (out, 1), x is (in, 1). We want (out, in).
    // grad_z * x^T
    let grad_w = grad_z * x.transpose();

    // dL/db = dL/dz
    let grad_b = grad_z.clone();

    (grad_x, grad_w, grad_b)
}

/// Applies the chain rule for element-wise activation functions.
///
/// Given \frac{\partial L}{\partial a} (gradient w.r.t activation output),
/// and the pre-activation input z, computes \frac{\partial L}{\partial z}.
///
/// \frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \odot f'(z)
#[verified_engine::verified]
pub fn activation_backward<F>(grad_a: &Vector, z: &Vector, prime_fn: F) -> Vector
where
    F: Fn(&Vector) -> Vector,
{
    let dz = prime_fn(z);
    grad_a.component_mul(&dz) // Element-wise multiplication (Hadamard product)
}

/// Tanh activation function.
#[verified_engine::verified]
pub fn tanh(z: &Vector) -> Vector {
    z.map(kernel::tanh)
}

/// Derivative of Tanh: 1 - tanh^2(z).
#[verified_engine::verified]
pub fn tanh_prime(z: &Vector) -> Vector {
    z.map(kernel::tanh_prime)
}

/// GELU (Gaussian Error Linear Unit) activation function.
/// Approximation: 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
#[verified_engine::verified]
pub fn gelu(z: &Vector) -> Vector {
    z.map(kernel::gelu)
}

/// Derivative of GELU (Approximation).
#[verified_engine::verified]
pub fn gelu_prime(z: &Vector) -> Vector {
    z.map(kernel::gelu_prime)
}
