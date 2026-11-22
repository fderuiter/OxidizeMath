use nalgebra::{DMatrix, DVector};
use rand::distributions::{Distribution, Uniform};

/// Represents a scalar value (x \in \mathbb{R}).
pub type Scalar = f64;

/// Represents a vector (\mathbf{x} \in \mathbb{R}^n).
/// A vector usually represents a single example (feature vector) in a dataset.
pub type Vector = DVector<f64>;

/// Represents a matrix (\mathbf{A} \in \mathbb{R}^{m \times n}).
/// In deep learning, weights connecting two layers are stored in a matrix.
pub type Matrix = DMatrix<f64>;

/// Computes the dot product of two vectors \mathbf{a} \cdot \mathbf{b} = \sum a_i b_i.
///
/// # Geometric Interpretation
/// The dot product is a measure of similarity. If a weight vector aligns with an input vector,
/// the activation is high. Geometrically, it relates to the cosine of the angle between them.
#[verified_engine::verified]
pub fn dot_product(a: &Vector, b: &Vector) -> Scalar {
    a.dot(b)
}

/// Performs a linear transformation: \mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}.
///
/// # Arguments
/// * `x` - The input vector (\mathbf{x}).
/// * `w` - The weight matrix (\mathbf{W}).
/// * `b` - The bias vector (\mathbf{b}).
///
/// # Returns
/// The pre-activation output (\mathbf{z}).
#[verified_engine::verified]
pub fn linear_transformation(x: &Vector, w: &Matrix, b: &Vector) -> Vector {
    // W is typically (output_dim, input_dim)
    // x is (input_dim)
    // b is (output_dim)
    (w * x) + b
}

/// A structure representing a dense (fully connected) layer's parameters.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DenseLayer {
    #[allow(missing_docs)]
    pub weights: Matrix,
    #[allow(missing_docs)]
    pub bias: Vector,
}

impl DenseLayer {
    /// Creates a new dense layer with random initialization.
    ///
    /// This method uses the default thread-local RNG.
    #[verified_engine::verified]
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = oxidize_core::rng::OxidizeRng::default();
        Self::new_with_rng(input_dim, output_dim, &mut rng)
    }

    /// Creates a new dense layer with random initialization using the provided RNG.
    #[verified_engine::verified]
    pub fn new_with_rng<R: rand::Rng + ?Sized>(
        input_dim: usize,
        output_dim: usize,
        rng: &mut R,
    ) -> Self {
        // He initialization or simple uniform can be used.
        // Using Uniform(-0.1, 0.1) for simplicity in this theory module.
        let dist = Uniform::new(-0.1, 0.1);

        let weights = DMatrix::from_fn(output_dim, input_dim, |_, _| dist.sample(rng));
        let bias = DVector::from_fn(output_dim, |_, _| dist.sample(rng));

        Self { weights, bias }
    }

    /// Performs the forward pass.
    #[verified_engine::verified]
    pub fn forward(&self, x: &Vector) -> Vector {
        linear_transformation(x, &self.weights, &self.bias)
    }
}
