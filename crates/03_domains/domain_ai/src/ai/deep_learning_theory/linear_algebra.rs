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
#[derive(Clone, Debug)]
pub struct DenseLayer {
    /// Weight matrix $\mathbf{W} \in \mathbb{R}^{m \times n}$ mapping inputs to pre-activations.
    pub weights: Matrix,
    /// Additive bias vector $\mathbf{b} \in \mathbb{R}^m$.
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

    /// Creates a new dense layer initialized on ChaCha8 `stream` with He et al. (2015) scaling.
    #[verified_engine::verified]
    pub fn new_on_stream(input_dim: usize, output_dim: usize, stream: u64) -> Self {
        let mut rng = oxidize_core::rng::OxidizeRng::default().with_stream(stream);
        Self::new_with_rng(input_dim, output_dim, &mut rng)
    }

    /// Creates a new dense layer with He et al. (2015) variance scaling ($\operatorname{Var}(W) = 2 / d_{\mathrm{in}}$).
    #[verified_engine::verified]
    pub fn new_with_rng<R: rand::Rng + ?Sized>(
        input_dim: usize,
        output_dim: usize,
        rng: &mut R,
    ) -> Self {
        let fan_in = input_dim.max(1) as f64;
        let limit = (6.0 / fan_in).sqrt();
        let dist = Uniform::new(-limit, limit);

        let weights = DMatrix::from_fn(output_dim, input_dim, |_, _| dist.sample(rng));
        let bias = DVector::zeros(output_dim);

        Self { weights, bias }
    }

    /// Creates a new dense layer with Glorot / Xavier uniform scaling ($\operatorname{Var}(W) = 2 / (d_{\mathrm{in}} + d_{\mathrm{out}})$).
    #[verified_engine::verified]
    pub fn new_with_xavier_rng<R: rand::Rng + ?Sized>(
        input_dim: usize,
        output_dim: usize,
        rng: &mut R,
    ) -> Self {
        let fan_sum = input_dim.saturating_add(output_dim).max(1) as f64;
        let limit = (6.0 / fan_sum).sqrt();
        let dist = Uniform::new(-limit, limit);

        let weights = DMatrix::from_fn(output_dim, input_dim, |_, _| dist.sample(rng));
        let bias = DVector::zeros(output_dim);

        Self { weights, bias }
    }

    /// Performs the forward pass.
    #[verified_engine::verified]
    pub fn forward(&self, x: &Vector) -> Vector {
        linear_transformation(x, &self.weights, &self.bias)
    }
}
