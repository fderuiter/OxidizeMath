use nalgebra::DMatrix;

/// Abstract interface for an Attention Mechanism.
pub trait AttentionMechanism {
    /// Performs the forward pass of the attention mechanism.
    ///
    /// # Arguments
    /// * `q` - Queries matrix.
    /// * `k` - Keys matrix.
    /// * `v` - Values matrix.
    /// * `mask` - Optional mask matrix.
    ///
    /// # Returns
    /// The context vectors (output matrix).
    #[verified_engine::verified]
    fn forward(
        &self,
        q: &DMatrix<f64>,
        k: &DMatrix<f64>,
        v: &DMatrix<f64>,
        mask: Option<&DMatrix<f64>>,
    ) -> DMatrix<f64>;
}

/// Abstract interface for a Feed-Forward Network.
pub trait FeedForwardNetwork {
    /// Performs the forward pass of the feed-forward network.
    ///
    /// # Arguments
    /// * `x` - Input matrix.
    ///
    /// # Returns
    /// The output matrix.
    #[verified_engine::verified]
    fn forward(&self, x: &DMatrix<f64>) -> DMatrix<f64>;
}

/// Abstract interface for a Normalization Layer.
pub trait NormalizationLayer {
    /// Applies normalization to the input.
    ///
    /// # Arguments
    /// * `x` - Input matrix.
    ///
    /// # Returns
    /// The normalized matrix.
    #[verified_engine::verified]
    fn forward(&self, x: &DMatrix<f64>) -> DMatrix<f64>;
}
