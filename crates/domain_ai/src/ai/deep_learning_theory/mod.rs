//! # Deep Learning Theory
//!
//! > **"There is nothing more practical than a good theory."** — Kurt Lewin
//!
//! This module deconstructs the "Black Box" of Deep Learning into its three fundamental mathematical pillars:
//! **Linear Algebra**, **Calculus**, and **Probability**.
//!
//! Instead of using high-level frameworks like PyTorch or TensorFlow, we implement the core mechanics
//! from first principles to understand *why* models learn.
//!
//! ## The Three Pillars
//!
//! 1.  **[Linear Algebra](linear_algebra)**: The language of data.
//!     *   **Why?** Neural networks are massive composition of linear transformations ($Wx + b$).
//!     *   **What?** Matrices, Vectors, Dot Products.
//!
//! 2.  **[Calculus](calculus)**: The engine of optimization.
//!     *   **Why?** To improve, we need to know the direction of steepest descent (Gradient).
//!     *   **What?** Derivatives, Chain Rule, Backpropagation.
//!
//! 3.  **[Probability](probability)**: The measure of uncertainty.
//!     *   **Why?** We predict probabilities (Softmax) and minimize surprise (Cross-Entropy).
//!     *   **What?** Likelihood, Distributions, MLE.
//!
//! ## The Learning Cycle
//!
//! Training a neural network is a cyclical process of prediction and correction.
//!
//! ```mermaid
//! graph TD
//!     subgraph "Forward Pass (Prediction)"
//!     Input[Input Data x] -->|Linear Algebra| Linear[Linear Layer z=Wx+b]
//!     Linear -->|Activation| ReLU[ReLU Activation a=max(0,z)]
//!     ReLU --> Output[Output Prediction y_hat]
//!     end
//!
//!     subgraph "Backward Pass (Correction)"
//!     Output -->|Probability| Loss{Loss Function J}
//!     Target[True Label y] --> Loss
//!     Loss -->|Calculus| Grads[Compute Gradients dJ/dW]
//!     Grads -->|Optimization| Update[Update Weights W = W - lr * dJ/dW]
//!     end
//!
//!     Update --> Linear
//! ```
//!
//! ##  Quick Start: "Deep Learning from Scratch"
//!
//! Train a simple 2-layer network to solve a classification problem using our theoretical primitives.
//!
//! ```rust
//! # use domain_ai::__doctest_facade as math_explorer;
//! use math_explorer::ai::deep_learning_theory::cycle::TrainingLoop;
//! use math_explorer::ai::optimization::SGD;
//! use math_explorer::nalgebra::DVector;
//!
//! // 1. Define Architecture
//! // Input Layer: 2 neurons
//! // Hidden Layer: 4 neurons
//! // Output Layer: 2 neurons (Binary Classification)
//! let input_dim = 2;
//! let hidden_dim = 4;
//! let output_dim = 2;
//!
//! // 2. Choose an Optimizer
//! // Stochastic Gradient Descent with Learning Rate 0.01
//! let optimizer = Box::new(SGD::new(0.01));
//!
//! // 3. Initialize the Training Loop
//! let mut model = TrainingLoop::new(input_dim, hidden_dim, output_dim, optimizer);
//!
//! // 4. Create Dummy Data (e.g., trying to learn Class 0)
//! let x = DVector::from_vec(vec![0.5, -0.5]);
//! let y_target = DVector::from_vec(vec![1.0, 0.0]); // One-hot for Class 0
//!
//! // 5. Perform a Single Training Step
//! let loss = model.train_step(&x, &y_target).unwrap();
//!
//! println!("Initial Loss: {:.4}", loss);
//!
//! // 6. Verify Backpropagation
//! // After one step, the model should have adjusted weights to lower the loss for this sample.
//! let prediction = model.predict(&x);
//! println!("Prediction for Class 0: {:.4}", prediction[0]);
//! ```
//!
//! ## Modules
//!
//! *   **[Calculus](calculus)**: Implementation of `relu`, `sigmoid`, and the `linear_backward` pass.
//! *   **[Linear Algebra](linear_algebra)**: `DenseLayer` struct and matrix operations.
//! *   **[Probability](probability)**: `softmax` and Maximum Likelihood Estimation concepts.
//! *   **[Model](model)**: The `Trainable` trait and `TwoLayerMLP` implementation.
//! *   **[Cycle](cycle)**: The `TrainingLoop` that ties it all together.

#[allow(missing_docs)]
pub mod calculus;
#[allow(missing_docs)]
pub mod cycle;
#[allow(missing_docs)]
pub mod linear_algebra;
#[allow(missing_docs)]
pub mod model;
#[allow(missing_docs)]
pub mod probability;

// [cite:dwarf_galaxy_empirical_dependencies]
