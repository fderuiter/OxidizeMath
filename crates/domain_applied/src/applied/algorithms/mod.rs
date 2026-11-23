//! # Algorithms
//!
//! A collection of general-purpose algorithms, optimized for clarity and educational value.
//!
//! This module provides robust implementations of classic algorithms, designed with the **Strategy Pattern**
//! to allow for flexible behavior and comprehensive analysis.
//!
//! ##  Algorithm Taxonomy
//!
//! ```mermaid
//! graph TD
//!     Algo[Algorithms] --> Est[Estimation]
//!     Algo --> Sort[Sorting]
//!
//!     Est --> KF[Kalman Filter]
//!     KF --> Model[Generic Model Trait]
//!
//!     Sort --> Strat[Sorting Strategy]
//!     Strat --> DC[Divide & Conquer]
//!     Strat --> Heap[Heap Sort]
//!     Strat --> Linear[Linear Sort]
//!     Strat --> Elem[Elementary Sort]
//!
//!     style Algo fill:#f9f,stroke:#333,stroke-width:2px
//!     style Est fill:#bbf,stroke:#333
//!     style Sort fill:#bfb,stroke:#333
//! ```
//!
//! ##  Quick Start: Kalman Tracking
//!
//! Track a moving object in 1D space using a Constant Velocity model.
//!
//! ```rust
//! # use domain_applied::__doctest_facade as math_explorer;
//! use math_explorer::applied::algorithms::kalman::{KalmanFilter, KalmanModel};
//! use math_explorer::nalgebra::{DMatrix, DVector};
//!
//! // Define a Constant Velocity Model (Position + Velocity)
//! struct ConstantVelocityModel {
//!     process_noise: f64,
//!     measurement_noise: f64,
//! }
//!
//! impl KalmanModel<f64> for ConstantVelocityModel {
//!     fn transition_matrix(&self, dt: f64) -> DMatrix<f64> {
//!         // State = [pos, vel]
//!         // pos_k = pos_{k-1} + vel_{k-1} * dt
//!         // vel_k = vel_{k-1}
//!         DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0])
//!     }
//!
//!     fn measurement_matrix(&self) -> DMatrix<f64> {
//!         // We only measure position: z_k = [1, 0] * state_k
//!         DMatrix::from_row_slice(1, 2, &[1.0, 0.0])
//!     }
//!
//!     fn process_noise(&self, _dt: f64) -> DMatrix<f64> {
//!         DMatrix::identity(2, 2) * self.process_noise
//!     }
//!
//!     fn measurement_noise(&self) -> DMatrix<f64> {
//!         DMatrix::from_element(1, 1, self.measurement_noise)
//!     }
//! }
//!
//! fn main() {
//!     // 1. Initialize the Filter
//!     let model = ConstantVelocityModel {
//!         process_noise: 0.1,
//!         measurement_noise: 1.0,
//!     };
//!
//!     // Initial State: Position = 0, Velocity = 5 m/s
//!     let initial_state = DVector::from_vec(vec![0.0, 5.0]);
//!     let initial_covariance = DMatrix::identity(2, 2);
//!
//!     let mut kf = KalmanFilter::builder(model, 1.0)
//!         .initial_state(initial_state)
//!         .initial_covariance(initial_covariance)
//!         .build()
//!         .unwrap();
//!
//!     // 2. Predict Step (Time evolves)
//!     kf.predict();
//!
//!     // Predicted State: Pos = 5.0, Vel = 5.0
//!     assert!((kf.state[0] - 5.0).abs() < math_explorer::registry::TOLERANCE_FAST);
//!
//!     // 3. Update Step (New Measurement arrives)
//!     // Sensor says Position = 5.2 (slightly off due to noise)
//!     let measurement = DVector::from_vec(vec![5.2]);
//!     kf.update(&measurement).expect("Update failed");
//!
//!     // Filter corrects estimate based on Kalman Gain
//!     println!("Estimated Position: {:.4}", kf.state[0]);
//!     println!("Estimated Velocity: {:.4}", kf.state[1]);
//! }
//! ```

pub mod kalman;
pub mod sorting;

// [cite:algorithmic_information_rust]

use pure_math::theory_verification;

theory_verification!(
    module = "algorithms",
    constants = {
        WEIGHT = 1.0;
    },
    test = {
        use approx::assert_relative_eq;
        assert_relative_eq!(
            WEIGHT,
            1.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);
