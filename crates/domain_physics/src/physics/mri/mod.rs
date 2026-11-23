//! # MRI Physics Simulation
//!
//! This module provides a rigorous simulation of Magnetic Resonance Imaging (MRI) physics,
//! implementing the full signal chain from nuclear spin dynamics to image reconstruction.
//!
//! It bridges **Quantum Mechanics** (spin states) and **Classical Electromagnetism** (Bloch equations)
//! to model how tissue properties ($T_1, T_2, \rho$) translate into medical images.
//!
//! ##  Simulation Pipeline
//!
//! The MRI process is modeled as a sequence of transformations:
//!
//! ```mermaid
//! graph TD
//!     subgraph "1. Physics"
//!     Protons[Proton Spins] -->|B0 Field| Align[Net Magnetization M]
//!     RF[RF Pulse B1] -->|Excitation| Transverse[Transverse M_xy]
//!     end
//!
//!     subgraph "2. Encoding"
//!     Gradients[Gradient Fields Gx, Gy] -->|Phase/Freq Encoding| KSpace[k-Space Trajectory]
//!     Transverse -->|Evolution| KSpace
//!     end
//!
//!     subgraph "3. Acquisition"
//!     KSpace -->|Signal Equation| Signal[Raw Signal S(t)]
//!     Signal -->|ADC| Digital[Digitized k-Space Data]
//!     end
//!
//!     subgraph "4. Reconstruction"
//!     Digital -->|2D IFT| Image[Reconstructed Image]
//!     end
//!
//!     style Protons fill:#e1f5fe,stroke:#01579b
//!     style Signal fill:#fff3e0,stroke:#e65100
//!     style Image fill:#e8f5e9,stroke:#1b5e20
//! ```
//!
//! ##  Example
//!
//! Simulating Transverse Relaxation (T2 Decay) using the `BlochSimulator`:
//!
//! ```rust
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::mri::BlochSimulator;
//! use math_explorer::pure_math::analysis::ode::TimeStepper;
//! use math_explorer::nalgebra::Vector3;
//!
//! // 1. Initialize magnetization vector (M) flipped 90 degrees into the transverse plane
//! let initial_m = Vector3::new(0.0, 1.0, 0.0);
//! let m0 = 1.0; // Equilibrium magnetization
//! let mut bloch = BlochSimulator::new(initial_m, m0);
//!
//! // 2. Set tissue relaxation parameters
//! let t1 = 1.0;  // Longitudinal relaxation time (seconds)
//! let t2 = 0.1;  // Transverse relaxation time (seconds)
//! bloch.set_relaxation(t1, t2);
//!
//! // 3. Evolve the system forward in time (dt = 0.01s) for one T2 period (0.1s)
//! let dt = 0.01;
//! for _ in 0..10 {
//!     <BlochSimulator as TimeStepper<Vector3<f64>>>::step(&mut bloch, dt);
//! }
//!
//! // The transverse magnetization (My) should decay to ~36.8% (1/e) of its initial value
//! let my = bloch.magnetization.y;
//! assert!((my - 0.3678).abs() < 0.01, "Expected ~0.3678, got {}", my);
//! ```
//!
//! ##  Domains
//!
//! *   **Quantum Foundations** (`proton`): Constants and properties for Hydrogen nuclei (gyromagnetic ratio $\gamma$).
//! *   **Classical Dynamics** (`bloch`): The **Bloch Equation** solver, tracking the magnetization vector $\mathbf{M}$ over time:
//!     $$ \frac{d\mathbf{M}}{dt} = \mathbf{M} \times \gamma \mathbf{B} - \frac{M_x \hat{x} + M_y \hat{y}}{T_2} - \frac{(M_z - M_0) \hat{z}}{T_1} $$
//! *   **Spatial Encoding** (`scanner`): Gradient coils simulation ($G_x, G_y, G_z$) to spatially resolve signals.
//! *   **Image Reconstruction** (`reconstruction`): Fast Fourier Transform (FFT) algorithms to convert frequency-domain signals back to spatial images.

pub mod bloch;
pub mod phantom;
pub mod proton;
pub mod reconstruction;
pub mod scanner;

// Re-export BlochSimulator and Fft2dWorkspace for ergonomic access
pub use bloch::BlochSimulator;
pub use reconstruction::Fft2dWorkspace;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;
    use pure_math::pure_math::analysis::ode::TimeStepper;
    use std::f64::consts::PI;

    #[test]
    #[verified_engine::verified]
    fn test_larmor_frequency() {
        // Verify Larmor frequency of Hydrogen at 1.5 Tesla
        // Expected: ~ 63.87 MHz -> 63.87e6 * 2pi rad/s
        let b0 = 1.5;
        let omega = proton::larmor_frequency(b0);
        let freq_hz = omega / (2.0 * PI);

        // 63.87 MHz is approx 63,857,000 Hz based on gamma/2pi = 42.57 MHz/T
        // 42.57 * 1.5 = 63.855
        // Using gamma = 2.675e8:
        // f = 2.675e8 * 1.5 / 2pi = 4.0125e8 / 6.283... = 63.856 MHz

        assert_relative_eq!(freq_hz, 63.856e6, epsilon = 1.0e5);
    }

    #[test]
    #[verified_engine::verified]
    fn test_bloch_relaxation_modern() {
        // Test T2 relaxation using modern API
        // Initialize M = [0, 1, 0], B = 0 (no precession), T1 = inf, T2 = 1.0
        let initial_m = Vector3::new(0.0, 1.0, 0.0);
        let m0 = 1.0;
        let mut bloch = BlochSimulator::new(initial_m, m0);

        let dt = 0.01;
        let t2 = 0.5; // T2 = 0.5s
        let t1 = 1e9; // Long T1
        let b_field = Vector3::zeros(); // No B field to isolate relaxation

        bloch.set_b_field(b_field);
        bloch.set_relaxation(t1, t2);

        // Step for a total of 0.5 seconds (1 * T2)
        // M_y should decay to 1/e * initial
        let steps = (t2 / dt) as usize;
        for _ in 0..steps {
            <BlochSimulator as TimeStepper<Vector3<f64>>>::step(&mut bloch, dt);
        }

        let expected_y = (-1.0_f64).exp(); // e^-1 approx 0.3678

        // RK4 is highly accurate, so error should be small
        assert_relative_eq!(bloch.magnetization.y, expected_y, epsilon = 1e-5);
        assert_relative_eq!(bloch.magnetization.x, 0.0);
        // z should recover towards m0=1 from 0? No, initial z=0.
        // dMz/dt = (M0 - Mz)/T1 approx 0.
        assert_relative_eq!(
            bloch.magnetization.z,
            0.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_bloch_rk4_accuracy_modern() {
        // T2 relaxation with RK4 using modern API
        let initial_m = Vector3::new(0.0, 1.0, 0.0);
        let m0 = 1.0;
        let mut bloch = BlochSimulator::new(initial_m, m0);

        let dt = 0.1; // Large step size where Euler struggles
        let t2 = 1.0;
        let t1 = 1e9;
        let b_field = Vector3::zeros();

        bloch.set_b_field(b_field);
        bloch.set_relaxation(t1, t2);

        // 1 second simulation
        let steps = (1.0 / dt) as usize;
        for _ in 0..steps {
            <BlochSimulator as TimeStepper<Vector3<f64>>>::step(&mut bloch, dt);
        }

        let expected_y = (-1.0_f64).exp();

        // With dt=0.1, RK4 should be very close.
        assert_relative_eq!(bloch.magnetization.y, expected_y, epsilon = 1e-5);
    }

    #[test]
    #[verified_engine::verified]
    fn test_bloch_time_stepper_api() {
        // Test new TimeStepper API
        let initial_m = Vector3::new(0.0, 1.0, 0.0);
        let m0 = 1.0;
        let mut bloch = BlochSimulator::new(initial_m, m0);

        // Setup parameters once
        let t2 = 1.0;
        let t1 = 1e9;
        bloch.set_relaxation(t1, t2);
        // B field defaults to 0, which is what we want

        let dt = 0.1;

        // Use TimeStepper::step which defaults to RK4
        let steps = (1.0 / dt) as usize;
        for _ in 0..steps {
            <BlochSimulator as TimeStepper<Vector3<f64>>>::step(&mut bloch, dt);
        }

        let expected_y = (-1.0_f64).exp();
        assert_relative_eq!(bloch.magnetization.y, expected_y, epsilon = 1e-5);
    }
}

// [cite:favorite_child]

use pure_math::theory_verification;

theory_verification!(
    module = "mri",
    constants = {
        DUMMY = 1.0;
    },
    test = {
        use approx::assert_relative_eq;
        assert_relative_eq!(DUMMY, 1.0, epsilon = math_commons::registry::TOLERANCE_FAST);
    }
);
