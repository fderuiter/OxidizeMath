//! Universal Physical Constant Registry
//! Centralized repository for standard values.

/// Speed of light in vacuum (m/s).
pub const C: f64 = 299_792_458.0;

/// Gravitational constant (m^3 kg^-1 s^-2).
pub const G: f64 = 6.674_30e-11;

/// Solar Mass (kg).
pub const SOLAR_MASS: f64 = 1.989e30;

/// Thomson Cross Section (m^2).
/// Value derived from approx 6.6524e-25 cm^2.
pub const SIGMA_T: f64 = 6.6524e-29;

/// Small epsilon for floating point comparisons.
pub const EPSILON: f64 = 1e-10;

/// Proton mass in MeV/c^2.
pub const PROTON_MASS: f64 = 938.272;

/// Neutron mass in MeV/c^2.
pub const NEUTRON_MASS: f64 = 939.565;

/// Reduced Planck constant times speed of light (hbar * c) in MeV fm.
pub const HBAR_C: f64 = 197.3;

/// Speed of light in fm/s (approx 3.0e23).
pub const LIGHT_SPEED: f64 = 2.99792458e23;

/// Squared elementary charge (e^2) in MeV fm.
/// Derived from fine-structure constant alpha = e^2 / (hbar c) ~ 1/137.036.
pub const E_SQUARED: f64 = 1.439976;

/// Natural logarithm of 0.7
pub const LN_0_7: f64 = -0.3566749439387324;

/// Liquid Drop Model Constants
pub mod liquid_drop_constants {
    /// Volume binding energy coefficient $a_V$ (MeV).
    pub const A_V: f64 = 15.75;
    /// Surface tension energy coefficient $a_S$ (MeV).
    pub const A_S: f64 = 17.8;
    /// Coulomb electrostatic repulsion coefficient $a_C$ (MeV).
    pub const A_C: f64 = 0.711;
    /// Asymmetry energy coefficient $a_{\mathrm{sym}}$ (MeV).
    pub const A_SYM: f64 = 23.7;
    /// Pairing energy scale coefficient $\delta_0$ (MeV).
    pub const DELTA_COEFF: f64 = 11.18;
}

/// Nuclear Properties Constants
pub mod property_constants {
    /// Empirical nuclear radius scale constant $r_0$ (fm).
    pub const R0: f64 = 1.2; // fm
}
