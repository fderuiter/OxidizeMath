//! Dimensional Physical Constant Registries (`si` and `nuclear`).
//!
//! All dimensional physical constants are partitioned by unit-system frame (`SiFrame` vs.
//! `NuclearMeVFmFrame`) using [`math_scalars::Quantity`] so constants from incompatible unit
//! scales (such as SI `m/s` vs. nuclear `fm/s`) cannot be combined without explicit frame
//! conversion:
//!
//! ```
//! use math_commons::constants::{nuclear, si};
//! use math_scalars::{Kelvin, Meters, Tesla};
//!
//! // Coherent SI horizon scale R_s = 2 G M_sun / c^2 (meters)
//! let r_s: Meters = (si::G * si::SOLAR_MASS * 2.0) / (si::C * si::C);
//! assert!((r_s.get() - 2_954.126).abs() < 1.0);
//!
//! // Nuclear reduced Planck constant hbar = (hbar * c) / c_nuc (MeV * s)
//! let hbar_nuc = nuclear::HBAR_C / nuclear::LIGHT_SPEED;
//! assert!(hbar_nuc.get() > 6.5e-22 && hbar_nuc.get() < 6.6e-22);
//!
//! // MRI Boltzmann exponent (hbar * gamma * B0) / (k_B * T) cancels to Dimensionless
//! let ratio = (si::H_BAR * si::PROTON_GYROMAGNETIC_RATIO * Tesla::new(3.0))
//!     / (si::K_B * Kelvin::new(310.0));
//! assert!(ratio.exp().get() > 1.0);
//! ```
//!
//! Mixing `si::C` (`m/s`) and `nuclear::LIGHT_SPEED` (`fm/s`) is rejected at compile time:
//!
//! ```compile_fail
//! use math_commons::constants::{nuclear, si};
//!
//! let _bad = si::C + nuclear::LIGHT_SPEED;
//! ```
//!
//! ```compile_fail
//! use math_commons::constants::{nuclear, si};
//!
//! let _bad_ratio = si::C / nuclear::LIGHT_SPEED;
//! ```
// [cite:math_commons]

/// Small dimensionless tolerance (`1e-10`) for numerical comparisons.
pub const EPSILON: f64 = 1e-10;

/// Dimensionless natural logarithm $\ln(0.7)$.
pub const LN_0_7: f64 = -0.356_674_943_938_732_4;

/// Coherent SI (`m`, `kg`, `s`, `A`, `K`, `mol`, `cd`) physical constants tagged with
/// [`math_scalars::SiFrame`].
pub mod si {
    use math_scalars::{
        GyromagneticRatioSi, JoulesPerKelvin, JoulesPerMoleKelvin, JoulesSecond, Kilograms,
        MetersPerSecond, SiGravitationalConstant, SquareMeters,
    };

    /// Speed of light in vacuum $c = 299\,792\,458\,\text{m}\cdot\text{s}^{-1}$.
    pub const C: MetersPerSecond = MetersPerSecond::new(299_792_458.0);

    /// Newtonian gravitational constant $G = 6.67430 \times 10^{-11}\,\text{m}^3\cdot\text{kg}^{-1}\cdot\text{s}^{-2}$.
    pub const G: SiGravitationalConstant = SiGravitationalConstant::new(6.674_30e-11);

    /// Nominal solar mass $M_\odot = 1.989 \times 10^{30}\,\text{kg}$.
    pub const SOLAR_MASS: Kilograms = Kilograms::new(1.989e30);

    /// Thomson electron scattering cross-section $\sigma_T = 6.6524 \times 10^{-29}\,\text{m}^2$.
    pub const SIGMA_T: SquareMeters = SquareMeters::new(6.6524e-29);

    /// Reduced Planck constant $\hbar = 1.054571817 \times 10^{-34}\,\text{J}\cdot\text{s}$.
    pub const H_BAR: JoulesSecond = JoulesSecond::new(1.054_571_817e-34);

    /// Boltzmann constant $k_B = 1.380649 \times 10^{-23}\,\text{J}\cdot\text{K}^{-1}$.
    pub const K_B: JoulesPerKelvin = JoulesPerKelvin::new(1.380_649e-23);

    /// Proton gyromagnetic ratio $\gamma_p = 2.6752218744 \times 10^8\,\text{rad}\cdot\text{s}^{-1}\cdot\text{T}^{-1}$.
    pub const PROTON_GYROMAGNETIC_RATIO: GyromagneticRatioSi =
        GyromagneticRatioSi::new(2.675_221_874_4e8);

    /// Molar universal gas constant $R = 8.314462618\,\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}$.
    pub const GAS_CONSTANT_R: JoulesPerMoleKelvin = JoulesPerMoleKelvin::new(8.314_462_618);
}

/// Nuclear structure & decay constants (`MeV`, `fm`, `s`) tagged with
/// [`math_scalars::NuclearMeVFmFrame`].
pub mod nuclear {
    use math_scalars::{FemtometersPerSecond, MeV, MeVFm};

    /// Speed of light in nuclear femtometers per second $c = 2.99792458 \times 10^{23}\,\text{fm}\cdot\text{s}^{-1}$.
    pub const LIGHT_SPEED: FemtometersPerSecond = FemtometersPerSecond::new(2.997_924_58e23);

    /// Reduced Planck constant times speed of light $\hbar c = 197.3\,\text{MeV}\cdot\text{fm}$.
    pub const HBAR_C: MeVFm = MeVFm::new(197.3);

    /// Squared elementary charge $e^2 = \alpha \hbar c \approx 1.439976\,\text{MeV}\cdot\text{fm}$.
    pub const E_SQUARED: MeVFm = MeVFm::new(1.439_976);

    /// Proton rest mass-energy $m_p c^2 = 938.272\,\text{MeV}$.
    pub const PROTON_MASS: MeV = MeV::new(938.272);

    /// Neutron rest mass-energy $m_n c^2 = 939.565\,\text{MeV}$.
    pub const NEUTRON_MASS: MeV = MeV::new(939.565);

    /// Semi-empirical Bethe-Weizsäcker liquid-drop energy coefficients in $\text{MeV}$.
    pub mod liquid_drop_constants {
        use math_scalars::MeV;

        /// Volume binding energy coefficient $a_V = 15.75\,\text{MeV}$.
        pub const A_V: MeV = MeV::new(15.75);
        /// Surface tension energy coefficient $a_S = 17.8\,\text{MeV}$.
        pub const A_S: MeV = MeV::new(17.8);
        /// Coulomb repulsion energy coefficient $a_C = 0.711\,\text{MeV}$.
        pub const A_C: MeV = MeV::new(0.711);
        /// Asymmetry energy coefficient $a_{\text{sym}} = 23.7\,\text{MeV}$.
        pub const A_SYM: MeV = MeV::new(23.7);
        /// Pairing energy coefficient $\delta_0 = 11.18\,\text{MeV}$.
        pub const DELTA_COEFF: MeV = MeV::new(11.18);
    }

    /// Empirical nuclear radius scaling constants in $\text{fm}$.
    pub mod property_constants {
        use math_scalars::Femtometers;

        /// Nuclear radius scale factor $r_0 = 1.2\,\text{fm}$ ($R = r_0 A^{1/3}$).
        pub const R0: Femtometers = Femtometers::new(1.2);
    }
}
