//! # Empirical Dependencies for Irregular Dwarf Galaxies
//!
//! This module provides an implementation of the empirical formulas for irregular dwarf galaxies
//! as described in the paper by Tillaboev, Tadjibaev, and Otojanova (2025).
//!
//! # Refactoring Note
//! This module uses the **Strategy Pattern** and **Strong Types** to ensure type safety and extensibility.
//! The `GalaxyModel` trait defines the interface for calculations, implemented by specific strategies
//! like `GeneralIrregular`, `TypeCode10`, and `TypeCode95To99`.
//!
//! Legacy functions are preserved for backward compatibility but are deprecated.

// --- Strong Types ---

/// Distance in Megaparsecs (Mpc).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Mpc(pub f64);

impl Mpc {
    /// Returns the value as `f64`.
    #[verified_engine::verified]
    pub fn as_f64(&self) -> f64 {
        self.0
    }
}

/// Logarithm of the mass in solar masses (log(M/M_Sun)).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SolarMassLog(pub f64);

impl SolarMassLog {
    /// Returns the value as `f64`.
    #[verified_engine::verified]
    pub fn as_f64(&self) -> f64 {
        self.0
    }
}

/// Astronomical Magnitude (Apparent or Absolute).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Magnitude(pub f64);

impl Magnitude {
    /// Returns the value as `f64`.
    #[verified_engine::verified]
    pub fn as_f64(&self) -> f64 {
        self.0
    }
}

/// Redshift (z).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Redshift(pub f64);

impl Redshift {
    /// Returns the value as `f64`.
    #[verified_engine::verified]
    pub fn as_f64(&self) -> f64 {
        self.0
    }
}

// --- GalaxyModel Trait ---

/// Defines the empirical relationships for a specific galaxy morphology.
pub trait GalaxyModel {
    /// Calculates log(Mass) from Distance.
    #[verified_engine::verified]
    fn log_mass_from_distance(&self, d: Mpc) -> SolarMassLog;

    /// Calculates Apparent Magnitude (B-band) from Distance.
    #[verified_engine::verified]
    fn apparent_magnitude_from_distance(&self, d: Mpc) -> Option<Magnitude>;

    /// Calculates log(Mass) from Absolute Magnitude (V-band).
    #[verified_engine::verified]
    fn log_mass_from_absolute_magnitude(&self, m: Magnitude) -> SolarMassLog;

    /// Calculates Redshift from log(Mass).
    #[verified_engine::verified]
    fn redshift_from_log_mass(&self, m: SolarMassLog) -> Redshift;
}

// --- Strategies ---

/// Strategy for "All" irregular dwarf galaxies (General Group).
#[derive(Debug, Clone, Copy, Default)]
pub struct GeneralIrregular;

impl GalaxyModel for GeneralIrregular {
    #[verified_engine::verified]
    fn log_mass_from_distance(&self, d: Mpc) -> SolarMassLog {
        SolarMassLog(0.0230 * d.0 + 0.7840)
    }

    #[verified_engine::verified]
    fn apparent_magnitude_from_distance(&self, d: Mpc) -> Option<Magnitude> {
        Some(Magnitude(0.0206 * d.0 + 16.0010))
    }

    #[verified_engine::verified]
    fn log_mass_from_absolute_magnitude(&self, m: Magnitude) -> SolarMassLog {
        SolarMassLog(-0.6670 * m.0 - 1.4975)
    }

    #[verified_engine::verified]
    fn redshift_from_log_mass(&self, m: SolarMassLog) -> Redshift {
        Redshift(0.0094 * m.0 - 0.7270)
    }
}

/// Strategy for Type Code 10 irregular dwarf galaxies.
#[derive(Debug, Clone, Copy, Default)]
pub struct TypeCode10;

impl GalaxyModel for TypeCode10 {
    #[verified_engine::verified]
    fn log_mass_from_distance(&self, d: Mpc) -> SolarMassLog {
        SolarMassLog(0.0250 * d.0 + 7.6860)
    }

    #[verified_engine::verified]
    fn apparent_magnitude_from_distance(&self, d: Mpc) -> Option<Magnitude> {
        Some(Magnitude(0.0140 * d.0 + 16.575))
    }

    #[verified_engine::verified]
    fn log_mass_from_absolute_magnitude(&self, m: Magnitude) -> SolarMassLog {
        // Uses the same formula as the general case for this relationship
        SolarMassLog(-0.6670 * m.0 - 1.4975)
    }

    #[verified_engine::verified]
    fn redshift_from_log_mass(&self, m: SolarMassLog) -> Redshift {
        Redshift(0.00093 * m.0 - 0.0716)
    }
}

/// Strategy for Type Code 9.5 - 9.9 irregular dwarf galaxies.
#[derive(Debug, Clone, Copy, Default)]
pub struct TypeCode95To99;

impl GalaxyModel for TypeCode95To99 {
    #[verified_engine::verified]
    fn log_mass_from_distance(&self, d: Mpc) -> SolarMassLog {
        SolarMassLog(0.0504 * d.0 + 7.5715)
    }

    #[verified_engine::verified]
    fn apparent_magnitude_from_distance(&self, _d: Mpc) -> Option<Magnitude> {
        None // Relationship not established in the paper
    }

    #[verified_engine::verified]
    fn log_mass_from_absolute_magnitude(&self, m: Magnitude) -> SolarMassLog {
        SolarMassLog(-0.3837 * m.0 - 2.2864)
    }

    #[verified_engine::verified]
    fn redshift_from_log_mass(&self, m: SolarMassLog) -> Redshift {
        Redshift(0.0031 * m.0 - 0.0223)
    }
}
