use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Exact CODATA 2018/2022 SI elementary charge $e$ in Coulombs (`J / eV`).
pub const ELEMENTARY_CHARGE_C: f64 = 1.602_176_634e-19;

/// Energy in Electron Volts (`eV`).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct ElectronVolts(f64);

impl ElectronVolts {
    /// Creates a new `ElectronVolts` instance.
    #[verified_engine::verified]
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    /// Converts electron-volts (`eV`) to SI Joules (`J`) using [`ELEMENTARY_CHARGE_C`].
    #[verified_engine::verified]
    pub fn to_joules(self) -> f64 {
        self.0 * ELEMENTARY_CHARGE_C
    }

    /// Constructs `ElectronVolts` from SI energy in Joules (`J`) using [`ELEMENTARY_CHARGE_C`].
    #[verified_engine::verified]
    pub fn from_joules(joules: f64) -> Self {
        Self(joules / ELEMENTARY_CHARGE_C)
    }

    /// Returns the absolute value.
    #[verified_engine::verified]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Returns the value as `f64`.
    #[verified_engine::verified]
    pub fn as_f64(&self) -> f64 {
        self.0
    }

    /// Raises the numerical energy magnitude (`eV`) to power `n`.
    #[verified_engine::verified]
    pub fn powf(&self, n: f64) -> Self {
        Self::new(self.0.powf(n))
    }

    /// Computes the square root of the numerical energy magnitude (`eV`).
    #[verified_engine::verified]
    pub fn sqrt(&self) -> Self {
        Self::new(self.0.sqrt())
    }

    /// Computes the natural logarithm of the numerical energy magnitude (`eV`).
    #[verified_engine::verified]
    pub fn ln(&self) -> Self {
        Self::new(self.0.ln())
    }

    /// Computes the exponential of the numerical energy magnitude (`eV`).
    #[verified_engine::verified]
    pub fn exp(&self) -> Self {
        Self::new(self.0.exp())
    }
}

impl fmt::Display for ElectronVolts {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:.4} eV", self.0)
    }
}

impl Add for ElectronVolts {
    type Output = Self;
    #[verified_engine::verified]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl Sub for ElectronVolts {
    type Output = Self;
    #[verified_engine::verified]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl Mul<f64> for ElectronVolts {
    type Output = Self;
    #[verified_engine::verified]
    fn mul(self, scalar: f64) -> Self {
        Self(self.0 * scalar)
    }
}

impl Mul<ElectronVolts> for f64 {
    type Output = ElectronVolts;
    #[verified_engine::verified]
    fn mul(self, ev: ElectronVolts) -> ElectronVolts {
        ElectronVolts(self * ev.0)
    }
}

impl Div<f64> for ElectronVolts {
    type Output = Self;
    #[verified_engine::verified]
    fn div(self, scalar: f64) -> Self {
        Self(self.0 / scalar)
    }
}

impl Div for ElectronVolts {
    type Output = f64;
    #[verified_engine::verified]
    fn div(self, other: Self) -> f64 {
        self.0 / other.0
    }
}

impl Neg for ElectronVolts {
    type Output = Self;
    #[verified_engine::verified]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

/// Temperature in Kelvin (`K`).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Kelvin(f64);

impl Kelvin {
    /// Creates a new `Kelvin` instance.
    #[verified_engine::verified]
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    /// Returns the value as `f64`.
    #[verified_engine::verified]
    pub fn as_f64(&self) -> f64 {
        self.0
    }

    /// Returns thermal energy $k_B T$ in SI Joules (`J`) using [`crate::physics::stat_mech::KB`].
    #[verified_engine::verified]
    pub fn to_joules(self) -> f64 {
        crate::physics::stat_mech::KB * self.0
    }

    /// Constructs a `Kelvin` temperature from thermal energy $k_B T$ in SI Joules (`J`).
    #[verified_engine::verified]
    pub fn from_joules(joules: f64) -> Self {
        Self(joules / crate::physics::stat_mech::KB)
    }

    /// Returns thermal energy $k_B T$ in [`ElectronVolts`] using [`crate::physics::stat_mech::KB`].
    #[verified_engine::verified]
    pub fn thermal_energy_ev(self) -> ElectronVolts {
        ElectronVolts::from_joules(self.to_joules())
    }

    /// Constructs a `Kelvin` temperature from thermal energy $k_B T$ given in [`ElectronVolts`].
    #[verified_engine::verified]
    pub fn from_thermal_energy_ev(energy: ElectronVolts) -> Self {
        Self::from_joules(energy.to_joules())
    }

    /// Raises the numerical Kelvin magnitude to power `n`.
    #[verified_engine::verified]
    pub fn powf(&self, n: f64) -> Self {
        Self::new(self.0.powf(n))
    }

    /// Computes the square root of the numerical Kelvin magnitude.
    #[verified_engine::verified]
    pub fn sqrt(&self) -> Self {
        Self::new(self.0.sqrt())
    }

    /// Computes the natural logarithm of the numerical Kelvin magnitude.
    #[verified_engine::verified]
    pub fn ln(&self) -> Self {
        Self::new(self.0.ln())
    }

    /// Computes the exponential of the numerical Kelvin magnitude.
    #[verified_engine::verified]
    pub fn exp(&self) -> Self {
        Self::new(self.0.exp())
    }

    /// Returns the absolute Kelvin magnitude.
    #[verified_engine::verified]
    pub fn abs(&self) -> Self {
        Self::new(self.0.abs())
    }
}

impl fmt::Display for Kelvin {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:.2} K", self.0)
    }
}

// [cite:mmwave_radiotherapy_setup]

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_codata_ev_joule_and_kelvin_thermal_bridges() {
        let one_ev = ElectronVolts::new(1.0);
        assert!((one_ev.to_joules() - ELEMENTARY_CHARGE_C).abs() < 1e-34);
        let roundtrip = ElectronVolts::from_joules(one_ev.to_joules());
        assert!((roundtrip.as_f64() - 1.0).abs() < 1e-15);

        let room_t = Kelvin::new(300.0);
        let kt_ev = room_t.thermal_energy_ev();
        // CODATA: k_B * 300 K / e = 0.02585199978643553... eV
        assert!((kt_ev.as_f64() - 0.025_851_999_8).abs() < 1e-9);
        let recovered_t = Kelvin::from_thermal_energy_ev(kt_ev);
        assert!((recovered_t.as_f64() - 300.0).abs() < 1e-12);
    }
}
