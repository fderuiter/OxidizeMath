use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Energy in Electron Volts (eV).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ElectronVolts(f64);

impl ElectronVolts {
    /// Creates a new `ElectronVolts` instance.
    #[verified_engine::verified]
    pub fn new(value: f64) -> Self {
        Self(value)
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

    #[allow(missing_docs)]
    pub fn powf(&self, n: f64) -> Self {
        Self::new(self.0.powf(n))
    }

    #[allow(missing_docs)]
    pub fn sqrt(&self) -> Self {
        Self::new(self.0.sqrt())
    }

    #[allow(missing_docs)]
    pub fn ln(&self) -> Self {
        Self::new(self.0.ln())
    }

    #[allow(missing_docs)]
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

/// Temperature in Kelvin (K).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    #[allow(missing_docs)]
    pub fn powf(&self, n: f64) -> Self {
        Self::new(self.0.powf(n))
    }

    #[allow(missing_docs)]
    pub fn sqrt(&self) -> Self {
        Self::new(self.0.sqrt())
    }

    #[allow(missing_docs)]
    pub fn ln(&self) -> Self {
        Self::new(self.0.ln())
    }

    #[allow(missing_docs)]
    pub fn exp(&self) -> Self {
        Self::new(self.0.exp())
    }

    #[allow(missing_docs)]
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
