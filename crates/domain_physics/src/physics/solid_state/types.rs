use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Energy in Electron Volts (`eV`), restricted to dimensionally homogeneous linear operations and
/// dimensionless ratios (`ElectronVolts / ElectronVolts -> f64`).
///
/// Dimension-erasing operations such as `.ln()`, `.exp()`, `.sqrt()`, `.powf()`, and
/// self-multiplication (`ElectronVolts * ElectronVolts`) are rejected at compile time:
///
/// ```compile_fail
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::solid_state::types::ElectronVolts;
/// let _ = ElectronVolts::new(1.5).ln();
/// ```
///
/// ```compile_fail
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::solid_state::types::ElectronVolts;
/// let _ = ElectronVolts::new(1.5).exp();
/// ```
///
/// ```compile_fail
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::solid_state::types::ElectronVolts;
/// let _ = ElectronVolts::new(1.5) * ElectronVolts::new(0.5);
/// ```
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct ElectronVolts(f64);

impl ElectronVolts {
    /// Creates a new `ElectronVolts` instance.
    #[verified_engine::verified]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Returns the wrapped scalar value in `eV` by copy.
    #[verified_engine::verified]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// Returns the wrapped scalar value in `eV`.
    #[verified_engine::verified]
    pub const fn value(&self) -> f64 {
        self.0
    }

    /// Returns the value as `f64`.
    #[verified_engine::verified]
    pub const fn as_f64(&self) -> f64 {
        self.0
    }

    /// Returns the absolute energy $|E|$ in `eV`.
    #[verified_engine::verified]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Evaluates the dimensionally homogeneous quasiparticle dispersion $\sqrt{E_1^2 + E_2^2}$ in `eV`.
    #[verified_engine::verified]
    pub fn hypot(self, other: Self) -> Self {
        Self(self.0.hypot(other.0))
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

/// Thermodynamic temperature in Kelvin (`K`), restricted to linear scaling and dimensionless ratios.
///
/// ```compile_fail
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::solid_state::types::Kelvin;
/// let _ = Kelvin::new(300.0).sqrt();
/// ```
///
/// ```compile_fail
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::solid_state::types::Kelvin;
/// let _ = Kelvin::new(300.0).exp();
/// ```
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Kelvin(f64);

impl Kelvin {
    /// Creates a new `Kelvin` instance.
    #[verified_engine::verified]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Returns the wrapped scalar value in `K` by copy.
    #[verified_engine::verified]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// Returns the wrapped scalar value in `K`.
    #[verified_engine::verified]
    pub const fn value(&self) -> f64 {
        self.0
    }

    /// Returns the value as `f64`.
    #[verified_engine::verified]
    pub const fn as_f64(&self) -> f64 {
        self.0
    }

    /// Returns $|T|$ in `K`.
    #[verified_engine::verified]
    pub fn abs(&self) -> Self {
        Self::new(self.0.abs())
    }
}

impl Add for Kelvin {
    type Output = Self;
    #[verified_engine::verified]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl Sub for Kelvin {
    type Output = Self;
    #[verified_engine::verified]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl Mul<f64> for Kelvin {
    type Output = Self;
    #[verified_engine::verified]
    fn mul(self, scalar: f64) -> Self {
        Self(self.0 * scalar)
    }
}

impl Mul<Kelvin> for f64 {
    type Output = Kelvin;
    #[verified_engine::verified]
    fn mul(self, k: Kelvin) -> Kelvin {
        Kelvin(self * k.0)
    }
}

impl Div<f64> for Kelvin {
    type Output = Self;
    #[verified_engine::verified]
    fn div(self, scalar: f64) -> Self {
        Self(self.0 / scalar)
    }
}

impl Div for Kelvin {
    type Output = f64;
    #[verified_engine::verified]
    fn div(self, other: Self) -> f64 {
        self.0 / other.0
    }
}

impl Neg for Kelvin {
    type Output = Self;
    #[verified_engine::verified]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl fmt::Display for Kelvin {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:.2} K", self.0)
    }
}
