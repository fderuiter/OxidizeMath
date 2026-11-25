//! Domain-ergonomic SI, nuclear `MeV`-`fm`, astrophysical `kpc`, viewport, and `mm` quantity
//! aliases and explicit cross-frame conversion methods.
// [cite:math_scalars]

use crate::dim::{
    DimAction, DimArea, DimEnergy, DimEnergyLength, DimForce, DimFrequency,
    DimGravitationalConstant, DimGyromagneticRatio, DimHeatCapacity, DimLength,
    DimMagneticFluxDensity, DimMass, DimMolarEnergy, DimMolarHeatCapacity, DimPower, DimPressure,
    DimTime, DimVelocity, DimVolume, Dimensionless,
};
use crate::frame::{GalacticKpcFrame, MillimeterFrame, NuclearMeVFmFrame, SiFrame, ViewportFrame};
use crate::quantity::Quantity;
use crate::scalar::Scalar;

/// Exact conversion factor from kiloparsecs to SI meters ($1\,\text{kpc} = 3.085677581491367 \times 10^{19}\,\text{m}$).
pub const METERS_PER_KILOPARSEC: f64 = 3.085_677_581_491_367e19;

/// Coherent SI dimensionless quantity.
pub type DimensionlessVal<T = f64> = Quantity<T, Dimensionless, SiFrame>;
/// Coherent SI length in meters ($\text{m}$).
pub type Meters<T = f64> = Quantity<T, DimLength, SiFrame>;
/// Coherent SI area in square meters ($\text{m}^2$).
pub type SquareMeters<T = f64> = Quantity<T, DimArea, SiFrame>;
/// Coherent SI volume in cubic meters ($\text{m}^3$).
pub type CubicMeters<T = f64> = Quantity<T, DimVolume, SiFrame>;
/// Coherent SI mass in kilograms ($\text{kg}$).
pub type Kilograms<T = f64> = Quantity<T, DimMass, SiFrame>;
/// Coherent SI time in seconds ($\text{s}$).
pub type Seconds<T = f64> = Quantity<T, DimTime, SiFrame>;
/// Coherent SI frequency in hertz ($\text{s}^{-1}$).
pub type Hertz<T = f64> = Quantity<T, DimFrequency, SiFrame>;
/// Coherent SI speed in meters per second ($\text{m}\cdot\text{s}^{-1}$).
pub type MetersPerSecond<T = f64> = Quantity<T, DimVelocity, SiFrame>;
/// Coherent SI force in newtons ($\text{kg}\cdot\text{m}\cdot\text{s}^{-2}$).
pub type Newtons<T = f64> = Quantity<T, DimForce, SiFrame>;
/// Coherent SI pressure / energy density in pascals ($\text{J}\cdot\text{m}^{-3}$).
pub type Pascals<T = f64> = Quantity<T, DimPressure, SiFrame>;
/// Coherent SI energy in joules ($\text{kg}\cdot\text{m}^2\cdot\text{s}^{-2}$).
pub type Joules<T = f64> = Quantity<T, DimEnergy, SiFrame>;
/// Coherent SI power in watts ($\text{J}\cdot\text{s}^{-1}$).
pub type Watts<T = f64> = Quantity<T, DimPower, SiFrame>;
/// Coherent SI action in joule-seconds ($\text{J}\cdot\text{s}$).
pub type JoulesSecond<T = f64> = Quantity<T, DimAction, SiFrame>;
/// Coherent SI magnetic flux density in tesla ($\text{kg}\cdot\text{s}^{-2}\cdot\text{A}^{-1}$).
pub type Tesla<T = f64> = Quantity<T, DimMagneticFluxDensity, SiFrame>;
/// Coherent SI gyromagnetic ratio ($\text{rad}\cdot\text{s}^{-1}\cdot\text{T}^{-1}$).
pub type GyromagneticRatioSi<T = f64> = Quantity<T, DimGyromagneticRatio, SiFrame>;
/// Coherent SI heat capacity / Boltzmann constant ($\text{J}\cdot\text{K}^{-1}$).
pub type JoulesPerKelvin<T = f64> = Quantity<T, DimHeatCapacity, SiFrame>;
/// Coherent SI molar energy ($\text{J}\cdot\text{mol}^{-1}$).
pub type JoulesPerMole<T = f64> = Quantity<T, DimMolarEnergy, SiFrame>;
/// Coherent SI molar heat capacity / gas constant ($\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}$).
pub type JoulesPerMoleKelvin<T = f64> = Quantity<T, DimMolarHeatCapacity, SiFrame>;
/// Coherent SI Newtonian gravitational constant ($\text{m}^3\cdot\text{kg}^{-1}\cdot\text{s}^{-2}$).
pub type SiGravitationalConstant<T = f64> = Quantity<T, DimGravitationalConstant, SiFrame>;

/// Nuclear length in femtometers ($1\,\text{fm} = 10^{-15}\,\text{m}$).
pub type Femtometers<T = f64> = Quantity<T, DimLength, NuclearMeVFmFrame>;
/// Nuclear speed in femtometers per second ($\text{fm}\cdot\text{s}^{-1}$).
pub type FemtometersPerSecond<T = f64> = Quantity<T, DimVelocity, NuclearMeVFmFrame>;
/// Nuclear energy or rest mass-energy in mega-electronvolts ($\text{MeV}$).
pub type MeV<T = f64> = Quantity<T, DimEnergy, NuclearMeVFmFrame>;
/// Nuclear coupling product in mega-electronvolt femtometers ($\text{MeV}\cdot\text{fm}$).
pub type MeVFm<T = f64> = Quantity<T, DimEnergyLength, NuclearMeVFmFrame>;
/// Nuclear action in mega-electronvolt seconds ($\text{MeV}\cdot\text{s}$).
pub type MeVSecond<T = f64> = Quantity<T, DimAction, NuclearMeVFmFrame>;

/// Astrophysical galactocentric or impact-plane length in kiloparsecs ($\text{kpc}$).
pub type Kiloparsecs<T = f64> = Quantity<T, DimLength, GalacticKpcFrame>;
/// Interactive 2D/3D viewport coordinate length.
pub type ViewportLength<T = f64> = Quantity<T, DimLength, ViewportFrame>;
/// Desktop-slicer calibrated length in millimeters ($\text{mm}$).
pub type Millimeters<T = f64> = Quantity<T, DimLength, MillimeterFrame>;

impl<T: Scalar> Kiloparsecs<T> {
    /// Explicitly converts kiloparsecs into coherent SI [`Meters`].
    #[inline(always)]
    pub fn to_meters(self) -> Meters<T> {
        Meters::new(self.get() * T::from_f64(METERS_PER_KILOPARSEC))
    }

    /// Explicitly converts coherent SI [`Meters`] into [`Kiloparsecs`].
    #[inline(always)]
    pub fn from_meters(meters: Meters<T>) -> Self {
        Self::new(meters.get() / T::from_f64(METERS_PER_KILOPARSEC))
    }

    /// Explicitly maps a physical [`Kiloparsecs`] length into [`ViewportLength`] given
    /// `kpc_per_viewport_unit` ($\text{kpc}/\text{viewport}$).
    #[inline(always)]
    pub fn to_viewport(self, kpc_per_viewport_unit: T) -> ViewportLength<T> {
        ViewportLength::new(self.get() / kpc_per_viewport_unit)
    }
}

impl<T: Scalar> ViewportLength<T> {
    /// Explicitly maps a [`ViewportLength`] into physical [`Kiloparsecs`] given
    /// `kpc_per_viewport_unit` ($\text{kpc}/\text{viewport}$).
    #[inline(always)]
    pub fn to_kiloparsecs(self, kpc_per_viewport_unit: T) -> Kiloparsecs<T> {
        Kiloparsecs::new(self.get() * kpc_per_viewport_unit)
    }
}

impl<T: Scalar> Millimeters<T> {
    /// Explicitly converts millimeters into coherent SI [`Meters`] ($1\,\text{mm} = 10^{-3}\,\text{m}$).
    #[inline(always)]
    pub fn to_meters(self) -> Meters<T> {
        Meters::new(self.get() * T::from_f64(1.0e-3))
    }

    /// Explicitly converts coherent SI [`Meters`] into [`Millimeters`].
    #[inline(always)]
    pub fn from_meters(meters: Meters<T>) -> Self {
        Self::new(meters.get() * T::from_f64(1.0e3))
    }
}
