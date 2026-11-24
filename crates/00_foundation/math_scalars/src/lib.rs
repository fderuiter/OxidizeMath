#![no_std]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
#![doc = include_str!("../README.md")]
// [cite:math_scalars]

pub mod aliases;
pub mod dim;
pub mod frame;
pub mod int;
pub mod quantity;
pub mod scalar;
pub mod temperature;

pub use aliases::{
    CubicMeters, DimensionlessVal, Femtometers, FemtometersPerSecond, GyromagneticRatioSi, Hertz,
    Joules, JoulesPerKelvin, JoulesPerMole, JoulesPerMoleKelvin, JoulesSecond, Kilograms,
    Kiloparsecs, METERS_PER_KILOPARSEC, MeV, MeVFm, MeVSecond, Meters, MetersPerSecond,
    Millimeters, Newtons, Pascals, Seconds, SiGravitationalConstant, SquareMeters, Tesla,
    ViewportLength, Watts,
};
pub use dim::{
    AddDim, Dim, DimAcceleration, DimAction, DimAmount, DimArea, DimCurrent, DimEnergy,
    DimEnergyDensity, DimEnergyLength, DimForce, DimFrequency, DimGravitationalConstant,
    DimGyromagneticRatio, DimHeatCapacity, DimLength, DimLuminousIntensity, DimMagneticFluxDensity,
    DimMass, DimMolarEnergy, DimMolarHeatCapacity, DimPower, DimPressure, DimSpec, DimTemperature,
    DimTime, DimVelocity, DimVolume, Dimensionless, HalfDim, NegDim, SubDim,
};
pub use frame::{
    GalacticKpcFrame, MillimeterFrame, NuclearMeVFmFrame, SiFrame, UnitFrame, ViewportFrame,
};
pub use int::{
    AddExp, HalfExp, N1, N2, N3, N4, N5, N6, N7, N8, NInt, Nat2, Nat3, Nat4, Nat5, Nat6, Nat7,
    Nat8, NegExp, NonZeroNat, P1, P2, P3, P4, P5, P6, P7, P8, PInt, SubExp, Succ, TypeInt, U1, Z0,
};
pub use quantity::Quantity;
pub use scalar::Scalar;
pub use temperature::{CELSIUS_ZERO_KELVIN_OFFSET, Celsius, Kelvin};

#[cfg(test)]
mod tests;
