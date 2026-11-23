//! Unified feature-gated error re-exports across all OxidizeMath domains.

pub use math_commons::error::{FitError, MathError, MathResult};

#[cfg(feature = "ai")]
pub use domain_ai::error::AIError;

#[cfg(feature = "applied")]
pub use domain_applied::error::{
    BatteryError, EngineeringError, GameTheoryError, IsosurfaceError, LoraError,
    PharmacokineticsError,
};

#[cfg(feature = "biology")]
pub use domain_biology::error::HodgkinHuxleyError;

#[cfg(feature = "epidemiology")]
pub use domain_epidemiology::error::EpidemiologyError;

#[cfg(feature = "physics")]
pub use domain_physics::error::{
    ChaosError, DoseFluenceError, FluidError, HighEnergyError, RadarError, SolidStateError,
    StandardModelError, StatMechError,
};

#[cfg(feature = "pure_math")]
pub use pure_math::error::{
    CopulaError, Glicko2Error, KellyError, MarkovError, NumberTheoryError, OuError, TdaError,
    ZipError,
};
