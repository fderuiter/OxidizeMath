use crate::error::GameTheoryError;
use std::num::NonZeroUsize;

/// Represents a 1D spatial position.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Position(pub f64);

/// Represents a probability density value.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Density(pub f64);

/// Represents a 1D Mean Field Game (MFG) configuration.
///
/// # Field Descriptions
/// - `viscosity` ($\nu$): Diffusion parameter representing noise/randomness in agent motion.
/// - `time_horizon` ($T$): Duration of the game.
/// - `dt`, `dx`: Discretization steps.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct MFGConfig {
    #[allow(missing_docs)]
    pub viscosity: f64, // nu
    #[allow(missing_docs)]
    pub time_horizon: f64, // T
    #[allow(missing_docs)]
    pub time_steps: NonZeroUsize, // Nt
    #[allow(missing_docs)]
    pub grid_points: NonZeroUsize, // Nx
    #[allow(missing_docs)]
    pub dt: f64,
    #[allow(missing_docs)]
    pub dx: f64,
    #[allow(missing_docs)]
    pub space_min: f64,
    #[allow(missing_docs)]
    pub space_max: f64,
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for MFGConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct MFGConfigRaw {
            viscosity: f64,
            time_horizon: f64,
            time_steps: NonZeroUsize,
            grid_points: NonZeroUsize,
            dt: f64,
            dx: f64,
            space_min: f64,
            space_max: f64,
        }
        let raw = MFGConfigRaw::deserialize(deserializer)?;
        if ![
            raw.viscosity,
            raw.time_horizon,
            raw.dt,
            raw.dx,
            raw.space_min,
            raw.space_max,
        ]
        .into_iter()
        .all(f64::is_finite)
        {
            return Err(serde::de::Error::custom(
                "MFGConfig float parameters must be finite",
            ));
        }
        if raw.grid_points.get() <= 1 || raw.space_max <= raw.space_min {
            return Err(serde::de::Error::custom(
                "MFGConfig requires grid_points > 1 and space_max > space_min",
            ));
        }
        Ok(Self {
            viscosity: raw.viscosity,
            time_horizon: raw.time_horizon,
            time_steps: raw.time_steps,
            grid_points: raw.grid_points,
            dt: raw.dt,
            dx: raw.dx,
            space_min: raw.space_min,
            space_max: raw.space_max,
        })
    }
}

// Type states
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct WithoutViscosity;
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct WithViscosity(f64);

#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct WithoutTimeHorizon;
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct WithTimeHorizon(f64);

#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct WithoutGridPoints;
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct WithGridPoints(NonZeroUsize);

#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct WithoutTimeSteps;
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct WithTimeSteps(NonZeroUsize);

#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct WithoutSpaceBounds;
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct WithSpaceBounds {
    min: f64,
    max: f64,
}

/// Type-State Builder for `MFGConfig`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MFGConfigBuilder<V, T, G, TS, S> {
    viscosity: V,
    time_horizon: T,
    grid_points: G,
    time_steps: TS,
    space_bounds: S,
}

impl
    MFGConfigBuilder<
        WithoutViscosity,
        WithoutTimeHorizon,
        WithoutGridPoints,
        WithoutTimeSteps,
        WithoutSpaceBounds,
    >
{
    /// Creates a new Type-State builder.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self {
            viscosity: WithoutViscosity,
            time_horizon: WithoutTimeHorizon,
            grid_points: WithoutGridPoints,
            time_steps: WithoutTimeSteps,
            space_bounds: WithoutSpaceBounds,
        }
    }
}

impl Default
    for MFGConfigBuilder<
        WithoutViscosity,
        WithoutTimeHorizon,
        WithoutGridPoints,
        WithoutTimeSteps,
        WithoutSpaceBounds,
    >
{
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, G, TS, S> MFGConfigBuilder<WithoutViscosity, T, G, TS, S> {
    /// Sets the viscosity (nu).
    #[verified_engine::verified]
    pub fn viscosity(self, viscosity: f64) -> MFGConfigBuilder<WithViscosity, T, G, TS, S> {
        MFGConfigBuilder {
            viscosity: WithViscosity(viscosity),
            time_horizon: self.time_horizon,
            grid_points: self.grid_points,
            time_steps: self.time_steps,
            space_bounds: self.space_bounds,
        }
    }
}

impl<V, G, TS, S> MFGConfigBuilder<V, WithoutTimeHorizon, G, TS, S> {
    /// Sets the time horizon (T).
    #[verified_engine::verified]
    pub fn time_horizon(self, time_horizon: f64) -> MFGConfigBuilder<V, WithTimeHorizon, G, TS, S> {
        MFGConfigBuilder {
            viscosity: self.viscosity,
            time_horizon: WithTimeHorizon(time_horizon),
            grid_points: self.grid_points,
            time_steps: self.time_steps,
            space_bounds: self.space_bounds,
        }
    }
}

impl<V, T, TS, S> MFGConfigBuilder<V, T, WithoutGridPoints, TS, S> {
    /// Sets the number of grid points (Nx).
    #[verified_engine::verified]
    pub fn grid_points(
        self,
        grid_points: NonZeroUsize,
    ) -> MFGConfigBuilder<V, T, WithGridPoints, TS, S> {
        MFGConfigBuilder {
            viscosity: self.viscosity,
            time_horizon: self.time_horizon,
            grid_points: WithGridPoints(grid_points),
            time_steps: self.time_steps,
            space_bounds: self.space_bounds,
        }
    }
}

impl<V, T, G, S> MFGConfigBuilder<V, T, G, WithoutTimeSteps, S> {
    /// Sets the number of time steps (Nt).
    #[verified_engine::verified]
    pub fn time_steps(
        self,
        time_steps: NonZeroUsize,
    ) -> MFGConfigBuilder<V, T, G, WithTimeSteps, S> {
        MFGConfigBuilder {
            viscosity: self.viscosity,
            time_horizon: self.time_horizon,
            grid_points: self.grid_points,
            time_steps: WithTimeSteps(time_steps),
            space_bounds: self.space_bounds,
        }
    }
}

impl<V, T, G, TS> MFGConfigBuilder<V, T, G, TS, WithoutSpaceBounds> {
    /// Sets the space bounds.
    #[verified_engine::verified]
    pub fn space_bounds(
        self,
        space_min: f64,
        space_max: f64,
    ) -> Result<MFGConfigBuilder<V, T, G, TS, WithSpaceBounds>, GameTheoryError> {
        if space_max <= space_min {
            return Err(GameTheoryError::InvalidParameter {
                name: "space_bounds".to_string(),
                value: space_max - space_min,
            });
        }
        Ok(MFGConfigBuilder {
            viscosity: self.viscosity,
            time_horizon: self.time_horizon,
            grid_points: self.grid_points,
            time_steps: self.time_steps,
            space_bounds: WithSpaceBounds {
                min: space_min,
                max: space_max,
            },
        })
    }
}

impl
    MFGConfigBuilder<WithViscosity, WithTimeHorizon, WithGridPoints, WithTimeSteps, WithSpaceBounds>
{
    /// Builds the `MFGConfig` struct.
    /// This method is only available when all required fields have been set.
    #[verified_engine::verified]
    pub fn build(self) -> Result<MFGConfig, GameTheoryError> {
        let viscosity = self.viscosity.0;
        let time_horizon = self.time_horizon.0;
        let grid_points = self.grid_points.0;
        let time_steps = self.time_steps.0;
        let space_min = self.space_bounds.min;
        let space_max = self.space_bounds.max;

        if grid_points.get() <= 1 {
            return Err(GameTheoryError::InvalidParameter {
                name: "grid_points".to_string(),
                value: grid_points.get() as f64,
            });
        }

        let dt = time_horizon / (time_steps.get() as f64);
        let dx = (space_max - space_min) / ((grid_points.get() - 1) as f64);

        Ok(MFGConfig {
            viscosity,
            time_horizon,
            time_steps,
            grid_points,
            dt,
            dx,
            space_min,
            space_max,
        })
    }
}
