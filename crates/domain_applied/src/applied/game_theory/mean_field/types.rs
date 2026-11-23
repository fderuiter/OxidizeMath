use crate::error::GameTheoryError;
use std::num::NonZeroUsize;

/// Represents a 1D spatial position.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Position(pub f64);

/// Represents a probability density value.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Density(pub f64);

/// Represents a 1D Mean Field Game (MFG) configuration.
///
/// # Field Descriptions
/// - `viscosity` ($\nu$): Diffusion parameter representing noise/randomness in agent motion.
/// - `time_horizon` ($T$): Duration of the game.
/// - `dt`, `dx`: Discretization steps.
#[derive(Clone, Debug, Copy, PartialEq)]
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

// Type states
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WithoutViscosity;
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WithViscosity(f64);

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WithoutTimeHorizon;
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WithTimeHorizon(f64);

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WithoutGridPoints;
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WithGridPoints(NonZeroUsize);

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WithoutTimeSteps;
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WithTimeSteps(NonZeroUsize);

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WithoutSpaceBounds;
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WithSpaceBounds {
    min: f64,
    max: f64,
}

/// Type-State Builder for `MFGConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mfg_builder_type_states_derives_and_unwrap_err() {
        let b0 = MFGConfigBuilder::default();
        assert_eq!(b0, MFGConfigBuilder::new());
        let def_without: WithoutViscosity = Default::default();
        assert_eq!(WithoutViscosity, def_without);
        assert!(!format!("{b0:?}").is_empty());

        let bad = b0.space_bounds(2.0, 1.0).unwrap_err();
        assert!(!format!("{bad:?}").is_empty());

        let nx = NonZeroUsize::new(11).unwrap();
        let nt = NonZeroUsize::new(20).unwrap();
        let ready = b0
            .viscosity(0.1)
            .time_horizon(1.0)
            .grid_points(nx)
            .time_steps(nt)
            .space_bounds(0.0, 1.0)
            .unwrap();
        assert_eq!(ready, ready);
        let cfg = ready.build().unwrap();
        assert_eq!(cfg, cfg);
        assert!(!format!("{cfg:?}").is_empty());
    }
}
