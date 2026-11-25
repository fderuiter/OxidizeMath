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
#[derive(Clone, Debug, PartialEq)]
pub struct MFGConfig {
    /// Diffusion viscosity coefficient $\nu \ge 0$.
    pub viscosity: f64,
    /// Terminal game horizon $T > 0$.
    pub time_horizon: f64,
    /// Number of uniform temporal subintervals $N_t$.
    pub time_steps: NonZeroUsize,
    /// Number of uniform spatial grid nodes $N_x$.
    pub grid_points: NonZeroUsize,
    /// Uniform time step $\Delta t = T / N_t$.
    pub dt: f64,
    /// Uniform spatial step $\Delta x = (x_{\max} - x_{\min}) / (N_x - 1)$.
    pub dx: f64,
    /// Lower spatial domain boundary $x_{\min}$.
    pub space_min: f64,
    /// Upper spatial domain boundary $x_{\max}$.
    pub space_max: f64,
}

/// Builder type-state indicating viscosity has not yet been configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WithoutViscosity;
/// Builder type-state storing a configured viscosity coefficient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WithViscosity(f64);

/// Builder type-state indicating the time horizon has not yet been configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WithoutTimeHorizon;
/// Builder type-state storing a configured time horizon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WithTimeHorizon(f64);

/// Builder type-state indicating spatial grid resolution has not yet been configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WithoutGridPoints;
/// Builder type-state storing a configured spatial grid node count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WithGridPoints(NonZeroUsize);

/// Builder type-state indicating temporal step count has not yet been configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WithoutTimeSteps;
/// Builder type-state storing a configured temporal step count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WithTimeSteps(NonZeroUsize);

/// Builder type-state indicating spatial domain bounds have not yet been configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WithoutSpaceBounds;
/// Builder type-state storing validated spatial domain bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    #[verified_engine::verified]
    fn test_mfg_config_builder_derives_and_unwrap_err() {
        let b0 = MFGConfigBuilder::default();
        assert_eq!(b0, MFGConfigBuilder::new());
        assert!(!format!("{b0:?}").is_empty());

        let nx = NonZeroUsize::new(11).unwrap();
        let nt = NonZeroUsize::new(20).unwrap();
        let ready = b0
            .viscosity(0.1)
            .time_horizon(1.0)
            .grid_points(nx)
            .time_steps(nt)
            .space_bounds(0.0, 1.0)
            .unwrap();
        let ready_copy = ready;
        assert_eq!(ready, ready_copy);
        assert!(!format!("{ready:?}").is_empty());

        let cfg = ready.build().unwrap();
        assert_eq!(cfg, cfg.clone());
        assert!(!format!("{cfg:?}").is_empty());

        let bounds_err = MFGConfigBuilder::new().space_bounds(2.0, 1.0).unwrap_err();
        assert!(!format!("{bounds_err:?}").is_empty());

        let one = NonZeroUsize::new(1).unwrap();
        let grid_err = MFGConfigBuilder::new()
            .viscosity(0.1)
            .time_horizon(1.0)
            .grid_points(one)
            .time_steps(nt)
            .space_bounds(0.0, 1.0)
            .unwrap()
            .build()
            .unwrap_err();
        assert!(!format!("{grid_err:?}").is_empty());
    }
}
