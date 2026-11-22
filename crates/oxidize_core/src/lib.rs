#![doc = include_str!("../README.md")]

/// Configuration contract for simulation models.
///
/// Implementors must support cloning, and serialization/deserialization when `serde` is enabled.
///
/// # Example
/// ```
/// use oxidize_core::ModelConfig;
///
/// #[derive(Clone, PartialEq, Debug)]
/// #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// struct GridConfig {
///     width: usize,
///     height: usize,
/// }
/// impl ModelConfig for GridConfig {}
///
/// let config = GridConfig { width: 32, height: 32 };
/// assert_eq!(config.width, 32);
/// ```
#[cfg(feature = "serde")]
pub trait ModelConfig: Clone + serde::Serialize + serde::Deserialize<'static> {}

/// Configuration contract for simulation models (minimal no-`serde` build).
#[cfg(not(feature = "serde"))]
pub trait ModelConfig: Clone {}

/// State representation contract for simulation models.
///
/// Implementors must be cloneable to support state snapshots and checkpoints.
///
/// # Example
/// ```
/// use oxidize_core::ModelState;
///
/// #[derive(Clone, PartialEq, Debug)]
/// struct HeatState {
///     temperature: f64,
/// }
/// impl ModelState for HeatState {}
///
/// let state = HeatState { temperature: 293.15 };
/// let cloned = state.clone();
/// assert_eq!(state, cloned);
/// ```
pub trait ModelState: Clone {}

/// Core simulation model interface defining the lifecycle of a simulation step.
///
/// # Example
/// ```
/// use oxidize_core::{ModelConfig, ModelState, SimulationModel};
///
/// #[derive(Clone)]
/// #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// struct CounterConfig {
///     start: i64,
/// }
/// impl ModelConfig for CounterConfig {}
///
/// #[derive(Clone, Debug, PartialEq)]
/// struct CounterState {
///     count: i64,
/// }
/// impl ModelState for CounterState {}
///
/// #[derive(Debug, thiserror::Error)]
/// #[error("Counter simulation error: {0}")]
/// struct CounterError(String);
///
/// struct CounterSim {
///     state: CounterState,
/// }
///
/// impl SimulationModel for CounterSim {
///     type Config = CounterConfig;
///     type State = CounterState;
///     type Error = CounterError;
///
///     fn initialize<R: rand::RngCore>(config: Self::Config, _provider: R) -> Result<Self, Self::Error> {
///         Ok(Self {
///             state: CounterState { count: config.start },
///         })
///     }
///
///     fn step(&mut self) -> Result<(), Self::Error> {
///         self.state.count += 1;
///         Ok(())
///     }
///
///     fn get_state(&self) -> Self::State {
///         self.state.clone()
///     }
/// }
///
/// let mut sim = CounterSim::initialize(CounterConfig { start: 10 }, oxidize_core::rng::OxidizeRng::new(42)).unwrap();
/// assert_eq!(sim.get_state().count, 10);
/// sim.step().unwrap();
/// assert_eq!(sim.get_state().count, 11);
/// ```
pub trait SimulationModel: Sized {
    #[allow(missing_docs)]
    type Config: ModelConfig;
    #[allow(missing_docs)]
    type State: ModelState;
    #[allow(missing_docs)]
    type Error: std::error::Error + Send + Sync + 'static;

    #[allow(missing_docs)]
    fn initialize<R: rand::RngCore>(config: Self::Config, provider: R)
    -> Result<Self, Self::Error>;
    #[allow(missing_docs)]
    fn step(&mut self) -> Result<(), Self::Error>;
    #[allow(missing_docs)]
    fn get_state(&self) -> Self::State;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct BooleanMaskConfig {
        pub width: usize,
        pub height: usize,
    }

    impl ModelConfig for BooleanMaskConfig {}

    #[derive(Clone)]
    pub struct BooleanMaskState {
        pub grid: Vec<bool>,
        pub width: usize,
        pub height: usize,
    }

    impl ModelState for BooleanMaskState {}

    pub struct BooleanMaskSimulation {
        state: BooleanMaskState,
    }

    #[derive(Debug, thiserror::Error)]
    pub enum SimulationError {
        #[error("Initialization error")]
        InitError,
    }

    impl SimulationModel for BooleanMaskSimulation {
        type Config = BooleanMaskConfig;
        type State = BooleanMaskState;
        type Error = SimulationError;

        fn initialize<R: rand::RngCore>(
            config: Self::Config,
            _provider: R,
        ) -> Result<Self, Self::Error> {
            if config.width == 0 || config.height == 0 {
                return Err(SimulationError::InitError);
            }
            let state = BooleanMaskState {
                grid: vec![false; config.width * config.height],
                width: config.width,
                height: config.height,
            };
            Ok(Self { state })
        }

        fn step(&mut self) -> Result<(), Self::Error> {
            // Flip all bits
            for cell in self.state.grid.iter_mut() {
                *cell = !*cell;
            }
            Ok(())
        }

        fn get_state(&self) -> Self::State {
            self.state.clone()
        }
    }

    #[test]
    fn test_boolean_mask_simulation() {
        let config = BooleanMaskConfig {
            width: 10,
            height: 10,
        };
        let provider = rng::OxidizeRng::new(42);
        let mut sim = BooleanMaskSimulation::initialize(config, provider).unwrap();

        let state = sim.get_state();
        assert_eq!(state.width, 10);
        assert_eq!(state.height, 10);

        // Initial state should be all false
        assert!(state.grid.iter().all(|&x| !x));

        // Step
        sim.step().unwrap();

        // State should be all true
        assert!(sim.get_state().grid.iter().all(|&x| x));

        // Zero dimensions should fail initialization with InitError
        let zero_width_config = BooleanMaskConfig {
            width: 0,
            height: 10,
        };
        assert!(matches!(
            BooleanMaskSimulation::initialize(zero_width_config, rng::OxidizeRng::new(42)),
            Err(SimulationError::InitError)
        ));

        let zero_height_config = BooleanMaskConfig {
            width: 10,
            height: 0,
        };
        assert!(matches!(
            BooleanMaskSimulation::initialize(zero_height_config, rng::OxidizeRng::new(42)),
            Err(SimulationError::InitError)
        ));
    }
}
#[allow(missing_docs)]
pub mod ast_visitor;
#[allow(missing_docs)]
pub mod boundary;
#[allow(missing_docs)]
pub mod iteration;
#[allow(missing_docs)]
pub mod mesh;
#[allow(missing_docs)]
pub mod path_utils;
#[allow(missing_docs)]
pub mod rng;
#[allow(missing_docs)]
pub mod traceability;
#[allow(missing_docs)]
pub mod vfs;
#[allow(missing_docs)]
pub mod workspace;

#[allow(missing_docs)]
pub mod prelude {
    pub use crate::{ModelConfig, ModelState, SimulationModel};
}
