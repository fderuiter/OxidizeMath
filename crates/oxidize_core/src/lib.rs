#![doc = include_str!("../README.md")]
use serde::{Deserialize, Serialize};

#[allow(missing_docs)]
pub trait ModelConfig: Clone + Serialize + Deserialize<'static> {}

#[allow(missing_docs)]
pub trait ModelState: Clone {}

#[allow(missing_docs)]
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
#[allow(dead_code)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Serialize, Deserialize)]
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
        config: BooleanMaskConfig,
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
            let state = BooleanMaskState {
                grid: vec![false; config.width * config.height],
                width: config.width,
                height: config.height,
            };
            Ok(Self { config, state })
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

        // Initial state should be all false
        assert!(sim.get_state().grid.iter().all(|&x| !x));

        // Step
        sim.step().unwrap();

        // State should be all true
        assert!(sim.get_state().grid.iter().all(|&x| x));
    }
}
#[allow(missing_docs)]
pub mod ast_visitor;
#[allow(missing_docs)]
pub mod boundary;
#[allow(missing_docs)]
pub mod double_buffer;
#[allow(missing_docs)]
pub mod grid;
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
