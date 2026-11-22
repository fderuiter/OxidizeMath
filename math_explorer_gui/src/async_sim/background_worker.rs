//! Typed `BackgroundWorker<T>` wrapper around [`super::SimulationController`].

use super::{
    SimCommand, SimCommandError, SimWorkerState, SimulationController, SimulationRunner,
    StateSnapshot, TimelineMetadata,
};

/// Generic background worker wrapper struct that manages offloaded simulation runners
/// and provides thread-safe channel communication between the UI and compute threads.
#[allow(missing_docs)]
pub struct BackgroundWorker<T: SimulationRunner> {
    controller: SimulationController,
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: SimulationRunner> BackgroundWorker<T> {
    /// Spawns a background worker executing the given simulation runner.
    #[allow(missing_docs)]
    pub fn new(runner: T) -> Self {
        Self {
            controller: SimulationController::new(runner),
            _marker: std::marker::PhantomData,
        }
    }

    #[allow(missing_docs)]
    pub fn state(&self) -> SimWorkerState {
        self.controller.state()
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn set_state_for_test(&mut self, state: SimWorkerState) {
        self.controller.set_state_for_test(state);
    }

    #[allow(missing_docs)]
    pub fn is_running(&self) -> bool {
        self.controller.is_running()
    }

    #[allow(missing_docs)]
    pub fn timeline_metadata(&self) -> Option<TimelineMetadata> {
        self.controller.timeline_metadata()
    }

    #[allow(missing_docs)]
    pub fn send_command(&mut self, cmd: SimCommand) -> Result<(), SimCommandError> {
        self.controller.send_command(cmd)
    }

    /// Resets the worker in place or respawns a fresh runner when disconnected.
    #[allow(missing_docs)]
    pub fn reset_or_respawn<F>(&mut self, make_runner: F) -> bool
    where
        F: FnOnce() -> T,
    {
        self.controller.reset_or_respawn(make_runner)
    }

    #[allow(missing_docs)]
    pub fn latest_snapshot(&self) -> Option<&StateSnapshot> {
        self.controller.latest_snapshot()
    }

    #[allow(missing_docs)]
    pub fn update(&mut self) -> Option<&StateSnapshot> {
        self.controller.update()
    }

    #[allow(missing_docs)]
    pub fn controller(&self) -> &SimulationController {
        &self.controller
    }

    #[allow(missing_docs)]
    pub fn controller_mut(&mut self) -> &mut SimulationController {
        &mut self.controller
    }

    #[allow(missing_docs)]
    pub fn into_controller(self) -> SimulationController {
        self.controller
    }
}

impl<T: SimulationRunner> std::ops::Deref for BackgroundWorker<T> {
    type Target = SimulationController;

    fn deref(&self) -> &Self::Target {
        &self.controller
    }
}

impl<T: SimulationRunner> std::ops::DerefMut for BackgroundWorker<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.controller
    }
}

impl<T: SimulationRunner> From<SimulationController> for BackgroundWorker<T> {
    fn from(controller: SimulationController) -> Self {
        Self {
            controller,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: SimulationRunner> From<BackgroundWorker<T>> for SimulationController {
    fn from(worker: BackgroundWorker<T>) -> Self {
        worker.controller
    }
}
