//! Typed `BackgroundWorker<T>` wrapper around [`super::SimulationController`].

use super::{
    SimCommand, SimCommandError, SimWorkerState, SimulationController, SimulationRunner,
    StateSnapshot, TimelineMetadata,
};

/// Generic background worker wrapper struct that manages offloaded simulation runners
/// and provides thread-safe channel communication between the UI and compute threads.
pub struct BackgroundWorker<T: SimulationRunner> {
    controller: SimulationController,
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: SimulationRunner> BackgroundWorker<T> {
    /// Spawns a background worker executing the given simulation runner.
    pub fn new(runner: T) -> Self {
        Self {
            controller: SimulationController::new(runner),
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns the current lifecycle state of the background simulation worker.
    pub fn state(&self) -> SimWorkerState {
        self.controller.state()
    }

    /// Overrides the cached worker state for deterministic state-machine unit tests.
    #[cfg(test)]
    pub fn set_state_for_test(&mut self, state: SimWorkerState) {
        self.controller.set_state_for_test(state);
    }

    /// Returns `true` when the worker is actively advancing frames.
    pub fn is_running(&self) -> bool {
        self.controller.is_running()
    }

    /// Returns the latest scrubbable timeline bounds when checkpointing is enabled.
    pub fn timeline_metadata(&self) -> Option<TimelineMetadata> {
        self.controller.timeline_metadata()
    }

    /// Enqueues a [`SimCommand`] if valid for the current worker state.
    pub fn send_command(&mut self, cmd: SimCommand) -> Result<(), SimCommandError> {
        self.controller.send_command(cmd)
    }

    /// Resets the worker in place or respawns a fresh runner when disconnected.
    pub fn reset_or_respawn<F>(&mut self, make_runner: F) -> bool
    where
        F: FnOnce() -> T,
    {
        self.controller.reset_or_respawn(make_runner)
    }

    /// Returns the most recently received [`StateSnapshot`], if any.
    pub fn latest_snapshot(&self) -> Option<&StateSnapshot> {
        self.controller.latest_snapshot()
    }

    /// Polls pending worker status and snapshot updates, returning a newly arrived snapshot.
    pub fn update(&mut self) -> Option<&StateSnapshot> {
        self.controller.update()
    }

    /// Borrows the underlying untyped [`SimulationController`].
    pub fn controller(&self) -> &SimulationController {
        &self.controller
    }

    /// Mutably borrows the underlying untyped [`SimulationController`].
    pub fn controller_mut(&mut self) -> &mut SimulationController {
        &mut self.controller
    }

    /// Unwraps this typed handle into its inner [`SimulationController`].
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
