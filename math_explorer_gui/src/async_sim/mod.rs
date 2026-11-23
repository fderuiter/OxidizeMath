#[allow(missing_docs)]
pub mod atlas;
#[cfg(all(test, feature = "ai", feature = "biology", feature = "physics"))]
mod fault_and_seed_tests;
#[cfg(all(
    test,
    feature = "ai",
    feature = "biology",
    feature = "physics",
    feature = "pure_math"
))]
mod runner_tests;
#[cfg(test)]
mod step_tests;
#[cfg(test)]
mod tests;
pub use atlas::{
    AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasPassMeta, AtlasRenderRequest, AtlasRunner,
    AtlasTeleportEvent, ParameterAtlasController, RegimeCartographer, ATLAS_SLICE_BUDGET,
    ATLAS_STRIDES, DEFAULT_ATLAS_SIDE, MAX_ATLAS_SIDE,
};
#[allow(missing_docs)]
pub mod ui;
#[allow(missing_docs)]
pub mod unified;
mod worker;

pub mod channel;
pub use channel::*;
pub mod state;
pub use state::*;

use std::sync::mpsc::{self, Sender};
pub use std::sync::Arc;

#[allow(missing_docs)]
pub trait SimulationRunner: Send + 'static {
    #[allow(missing_docs)]
    fn process_command(&mut self, cmd: SimCommand);
    #[allow(missing_docs)]
    fn step(&mut self);
    #[allow(missing_docs)]
    fn step_once(&mut self) {
        self.step();
    }
    #[allow(missing_docs)]
    fn get_snapshot(&self) -> StateSnapshot;
    #[allow(missing_docs)]
    fn get_steps_per_frame(&self) -> usize;
    #[allow(missing_docs)]
    fn is_complete(&self) -> bool {
        false
    }
    #[allow(missing_docs)]
    fn is_finite_state(&self) -> bool {
        true
    }
    #[allow(missing_docs)]
    fn diagnostic_report(&self) -> Option<crate::framework::recovery::DiagnosticReport> {
        None
    }
}

#[allow(missing_docs)]
pub struct SimulationController {
    cmd_tx: Sender<SimCommand>,
    state_rx: SimStateReceiver,
    latest_snapshot: Option<StateSnapshot>,
    pending_snapshot: bool,
    awaiting_pause_ack: bool,
    state: SimWorkerState,
    diagnostic_report: Option<crate::framework::recovery::DiagnosticReport>,
    #[cfg(target_arch = "wasm32")]
    handle: u32,
}

#[cfg(target_arch = "wasm32")]
// theory_verification!
impl Drop for SimulationController {
    fn drop(&mut self) {
        let _ = unregister_worker_context(self.handle);
        wake_worker(self.handle);
    }
}

impl SimulationController {
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(missing_docs)]
    pub fn new<T: SimulationRunner>(runner: T) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<SimCommand>();
        let (state_tx, state_rx) = sim_state_channel();
        let initial_snapshot = runner.get_snapshot();
        if !state_tx.mark_seeded() {
            let _ =
                state_tx.send_snapshot_and_status(runner.get_snapshot(), WorkerStatusEvent::Ready);
        }

        worker::spawn_native_worker(runner, cmd_rx, state_tx);

        Self {
            cmd_tx,
            state_rx,
            latest_snapshot: Some(initial_snapshot),
            pending_snapshot: false,
            awaiting_pause_ack: false,
            state: SimWorkerState::Ready,
            diagnostic_report: None,
        }
    }

    #[cfg(target_arch = "wasm32")]
    // theory_verification!
    pub fn new<T: SimulationRunner>(runner: T) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<SimCommand>();
        let (state_tx, state_rx) = sim_state_channel();
        let initial_snapshot = runner.get_snapshot();
        if !state_tx.mark_seeded() {
            let _ =
                state_tx.send_snapshot_and_status(runner.get_snapshot(), WorkerStatusEvent::Ready);
        }

        let handle = register_worker_context(WorkerContext {
            runner: Box::new(runner),
            cmd_rx,
            state_tx,
        });
        wasm_bindgen_futures::spawn_local(async move {
            run_simulation_worker(handle).await;
        });

        Self {
            cmd_tx,
            state_rx,
            latest_snapshot: Some(initial_snapshot),
            pending_snapshot: false,
            awaiting_pause_ack: false,
            state: SimWorkerState::Ready,
            diagnostic_report: None,
            handle,
        }
    }

    #[allow(missing_docs)]
    pub fn state(&self) -> SimWorkerState {
        self.state
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn set_state_for_test(&mut self, state: SimWorkerState) {
        if state.is_faulted() || state.is_terminated() {
            self.state_rx.clear();
            let _ = self.cmd_tx.send(SimCommand::Pause);
            self.pending_snapshot = false;
            self.awaiting_pause_ack = false;
        }
        self.state = state;
    }

    #[allow(missing_docs)]
    pub fn is_running(&self) -> bool {
        self.state.is_running()
    }

    /// Returns `true` when a `Pause` command was sent to a running worker and
    /// its `Ready` status acknowledgment has not yet been received.
    pub fn is_pause_pending(&self) -> bool {
        self.awaiting_pause_ack
    }

    fn record_status_pause_ack(&mut self, event: &WorkerStatusEvent) {
        if matches!(
            event,
            WorkerStatusEvent::Ready
                | WorkerStatusEvent::Faulted(_)
                | WorkerStatusEvent::Terminated
        ) {
            self.awaiting_pause_ack = false;
        }
    }

    fn drain_pre_command_updates(&mut self, is_reset: bool) {
        let batch = self.state_rx.drain();
        for event in batch.events {
            match event {
                SimControlEvent::Divergence(report) => {
                    self.diagnostic_report = Some(report);
                }
                SimControlEvent::Status(event) => {
                    self.record_status_pause_ack(&event);
                }
            }
        }
        if let Some(snapshot) = batch.latest_snapshot {
            if is_reset {
                self.pending_snapshot = false;
            } else if !self.state.is_faulted() && !self.state.is_terminated() {
                self.latest_snapshot = Some(snapshot);
                self.pending_snapshot = true;
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn wait_until_paused(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        for _ in 0..400 {
            if self.update().is_some() {
                self.pending_snapshot = true;
            }
            if !self.is_pause_pending() || self.state.is_faulted() || self.state.is_terminated() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        #[cfg(target_arch = "wasm32")]
        if self.update().is_some() {
            self.pending_snapshot = true;
        }
    }

    #[allow(missing_docs)]
    pub fn send_command(&mut self, cmd: SimCommand) -> Result<(), SimCommandError> {
        if !self.state.is_command_valid(&cmd) {
            return Err(SimCommandError::InvalidTransition {
                current_state: self.state,
                command: cmd.name(),
            });
        }

        let was_running = self.state.is_running();
        let is_pause = matches!(cmd, SimCommand::Pause);
        let is_reset = matches!(cmd, SimCommand::Reset);
        let cmd_target_state = match cmd {
            SimCommand::Start | SimCommand::TrainEpisodes(_) => Some(SimWorkerState::Running),
            SimCommand::Step(_) => Some(SimWorkerState::Stepping),
            SimCommand::Pause | SimCommand::Reset => Some(SimWorkerState::Ready),
            _ => None,
        };

        if cmd_target_state.is_some() {
            self.drain_pre_command_updates(is_reset);
        }

        if self.cmd_tx.send(cmd).is_err() {
            self.state = SimWorkerState::Faulted;
            self.awaiting_pause_ack = false;
            return Err(SimCommandError::ChannelDisconnected);
        }

        if is_pause && was_running {
            self.awaiting_pause_ack = true;
        } else if cmd_target_state.is_some() {
            self.awaiting_pause_ack = false;
        }

        if is_reset {
            self.diagnostic_report = None;
            self.pending_snapshot = false;
        }

        if let Some(target) = cmd_target_state {
            self.state = target;
        }

        #[cfg(target_arch = "wasm32")]
        self.wake_wasm_worker();

        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    // theory_verification!
    fn wake_wasm_worker(&self) {
        wake_worker(self.handle);
    }

    /// Resets the simulation worker in place if the command channel is alive,
    /// or spawns a fresh worker from `make_runner` when disconnected or terminated.
    /// Returns `true` if a new worker was spawned.
    #[allow(missing_docs)]
    pub fn reset_or_respawn<R, F>(&mut self, make_runner: F) -> bool
    where
        R: SimulationRunner,
        F: FnOnce() -> R,
    {
        match self.send_command(SimCommand::Reset) {
            Ok(()) => false,
            Err(
                SimCommandError::ChannelDisconnected | SimCommandError::InvalidTransition { .. },
            ) => {
                *self = Self::new(make_runner());
                true
            }
        }
    }

    #[allow(missing_docs)]
    pub fn latest_snapshot(&self) -> Option<&StateSnapshot> {
        self.latest_snapshot.as_ref()
    }

    #[allow(missing_docs)]
    pub fn diagnostic_report(&self) -> Option<&crate::framework::recovery::DiagnosticReport> {
        self.diagnostic_report.as_ref()
    }

    #[allow(missing_docs)]
    pub fn update(&mut self) -> Option<&StateSnapshot> {
        let mut received_new_snapshot = std::mem::take(&mut self.pending_snapshot);
        let mut stepping_in_flight = self.state.is_stepping();
        let batch = self.state_rx.drain();

        for control in batch.events {
            match control {
                SimControlEvent::Status(event) => {
                    self.record_status_pause_ack(&event);
                    let next = event.to_state();
                    if !self.state.is_terminated()
                        && (!self.state.is_faulted() || next.is_terminated())
                    {
                        if matches!(event, WorkerStatusEvent::Stepping) {
                            stepping_in_flight = true;
                        }
                        self.state = next;
                    }
                }
                SimControlEvent::Divergence(report) => {
                    self.diagnostic_report = Some(report);
                }
            }
        }

        if let Some(snapshot) = batch.latest_snapshot {
            if !self.state.is_faulted() && !self.state.is_terminated() {
                self.latest_snapshot = Some(snapshot);
                received_new_snapshot = true;
            }
        }

        if batch.disconnected {
            self.awaiting_pause_ack = false;
            if self.state != SimWorkerState::Terminated {
                self.state = SimWorkerState::Faulted;
            }
        }

        if stepping_in_flight {
            if self.state.is_faulted() {
                crate::accessibility::announce_status(
                    "Simulation step faulted due to numerical divergence",
                );
            } else if received_new_snapshot && self.state.is_ready() {
                crate::accessibility::announce_status("Simulation step completed");
            }
        }

        self.latest_snapshot
            .as_ref()
            .filter(|_| received_new_snapshot)
    }
}

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
// theory_verification!

#[cfg(any(target_arch = "wasm32", test))]
pub mod registry;
#[cfg(any(target_arch = "wasm32", test))]
pub use registry::*;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
#[cfg(target_arch = "wasm32")]
pub use wasm::*;
