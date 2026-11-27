#[allow(missing_docs)]
pub mod atlas;
#[cfg(all(test, any(feature = "biology", feature = "physics")))]
mod checkpoint_tests;
#[cfg(test)]
mod fault_and_seed_tests;
#[cfg(all(test, any(feature = "biology", feature = "physics")))]
mod mvp_time_travel_tests;
#[cfg(test)]
mod runner_tests;
#[cfg(test)]
mod scheduler_and_suspension_tests;
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

mod background_worker;
pub use background_worker::BackgroundWorker;
pub mod channel;
pub use channel::*;
pub mod checkpoint;
pub use checkpoint::*;
mod controller_suspend;
pub mod snapshot_history;
use snapshot_history::{initial_runner_timeline, initial_snapshot_history};
pub use snapshot_history::{
    SnapshotHistory, DEFAULT_MAX_SNAPSHOT_FRAMES, MAX_SNAPSHOT_HISTORY_BYTES,
};
pub mod state;
pub use state::*;
pub mod twin;
pub use twin::*;

use std::sync::mpsc::Sender;
pub use std::sync::Arc;

#[allow(missing_docs)]
pub trait SimulationRunner: Send + 'static {
    #[allow(missing_docs)]
    fn process_command(&mut self, cmd: SimCommand);
    #[allow(missing_docs)]
    fn step(&mut self);
    /// Advances one simulation step bounded by an optional cooperative `max_slice` wall-clock budget.
    fn step_sliced(&mut self, _max_slice: Option<std::time::Duration>) {
        self.step();
    }
    #[allow(missing_docs)]
    fn step_once(&mut self) {
        self.step();
    }
    /// Returns `true` when a lockstep counterfactual shadow twin (`Branch A`) is active.
    fn has_counterfactual_twin(&self) -> bool {
        false
    }
    /// Advances frozen baseline `Branch A` by one simulation tick.
    fn step_branch_a(&mut self) {}
    /// Advances active `Branch B` by one simulation tick and records divergence metrics.
    fn step_branch_b(&mut self) {
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
    /// Opt-in keyframe capture stride `K` (`Some(1)` for compact ODEs, `Some(16)` for 2D grids).
    fn checkpoint_stride(&self) -> Option<usize> {
        None
    }
    /// Current deterministic step index for opted-in time-travel runners.
    fn current_step(&self) -> usize {
        0
    }
    /// Captures a self-contained [`TimeTravelCheckpoint`] at `current_step()`.
    fn capture_checkpoint(&self) -> Option<TimeTravelCheckpoint> {
        None
    }
    /// Captures active model parameters for mid-run parameter epoch tracking.
    fn capture_active_params(&self) -> Option<CheckpointParams> {
        None
    }
    /// Fast allocation-free check returning `true` when active parameters match `last`.
    fn active_params_match(&self, _last: &CheckpointParams) -> bool {
        false
    }
    /// Restores dynamical fields, active parameters, step index, and cloned `OxidizeRng`.
    fn restore_checkpoint(&mut self, _checkpoint: &TimeTravelCheckpoint) {}
    /// Reconstructs bounded trajectory histories from retained keyframes `..=kf_idx`.
    fn rebuild_history(&mut self, _front: &[TimeTravelCheckpoint], _back: &[TimeTravelCheckpoint]) {
    }
    /// Advances one zero-allocation sub-step during deterministic checkpoint replay.
    fn replay_step(&mut self, _params: &CheckpointParams) {
        self.step_once();
    }
    /// Number of sub-steps executed per running frame tick when checkpointing is active.
    fn running_substeps_per_tick(&self) -> usize {
        1
    }
    /// Advances a single running sub-step while keeping runner execution state in `Running`.
    fn step_running_substep(&mut self) {
        self.step();
    }
}

pub(super) fn seed_controller_channel<T: SimulationRunner + ?Sized>(
    runner: &T,
    state_tx: &SimStateSender,
) -> (StateSnapshot, Option<TimelineMetadata>) {
    let initial_snapshot = runner.get_snapshot();
    let initial_timeline = initial_runner_timeline(runner);
    state_tx.set_timeline(initial_timeline);
    if !state_tx.mark_seeded() {
        let _ = state_tx.send_snapshot_and_status(runner.get_snapshot(), WorkerStatusEvent::Ready);
    }
    (initial_snapshot, initial_timeline)
}

#[allow(missing_docs)]
pub struct SimulationController {
    cmd_tx: Sender<SimCommand>,
    state_rx: SimStateReceiver,
    latest_snapshot: Option<StateSnapshot>,
    timeline_metadata: Option<TimelineMetadata>,
    snapshot_history: Option<SnapshotHistory>,
    pending_snapshot: bool,
    awaiting_pause_ack: bool,
    suspended: bool,
    state: SimWorkerState,
    diagnostic_report: Option<crate::framework::recovery::DiagnosticReport>,
    #[cfg(not(target_arch = "wasm32"))]
    native_suspension: controller_suspend::NativeSuspensionDropGuard,
    #[cfg(target_arch = "wasm32")]
    handle: u32,
}

#[cfg(target_arch = "wasm32")]
impl Drop for SimulationController {
    fn drop(&mut self) {
        let _ = unregister_worker_context(self.handle);
        wake_worker(self.handle);
        clear_wasm_worker_state(self.handle);
    }
}

impl SimulationController {
    #[allow(missing_docs)]
    pub fn new<T: SimulationRunner>(runner: T) -> Self {
        Self::new_with_suspended(runner, false)
    }

    #[allow(missing_docs)]
    pub fn state(&self) -> SimWorkerState {
        self.state
    }

    /// Latest scrubbable timeline metadata if the underlying runner opted into time travel.
    pub fn timeline_metadata(&self) -> Option<TimelineMetadata> {
        self.timeline_metadata
    }

    /// Opt-in UI-side [`SnapshotHistory`] ring buffer (`<= 2 MiB`) if active on this controller.
    pub fn snapshot_history(&self) -> Option<&SnapshotHistory> {
        self.snapshot_history.as_ref()
    }

    /// Enables or reconfigures the bounded UI-side [`SnapshotHistory`] ring (`<= 2 MiB`).
    pub fn enable_snapshot_history(&mut self, max_frames: usize, max_bytes: usize) {
        self.snapshot_history = Some(SnapshotHistory::seeded(
            max_frames,
            max_bytes,
            self.timeline_metadata,
            self.latest_snapshot.as_ref(),
        ));
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

    fn record_incoming_snapshot(&mut self, snapshot: StateSnapshot) {
        if let (Some(hist), Some(meta)) = (self.snapshot_history.as_mut(), self.timeline_metadata) {
            hist.record(meta.current_step, &snapshot);
        }
        self.latest_snapshot = Some(snapshot);
    }

    fn drain_pre_command_updates(&mut self, is_reset: bool) {
        let batch = self.state_rx.drain();
        for event in batch.events {
            match event {
                SimControlEvent::Divergence(report) => self.diagnostic_report = Some(report),
                SimControlEvent::Status(event) => self.record_status_pause_ack(&event),
            }
        }
        if !is_reset && !self.state.is_terminated() {
            self.timeline_metadata = batch.timeline_metadata;
        }
        if let Some(snapshot) = batch.latest_snapshot {
            if is_reset {
                self.pending_snapshot = false;
            } else if !self.state.is_faulted() && !self.state.is_terminated() {
                self.record_incoming_snapshot(snapshot);
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

    fn apply_optimistic_timeline(&mut self, cmd: &SimCommand) {
        if matches!(cmd, SimCommand::Reset) {
            if let Some(hist) = self.snapshot_history.as_mut() {
                hist.clear();
            }
        }
        let Some(meta) = &mut self.timeline_metadata else {
            return;
        };
        match *cmd {
            SimCommand::Reset => {
                meta.current_step = 0;
                meta.min_step = 0;
                meta.max_step = 0;
            }
            SimCommand::SeekStep(target) => {
                meta.current_step = target.clamp(meta.min_step, meta.max_step);
            }
            SimCommand::Start | SimCommand::Step(_) => {
                if let Some(hist) = self.snapshot_history.as_mut() {
                    hist.truncate_after(meta.current_step);
                }
                meta.max_step = meta.current_step;
            }
            _ => {}
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
        let pauses_running = matches!(cmd, SimCommand::Pause | SimCommand::SeekStep(_));
        let is_reset = matches!(cmd, SimCommand::Reset);
        let cmd_target_state = match cmd {
            SimCommand::Start | SimCommand::TrainEpisodes(_) => Some(SimWorkerState::Running),
            SimCommand::Step(_) | SimCommand::StepBack(_) => Some(SimWorkerState::Stepping),
            SimCommand::Pause | SimCommand::Reset | SimCommand::SeekStep(_) => {
                Some(SimWorkerState::Ready)
            }
            _ => None,
        };

        if cmd_target_state.is_some() {
            self.drain_pre_command_updates(is_reset);
        }
        self.apply_optimistic_timeline(&cmd);

        if self.cmd_tx.send(cmd).is_err() {
            self.state = SimWorkerState::Faulted;
            self.awaiting_pause_ack = false;
            return Err(SimCommandError::ChannelDisconnected);
        }

        self.awaiting_pause_ack = pauses_running && was_running && !self.suspended;
        if is_reset {
            self.diagnostic_report = None;
            self.pending_snapshot = false;
        }
        if let Some(target) = cmd_target_state {
            self.state = target;
        }

        #[cfg(not(target_arch = "wasm32"))]
        if self.suspended {
            self.native_suspension.0.notify();
        }
        #[cfg(target_arch = "wasm32")]
        self.wake_wasm_worker();

        Ok(())
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
        let was_suspended = self.suspended;
        let prev_limits = self.snapshot_history.as_ref().map(SnapshotHistory::limits);
        match self.send_command(SimCommand::Reset) {
            Ok(()) => false,
            Err(
                SimCommandError::ChannelDisconnected | SimCommandError::InvalidTransition { .. },
            ) => {
                let mut next = Self::new_with_suspended(make_runner(), was_suspended);
                if let Some((max_frames, max_bytes)) = prev_limits {
                    next.enable_snapshot_history(max_frames, max_bytes);
                }
                *self = next;
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

        if !self.state.is_terminated() {
            self.timeline_metadata = batch.timeline_metadata;
        }
        if let Some(snapshot) = batch.latest_snapshot {
            if !self.state.is_faulted() && !self.state.is_terminated() {
                self.record_incoming_snapshot(snapshot);
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

#[cfg(any(target_arch = "wasm32", test))]
mod registry;
/// Dedicated Web Worker (`SharedArrayBuffer` + transferable `postMessage`) volumetric runtime bridge.
pub mod web_worker;
/// Thin `web-sys` / `wasm-bindgen` WebGPU `@compute` pipeline and double-buffered `MAP_READ` staging dispatcher.
pub mod webgpu;
#[cfg(target_arch = "wasm32")]
use registry::{
    register_worker_context, take_worker_context, unregister_worker_context, WorkerContext,
};

#[cfg(target_arch = "wasm32")]
mod wasm;
#[cfg(target_arch = "wasm32")]
mod wasm_scheduler;
#[cfg(target_arch = "wasm32")]
use wasm::run_simulation_worker;
#[cfg(target_arch = "wasm32")]
use wasm_scheduler::{
    clear_worker_entries as clear_wasm_worker_state, register_wasm_worker,
    sync_controller_state as sync_wasm_worker_state, wake_worker,
};
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;
