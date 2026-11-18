#[cfg(test)]
mod runner_tests;
#[cfg(test)]
mod step_tests;
#[cfg(test)]
mod tests;
#[allow(missing_docs)]
pub mod unified;

pub mod state;
pub use state::*;

pub mod worker;
pub use worker::*;

use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
pub use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use std::thread;

const STATE_CHANNEL_CAPACITY: usize = 8;

#[allow(missing_docs)]
pub enum SimStateUpdate {
    #[allow(missing_docs)]
    Snapshot(StateSnapshot),
    #[allow(missing_docs)]
    Status(WorkerStatusEvent),
    #[allow(missing_docs)]
    Divergence(crate::framework::recovery::DiagnosticReport),
}

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

pub(crate) enum FrameTickOutcome {
    ContinueRunning,
    CompletedOrFaulted,
    Disconnected,
}

pub(crate) fn emit_worker_divergence<R: SimulationRunner + ?Sized>(
    runner: &R,
    state_tx: &SyncSender<SimStateUpdate>,
) {
    let report = runner.diagnostic_report().unwrap_or_else(|| {
        crate::framework::recovery::DiagnosticReport::new(
            0.0,
            0,
            "state",
            "Non-finite floating-point state encountered",
        )
    });
    let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Faulted(
        report.message.clone(),
    )));
    let _ = state_tx.send(SimStateUpdate::Divergence(report));
    let snapshot = runner.get_snapshot();
    let _ = state_tx.try_send(SimStateUpdate::Snapshot(snapshot));
}

pub(crate) fn run_worker_step_batch<R: SimulationRunner + ?Sized>(
    runner: &mut R,
    step_count: usize,
    state: &mut SimWorkerState,
    state_tx: &SyncSender<SimStateUpdate>,
) -> bool {
    *state = SimWorkerState::Stepping;
    let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Stepping));
    for _ in 0..step_count {
        runner.step_once();
        if !runner.is_finite_state() {
            *state = SimWorkerState::Faulted;
            emit_worker_divergence(runner, state_tx);
            return true;
        }
    }
    let snapshot = runner.get_snapshot();
    if state_tx.send(SimStateUpdate::Snapshot(snapshot)).is_err() {
        return false;
    }
    *state = SimWorkerState::Ready;
    let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
    true
}

pub(crate) fn dispatch_worker_command<R: SimulationRunner + ?Sized>(
    cmd: SimCommand,
    runner: &mut R,
    state: &mut SimWorkerState,
    steps_per_frame: &mut usize,
    state_tx: &SyncSender<SimStateUpdate>,
) -> bool {
    match cmd {
        SimCommand::Start => {
            *state = SimWorkerState::Running;
            let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Running));
            true
        }
        SimCommand::Pause => {
            *state = SimWorkerState::Ready;
            let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
            true
        }
        SimCommand::Step(count) => run_worker_step_batch(runner, count, state, state_tx),
        SimCommand::Reset => {
            *state = SimWorkerState::Ready;
            runner.process_command(SimCommand::Reset);
            let snapshot = runner.get_snapshot();
            let ok = state_tx.send(SimStateUpdate::Snapshot(snapshot)).is_ok();
            let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
            ok
        }
        SimCommand::SetSpeed(speed) => {
            *steps_per_frame = speed;
            runner.process_command(SimCommand::SetSpeed(speed));
            true
        }
        SimCommand::TrainEpisodes(_) => {
            *state = SimWorkerState::Running;
            runner.process_command(cmd);
            let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Running));
            true
        }
        other => {
            runner.process_command(other);
            true
        }
    }
}

pub(crate) fn drain_worker_commands<R: SimulationRunner + ?Sized>(
    first_cmd: SimCommand,
    cmd_rx: &Receiver<SimCommand>,
    runner: &mut R,
    state: &mut SimWorkerState,
    steps_per_frame: &mut usize,
    state_tx: &SyncSender<SimStateUpdate>,
) -> bool {
    let mut cmd = first_cmd;
    loop {
        if !dispatch_worker_command(cmd, runner, state, steps_per_frame, state_tx) {
            return false;
        }
        if state.is_faulted() {
            return true;
        }
        match cmd_rx.try_recv() {
            Ok(next) => cmd = next,
            Err(_) => return true,
        }
    }
}

pub(crate) fn advance_running_frame<R: SimulationRunner + ?Sized>(
    runner: &mut R,
    steps_per_frame: usize,
    state: &mut SimWorkerState,
    state_tx: &SyncSender<SimStateUpdate>,
) -> FrameTickOutcome {
    for _ in 0..steps_per_frame {
        runner.step();
        if !runner.is_finite_state() {
            *state = SimWorkerState::Faulted;
            emit_worker_divergence(runner, state_tx);
            return FrameTickOutcome::CompletedOrFaulted;
        }
        if runner.is_complete() {
            break;
        }
    }
    let is_done = runner.is_complete();
    let snapshot = runner.get_snapshot();
    if is_done {
        *state = SimWorkerState::Ready;
        if state_tx.send(SimStateUpdate::Snapshot(snapshot)).is_err() {
            return FrameTickOutcome::Disconnected;
        }
        let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
        FrameTickOutcome::CompletedOrFaulted
    } else if matches!(
        state_tx.try_send(SimStateUpdate::Snapshot(snapshot)),
        Err(mpsc::TrySendError::Disconnected(_))
    ) {
        FrameTickOutcome::Disconnected
    } else {
        FrameTickOutcome::ContinueRunning
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_native_worker<T: SimulationRunner>(
    mut runner: T,
    cmd_rx: Receiver<SimCommand>,
    state_tx: SyncSender<SimStateUpdate>,
) {
    thread::spawn(move || {
        let mut state = SimWorkerState::Ready;
        let mut steps_per_frame = runner.get_steps_per_frame();
        let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));

        loop {
            let cmd_opt = if state.is_running() {
                cmd_rx.try_recv().ok()
            } else {
                match cmd_rx.recv() {
                    Ok(c) => Some(c),
                    Err(_) => break,
                }
            };

            if let Some(first_cmd) = cmd_opt {
                if !drain_worker_commands(
                    first_cmd,
                    &cmd_rx,
                    &mut runner,
                    &mut state,
                    &mut steps_per_frame,
                    &state_tx,
                ) {
                    break;
                }
            }

            if state.is_running() {
                match advance_running_frame(&mut runner, steps_per_frame, &mut state, &state_tx) {
                    FrameTickOutcome::ContinueRunning => {
                        thread::sleep(std::time::Duration::from_millis(16));
                    }
                    FrameTickOutcome::CompletedOrFaulted => {}
                    FrameTickOutcome::Disconnected => break,
                }
            }
        }
    });
}

#[allow(missing_docs)]
pub struct SimulationController {
    cmd_tx: Sender<SimCommand>,
    state_rx: Receiver<SimStateUpdate>,
    latest_snapshot: Option<StateSnapshot>,
    state: SimWorkerState,
    diagnostic_report: Option<crate::framework::recovery::DiagnosticReport>,
    pending_step_announcement: bool,
    #[cfg(target_arch = "wasm32")]
    worker: Option<web_sys::Worker>,
}

impl SimulationController {
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(missing_docs)]
    pub fn new<T: SimulationRunner>(runner: T) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<SimCommand>();
        let (state_tx, state_rx) = mpsc::sync_channel::<SimStateUpdate>(STATE_CHANNEL_CAPACITY);
        spawn_native_worker(runner, cmd_rx, state_tx);

        Self {
            cmd_tx,
            state_rx,
            latest_snapshot: None,
            state: SimWorkerState::Ready,
            diagnostic_report: None,
            pending_step_announcement: false,
        }
    }

    #[cfg(target_arch = "wasm32")]
    // theory_verification!
    pub fn new<T: SimulationRunner>(runner: T) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<SimCommand>();
        let (state_tx, state_rx) = mpsc::sync_channel::<SimStateUpdate>(STATE_CHANNEL_CAPACITY);
        let handle = register_worker_context(WorkerContext {
            runner: Box::new(runner),
            cmd_rx,
            state_tx,
        });

        let opt_worker = spawn_web_worker(handle);
        if opt_worker.is_none() {
            wasm_bindgen_futures::spawn_local(async move {
                run_simulation_worker(handle).await;
            });
        }

        Self {
            cmd_tx,
            state_rx,
            latest_snapshot: None,
            state: SimWorkerState::Ready,
            diagnostic_report: None,
            pending_step_announcement: false,
            worker: opt_worker,
        }
    }

    #[allow(missing_docs)]
    pub fn state(&self) -> SimWorkerState {
        self.state
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn set_state_for_test(&mut self, state: SimWorkerState) {
        self.state = state;
    }

    #[allow(missing_docs)]
    pub fn is_running(&self) -> bool {
        self.state.is_running()
    }

    #[cfg(target_arch = "wasm32")]
    fn notify_wasm_worker(&self) {
        if let Some(worker) = &self.worker {
            let msg = js_sys::Object::new();
            let _ = js_sys::Reflect::set(
                &msg,
                &wasm_bindgen::JsValue::from_str("type"),
                &wasm_bindgen::JsValue::from_str("wake"),
            );
            let _ = worker.post_message(&msg);
        } else {
            wake_worker();
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

        if matches!(cmd, SimCommand::Reset) {
            self.diagnostic_report = None;
            self.pending_step_announcement = false;
        }
        let is_step = matches!(cmd, SimCommand::Step(_));
        let cmd_target_state = match cmd {
            SimCommand::Start | SimCommand::TrainEpisodes(_) => Some(SimWorkerState::Running),
            SimCommand::Step(_) => Some(SimWorkerState::Stepping),
            SimCommand::Pause | SimCommand::Reset => Some(SimWorkerState::Ready),
            _ => None,
        };

        if self.cmd_tx.send(cmd).is_err() {
            self.state = SimWorkerState::Faulted;
            return Err(SimCommandError::ChannelDisconnected);
        }

        if is_step {
            self.pending_step_announcement = true;
        }
        if let Some(target) = cmd_target_state {
            self.state = target;
        }

        #[cfg(target_arch = "wasm32")]
        self.notify_wasm_worker();

        Ok(())
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
        let mut disconnected = false;
        let mut received_new_snapshot = false;

        loop {
            match self.state_rx.try_recv() {
                Ok(SimStateUpdate::Snapshot(snapshot)) => {
                    self.latest_snapshot = Some(snapshot);
                    received_new_snapshot = true;
                }
                Ok(SimStateUpdate::Status(event)) => {
                    self.state = event.to_state();
                }
                Ok(SimStateUpdate::Divergence(report)) => {
                    self.diagnostic_report = Some(report);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }

        if disconnected && self.state != SimWorkerState::Terminated {
            self.state = SimWorkerState::Faulted;
        }

        if self.pending_step_announcement {
            if self.state.is_faulted() {
                self.pending_step_announcement = false;
                crate::accessibility::announce_status(
                    "Simulation step faulted due to numerical divergence",
                );
            } else if received_new_snapshot && self.state.is_ready() {
                self.pending_step_announcement = false;
                crate::accessibility::announce_status("Simulation step completed");
            }
        }

        if received_new_snapshot {
            self.latest_snapshot.as_ref()
        } else {
            None
        }
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
