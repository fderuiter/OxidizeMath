#[cfg(test)]
mod runner_tests;
#[cfg(test)]
mod tests;
#[allow(missing_docs)]
pub mod unified;

pub mod state;
pub use state::*;

use std::sync::mpsc::{self, Receiver, Sender};
pub use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use std::thread;

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
    state_rx: Receiver<SimStateUpdate>,
    latest_snapshot: Option<StateSnapshot>,
    state: SimWorkerState,
    diagnostic_report: Option<crate::framework::recovery::DiagnosticReport>,
    #[cfg(target_arch = "wasm32")]
    worker: Option<web_sys::Worker>,
}

impl SimulationController {
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(missing_docs, clippy::too_many_lines)]
    pub fn new<T: SimulationRunner>(mut runner: T) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<SimCommand>();
        let (state_tx, state_rx) = mpsc::sync_channel::<SimStateUpdate>(4);

        thread::spawn(move || {
            let mut state = SimWorkerState::Ready;
            let mut steps_per_frame = runner.get_steps_per_frame();
            let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));

            loop {
                let mut cmd_opt = None;
                if state.is_running() {
                    if let Ok(c) = cmd_rx.try_recv() {
                        cmd_opt = Some(c);
                    }
                } else if let Ok(c) = cmd_rx.recv() {
                    cmd_opt = Some(c);
                } else {
                    break;
                }

                if let Some(mut cmd) = cmd_opt {
                    loop {
                        match cmd {
                            SimCommand::Start => {
                                state = SimWorkerState::Running;
                                let _ = state_tx
                                    .send(SimStateUpdate::Status(WorkerStatusEvent::Running));
                            }
                            SimCommand::Pause => {
                                state = SimWorkerState::Ready;
                                let _ =
                                    state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
                            }
                            SimCommand::Reset => {
                                state = SimWorkerState::Ready;
                                runner.process_command(SimCommand::Reset);
                                let snapshot = runner.get_snapshot();
                                let _ = state_tx.send(SimStateUpdate::Snapshot(snapshot));
                                let _ =
                                    state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
                            }
                            SimCommand::SetSpeed(speed) => {
                                steps_per_frame = speed;
                                runner.process_command(SimCommand::SetSpeed(speed));
                            }
                            SimCommand::TrainEpisodes(_) => {
                                state = SimWorkerState::Running;
                                runner.process_command(cmd);
                                let _ = state_tx
                                    .send(SimStateUpdate::Status(WorkerStatusEvent::Running));
                            }
                            _ => {
                                runner.process_command(cmd);
                            }
                        }
                        if let Ok(c) = cmd_rx.try_recv() {
                            cmd = c;
                        } else {
                            break;
                        }
                    }
                }

                if state.is_running() {
                    let mut diverged = false;
                    for _ in 0..steps_per_frame {
                        runner.step();
                        if !runner.is_finite_state() {
                            diverged = true;
                            break;
                        }
                        if runner.is_complete() {
                            break;
                        }
                    }
                    if diverged {
                        state = SimWorkerState::Faulted;
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
                        continue;
                    }
                    let is_done = runner.is_complete();
                    let snapshot = runner.get_snapshot();
                    if is_done {
                        state = SimWorkerState::Ready;
                        if state_tx.send(SimStateUpdate::Snapshot(snapshot)).is_err() {
                            break;
                        }
                        let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
                    } else {
                        if let Err(mpsc::TrySendError::Disconnected(_)) =
                            state_tx.try_send(SimStateUpdate::Snapshot(snapshot))
                        {
                            break;
                        }
                        thread::sleep(std::time::Duration::from_millis(16));
                    }
                }
            }
        });

        Self {
            cmd_tx,
            state_rx,
            latest_snapshot: None,
            state: SimWorkerState::Ready,
            diagnostic_report: None,
        }
    }

    #[cfg(target_arch = "wasm32")]
    // theory_verification!
    pub fn new<T: SimulationRunner>(runner: T) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<SimCommand>();
        let (state_tx, state_rx) = mpsc::sync_channel::<SimStateUpdate>(4);

        let ctx = WorkerContext {
            runner: Box::new(runner),
            cmd_rx,
            state_tx,
        };

        let handle = register_worker_context(ctx);

        let mut opt_worker = None;
        // Try to initialize a Web Worker
        let worker_opts = web_sys::WorkerOptions::new();
        worker_opts.set_type(web_sys::WorkerType::Module);
        if let Ok(worker) = web_sys::Worker::new_with_options("worker.js", &worker_opts) {
            let msg = js_sys::Object::new();
            let _ = js_sys::Reflect::set(
                &msg,
                &wasm_bindgen::JsValue::from_str("type"),
                &wasm_bindgen::JsValue::from_str("init"),
            );
            let _ = js_sys::Reflect::set(
                &msg,
                &wasm_bindgen::JsValue::from_str("module"),
                &wasm_bindgen::module(),
            );
            let _ = js_sys::Reflect::set(
                &msg,
                &wasm_bindgen::JsValue::from_str("memory"),
                &wasm_bindgen::memory(),
            );
            let _ = js_sys::Reflect::set(
                &msg,
                &wasm_bindgen::JsValue::from_str("handle"),
                &wasm_bindgen::JsValue::from_f64(handle as f64),
            );
            if worker.post_message(&msg).is_ok() {
                opt_worker = Some(worker);
            }
        }

        // If Web Worker failed to initialize or isn't supported, run in-thread via microtask
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
        }

        let cmd_target_state = match cmd {
            SimCommand::Start | SimCommand::TrainEpisodes(_) => Some(SimWorkerState::Running),
            SimCommand::Pause | SimCommand::Reset => Some(SimWorkerState::Ready),
            _ => None,
        };

        if self.cmd_tx.send(cmd).is_err() {
            self.state = SimWorkerState::Faulted;
            return Err(SimCommandError::ChannelDisconnected);
        }

        if let Some(target) = cmd_target_state {
            self.state = target;
        }

        #[cfg(target_arch = "wasm32")]
        {
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
                Ok(update) => match update {
                    SimStateUpdate::Snapshot(snapshot) => {
                        self.latest_snapshot = Some(snapshot);
                        received_new_snapshot = true;
                    }
                    SimStateUpdate::Status(event) => {
                        self.state = event.to_state();
                    }
                    SimStateUpdate::Divergence(report) => {
                        self.diagnostic_report = Some(report);
                    }
                },
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

        if received_new_snapshot {
            self.latest_snapshot.as_ref()
        } else {
            None
        }
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
