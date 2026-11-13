#[cfg(test)]
mod tests;
#[allow(missing_docs)]
pub mod unified;

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use std::thread;

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SimWorkerState {
    #[default]
    Uninitialized,
    Ready,
    Running,
    Stepping,
    Faulted,
    Terminated,
}

impl SimWorkerState {
    #[allow(missing_docs)]
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running)
    }

    #[allow(missing_docs)]
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    #[allow(missing_docs)]
    pub fn is_faulted(&self) -> bool {
        matches!(self, Self::Faulted)
    }

    #[allow(missing_docs)]
    pub fn is_terminated(&self) -> bool {
        matches!(self, Self::Terminated)
    }

    #[allow(missing_docs)]
    pub fn is_command_valid(&self, cmd: &SimCommand) -> bool {
        match self {
            Self::Ready => !matches!(cmd, SimCommand::Pause),
            Self::Running => !matches!(cmd, SimCommand::Start),
            Self::Stepping => !matches!(cmd, SimCommand::Start | SimCommand::Pause),
            Self::Uninitialized | Self::Faulted | Self::Terminated => false,
        }
    }
}

impl std::fmt::Display for SimWorkerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uninitialized => write!(f, "Uninitialized"),
            Self::Ready => write!(f, "Ready"),
            Self::Running => write!(f, "Running"),
            Self::Stepping => write!(f, "Stepping"),
            Self::Faulted => write!(f, "Faulted"),
            Self::Terminated => write!(f, "Terminated"),
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerStatusEvent {
    Ready,
    Running,
    Stepping,
    Paused,
    Faulted(String),
    Terminated,
}

impl WorkerStatusEvent {
    #[allow(missing_docs)]
    pub fn to_state(&self) -> SimWorkerState {
        match self {
            Self::Ready | Self::Paused => SimWorkerState::Ready,
            Self::Running => SimWorkerState::Running,
            Self::Stepping => SimWorkerState::Stepping,
            Self::Faulted(_) => SimWorkerState::Faulted,
            Self::Terminated => SimWorkerState::Terminated,
        }
    }
}

impl From<&WorkerStatusEvent> for SimWorkerState {
    fn from(event: &WorkerStatusEvent) -> Self {
        event.to_state()
    }
}

impl From<WorkerStatusEvent> for SimWorkerState {
    fn from(event: WorkerStatusEvent) -> Self {
        event.to_state()
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimCommandError {
    InvalidTransition {
        current_state: SimWorkerState,
        command: &'static str,
    },
    ChannelDisconnected,
}

impl std::fmt::Display for SimCommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTransition {
                current_state,
                command,
            } => {
                write!(
                    f,
                    "Invalid simulation command '{command}' in state {current_state}"
                )
            }
            Self::ChannelDisconnected => {
                write!(f, "Simulation worker channel disconnected")
            }
        }
    }
}

impl std::error::Error for SimCommandError {}

#[allow(missing_docs)]
pub enum SimCommand {
    #[allow(missing_docs)]
    Start,
    #[allow(missing_docs)]
    Pause,
    #[allow(missing_docs)]
    Reset,
    #[allow(missing_docs)]
    SetSpeed(usize),
    #[allow(missing_docs)]
    ApplyBrush {
        #[allow(missing_docs)]
        cx: i32,
        #[allow(missing_docs)]
        cy: i32,
        #[allow(missing_docs)]
        r: i32,
        #[allow(missing_docs)]
        is_obstacle: bool,
    },
    #[allow(missing_docs)]
    ClearObstacles,
    #[allow(missing_docs)]
    Custom(String),
}

impl SimCommand {
    #[allow(missing_docs)]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Start => "Start",
            Self::Pause => "Pause",
            Self::Reset => "Reset",
            Self::SetSpeed(_) => "SetSpeed",
            Self::ApplyBrush { .. } => "ApplyBrush",
            Self::ClearObstacles => "ClearObstacles",
            Self::Custom(_) => "Custom",
        }
    }
}

#[allow(missing_docs)]
pub struct StateSnapshot {
    #[allow(missing_docs)]
    pub width: usize,
    #[allow(missing_docs)]
    pub height: usize,
    #[allow(missing_docs)]
    pub pixels: Arc<std::sync::RwLock<Vec<eframe::egui::Color32>>>,
    #[allow(missing_docs)]
    pub custom_data: Vec<f64>,
    #[allow(missing_docs)]
    pub structured_data: Option<Box<dyn std::any::Any + Send>>,
}

#[allow(missing_docs)]
pub enum SimStateUpdate {
    #[allow(missing_docs)]
    Snapshot(StateSnapshot),
    #[allow(missing_docs)]
    Status(WorkerStatusEvent),
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
}

#[allow(missing_docs)]
pub struct SimulationController {
    cmd_tx: Sender<SimCommand>,
    state_rx: Receiver<SimStateUpdate>,
    latest_snapshot: Option<StateSnapshot>,
    state: SimWorkerState,
    #[cfg(target_arch = "wasm32")]
    worker: Option<web_sys::Worker>,
    #[cfg(target_arch = "wasm32")]
    handle: Option<u32>,
}

impl Drop for SimulationController {
    fn drop(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(worker) = &self.worker {
                worker.terminate();
            }
            if let Some(handle) = self.handle.take() {
                unregister_worker_context(handle);
            }
        }
    }
}

impl SimulationController {
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    #[allow(missing_docs)]
    pub fn new<T: SimulationRunner>(mut runner: T) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<SimCommand>();
        let (state_tx, state_rx) = mpsc::channel::<SimStateUpdate>();

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
                    for _ in 0..steps_per_frame {
                        runner.step();
                    }
                    let snapshot = runner.get_snapshot();
                    if state_tx.send(SimStateUpdate::Snapshot(snapshot)).is_err() {
                        break;
                    }
                }
            }
        });

        Self {
            cmd_tx,
            state_rx,
            latest_snapshot: None,
            state: SimWorkerState::Ready,
        }
    }

    #[cfg(target_arch = "wasm32")]
    // theory_verification!
    pub fn new<T: SimulationRunner>(runner: T) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<SimCommand>();
        let (state_tx, state_rx) = mpsc::channel::<SimStateUpdate>();

        let ctx = WorkerContext {
            runner: Box::new(runner),
            cmd_rx,
            state_tx,
        };

        let handle = register_worker_context(ctx);

        let mut opt_worker = None;
        // Try to initialize a Web Worker
        if let Ok(worker) = web_sys::Worker::new("worker.js") {
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

            let _ = worker.post_message(&msg);
            opt_worker = Some(worker);
        } else {
            // Fallback if worker.js doesn't exist
            wasm_bindgen_futures::spawn_local(async move {
                run_simulation_worker(handle).await;
            });
        }

        Self {
            cmd_tx,
            state_rx,
            latest_snapshot: None,
            state: SimWorkerState::Ready,
            worker: opt_worker,
            handle: Some(handle),
        }
    }

    #[allow(missing_docs)]
    pub fn state(&self) -> SimWorkerState {
        self.state
    }

    #[allow(missing_docs)]
    pub fn send_command(&mut self, cmd: SimCommand) -> Result<(), SimCommandError> {
        if !self.state.is_command_valid(&cmd) {
            return Err(SimCommandError::InvalidTransition {
                current_state: self.state,
                command: cmd.name(),
            });
        }

        let cmd_target_state = match cmd {
            SimCommand::Start => Some(SimWorkerState::Running),
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
// theory_verification!

#[cfg(any(target_arch = "wasm32", test))]
pub mod registry;
#[cfg(any(target_arch = "wasm32", test))]
pub use registry::*;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
#[cfg(target_arch = "wasm32")]
pub use wasm::*;
