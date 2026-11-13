#[cfg(test)]
mod tests;
#[allow(missing_docs)]
pub mod unified;
#[cfg(any(target_arch = "wasm32", test))]
use std::sync::mpsc::SyncSender;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

#[cfg(any(target_arch = "wasm32", test))]
use std::collections::HashMap;
#[cfg(any(target_arch = "wasm32", test))]
use std::sync::atomic::{AtomicU32, Ordering};
#[cfg(any(target_arch = "wasm32", test))]
use std::sync::Mutex;

#[cfg(not(target_arch = "wasm32"))]
use std::thread;

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
    TrainEpisodes(usize),
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
    Status { running: bool },
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
}

#[allow(missing_docs)]
pub struct SimulationController {
    cmd_tx: Sender<SimCommand>,
    state_rx: Receiver<SimStateUpdate>,
    latest_snapshot: Option<StateSnapshot>,
    #[allow(missing_docs)]
    pub running: bool,
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
        let (state_tx, state_rx) = mpsc::sync_channel::<SimStateUpdate>(4);

        thread::spawn(move || {
            let mut running = false;
            let mut steps_per_frame = runner.get_steps_per_frame();

            loop {
                let mut cmd_opt = None;
                if running {
                    if let Ok(c) = cmd_rx.try_recv() {
                        cmd_opt = Some(c);
                    }
                } else {
                    if let Ok(c) = cmd_rx.recv() {
                        cmd_opt = Some(c);
                    } else {
                        break;
                    }
                }

                if let Some(mut cmd) = cmd_opt {
                    loop {
                        match cmd {
                            SimCommand::Start => {
                                running = true;
                                let _ = state_tx.send(SimStateUpdate::Status { running: true });
                            }
                            SimCommand::Pause => {
                                running = false;
                                let _ = state_tx.send(SimStateUpdate::Status { running: false });
                            }
                            SimCommand::SetSpeed(speed) => {
                                steps_per_frame = speed;
                                runner.process_command(SimCommand::SetSpeed(speed));
                            }
                            SimCommand::TrainEpisodes(_) => {
                                running = true;
                                runner.process_command(cmd);
                                let _ = state_tx.send(SimStateUpdate::Status { running: true });
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

                if running {
                    for _ in 0..steps_per_frame {
                        runner.step();
                        if runner.is_complete() {
                            break;
                        }
                    }
                    let is_done = runner.is_complete();
                    let snapshot = runner.get_snapshot();
                    if is_done {
                        running = false;
                        let _ = state_tx.send(SimStateUpdate::Snapshot(snapshot));
                        let _ = state_tx.send(SimStateUpdate::Status { running: false });
                    } else {
                        let _ = state_tx.try_send(SimStateUpdate::Snapshot(snapshot));
                        thread::sleep(std::time::Duration::from_millis(16));
                    }
                }
            }
        });

        Self {
            cmd_tx,
            state_rx,
            latest_snapshot: None,
            running: false,
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
            running: false,
            worker: opt_worker,
            handle: Some(handle),
        }
    }

    #[allow(missing_docs)]
    pub fn send_command(&self, cmd: SimCommand) {
        let _ = self.cmd_tx.send(cmd);
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
    }

    #[allow(missing_docs)]
    pub fn latest_snapshot(&self) -> Option<&StateSnapshot> {
        self.latest_snapshot.as_ref()
    }

    #[allow(missing_docs)]
    pub fn update(&mut self) -> Option<&StateSnapshot> {
        while let Ok(update) = self.state_rx.try_recv() {
            match update {
                SimStateUpdate::Snapshot(snapshot) => {
                    self.latest_snapshot = Some(snapshot);
                }
                SimStateUpdate::Status { running } => {
                    self.running = running;
                }
            }
        }
        self.latest_snapshot.as_ref()
    }
}
// theory_verification!

#[cfg(any(target_arch = "wasm32", test))]
#[allow(missing_docs)]
pub struct WorkerContext {
    #[allow(missing_docs)]
    pub runner: Box<dyn SimulationRunner>,
    #[allow(missing_docs)]
    pub cmd_rx: Receiver<SimCommand>,
    #[allow(missing_docs)]
    pub state_tx: SyncSender<SimStateUpdate>,
}

#[cfg(any(target_arch = "wasm32", test))]
static NEXT_HANDLE: AtomicU32 = AtomicU32::new(1);
#[cfg(any(target_arch = "wasm32", test))]
static WORKER_REGISTRY: Mutex<Option<HashMap<u32, WorkerContext>>> = Mutex::new(None);

#[cfg(any(target_arch = "wasm32", test))]
#[allow(missing_docs)]
pub fn register_worker_context(ctx: WorkerContext) -> u32 {
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    let mut guard = WORKER_REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    guard.get_or_insert_with(HashMap::new).insert(handle, ctx);
    handle
}

#[cfg(any(target_arch = "wasm32", test))]
#[allow(missing_docs)]
pub fn take_worker_context(handle: u32) -> Option<WorkerContext> {
    let mut guard = WORKER_REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_mut()?.remove(&handle)
}

#[cfg(any(target_arch = "wasm32", test))]
#[allow(missing_docs)]
pub fn unregister_worker_context(handle: u32) -> Option<WorkerContext> {
    take_worker_context(handle)
}

#[cfg(target_arch = "wasm32")]
pub mod wasm;
#[cfg(target_arch = "wasm32")]
pub use wasm::{run_simulation_worker, wake_worker};
