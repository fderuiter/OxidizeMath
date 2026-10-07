#[allow(missing_docs)]
pub mod unified;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use std::thread;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

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
}

impl Drop for SimulationController {
    fn drop(&mut self) {
        #[cfg(target_arch = "wasm32")]
        if let Some(worker) = &self.worker {
            worker.terminate();
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
            running: false,
        }
    }

    #[cfg(target_arch = "wasm32")]
    // theory_verification!
    pub fn new<T: SimulationRunner>(runner: T) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<SimCommand>();
        let (state_tx, state_rx) = mpsc::channel::<SimStateUpdate>();

        let ctx = Box::new(WorkerContext {
            runner: Box::new(runner),
            cmd_rx,
            state_tx,
        });

        let ptr = Box::into_raw(ctx) as u32;

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
                &wasm_bindgen::JsValue::from_str("ptr"),
                &wasm_bindgen::JsValue::from_f64(ptr as f64),
            );

            let _ = worker.post_message(&msg);
            opt_worker = Some(worker);
        } else {
            // Fallback if worker.js doesn't exist
            wasm_bindgen_futures::spawn_local(async move {
                run_simulation_worker(ptr).await;
            });
        }

        Self {
            cmd_tx,
            state_rx,
            latest_snapshot: None,
            running: false,
            worker: opt_worker,
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

#[cfg(target_arch = "wasm32")]
pub struct WorkerContext {
    runner: Box<dyn SimulationRunner>,
    cmd_rx: Receiver<SimCommand>,
    state_tx: Sender<SimStateUpdate>,
}

#[cfg(target_arch = "wasm32")]
pub fn wake_worker() {
    let global = js_sys::global();
    if let Ok(wake_fn) = js_sys::Reflect::get(
        &global,
        &wasm_bindgen::JsValue::from_str("__sim_worker_wake"),
    ) {
        if wake_fn.is_function() {
            let func = wake_fn.unchecked_into::<js_sys::Function>();
            let _ = func.call0(&wasm_bindgen::JsValue::UNDEFINED);
        }
    }
}

#[cfg(target_arch = "wasm32")]
async fn yield_to_event_loop() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let global = js_sys::global();
        let timeout_scheduled = if let Ok(worker_scope) = global
            .clone()
            .dyn_into::<web_sys::DedicatedWorkerGlobalScope>(
        ) {
            worker_scope
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0)
                .is_ok()
        } else if let Ok(window) = global.dyn_into::<web_sys::Window>() {
            window
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0)
                .is_ok()
        } else {
            false
        };
        if !timeout_scheduled {
            let _ = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

#[cfg(target_arch = "wasm32")]
async fn suspend_worker() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let global = js_sys::global();
        let _ = js_sys::Reflect::set(
            &global,
            &wasm_bindgen::JsValue::from_str("__sim_worker_wake"),
            &resolve,
        );
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    // Remove the wake function after waking up
    let global = js_sys::global();
    let _ = js_sys::Reflect::delete_property(
        &global,
        &wasm_bindgen::JsValue::from_str("__sim_worker_wake"),
    );
}

#[cfg(target_arch = "wasm32")]
pub fn setup_wake_listener() {
    let global = js_sys::global();
    if let Ok(worker_scope) = global.dyn_into::<web_sys::DedicatedWorkerGlobalScope>() {
        let cb = wasm_bindgen::closure::Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
            if let Ok(data) = e.data().dyn_into::<js_sys::Object>() {
                if let Ok(type_val) =
                    js_sys::Reflect::get(&data, &wasm_bindgen::JsValue::from_str("type"))
                {
                    if type_val.as_string().as_deref() == Some("wake") {
                        wake_worker();
                    }
                }
            }
        })
            as Box<dyn FnMut(web_sys::MessageEvent)>);
        let _ =
            worker_scope.add_event_listener_with_callback("message", cb.as_ref().unchecked_ref());
        cb.forget(); // Leak to keep it alive
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_simulation_worker(ptr: u32) {
    setup_wake_listener();
    let ctx = unsafe { Box::from_raw(ptr as *mut WorkerContext) };
    let mut runner = ctx.runner;
    let cmd_rx = ctx.cmd_rx;
    let state_tx = ctx.state_tx;

    let mut running = false;
    let mut steps_per_frame = runner.get_steps_per_frame();

    loop {
        let mut cmd_opt = None;
        if running {
            if let Ok(c) = cmd_rx.try_recv() {
                cmd_opt = Some(c);
            }
        } else {
            if let Ok(c) = cmd_rx.try_recv() {
                cmd_opt = Some(c);
            } else {
                suspend_worker().await;
                // After waking up, try to receive the command again
                if let Ok(c) = cmd_rx.try_recv() {
                    cmd_opt = Some(c);
                }
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
            }
            let snapshot = runner.get_snapshot();
            if state_tx.send(SimStateUpdate::Snapshot(snapshot)).is_err() {
                break;
            }
            yield_to_event_loop().await;
        }
    }
}
