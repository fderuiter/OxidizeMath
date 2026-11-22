//! Cooperative WebAssembly simulation scheduler and handle-keyed wake registry.

// theory_verification!

use super::worker::{advance_running_frame, drain_worker_commands, FrameTickOutcome};
use super::{
    take_worker_context, SimCommand, SimWorkerState, WorkerStatusEvent, WASM_FRAME_SLICE_BUDGET,
};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{Receiver, TryRecvError};
use wasm_bindgen::JsCast;

#[derive(Default)]
struct WakeRegistry {
    callbacks: HashMap<u32, js_sys::Function>,
    pending_wakes: HashSet<u32>,
}

thread_local! {
    static WAKE_REGISTRY: RefCell<WakeRegistry> = RefCell::new(WakeRegistry::default());
    static WAKE_LISTENER_INSTALLED: Cell<bool> = const { Cell::new(false) };
}

fn is_shared_wasm_memory() -> bool {
    let Ok(buffer) = js_sys::Reflect::get(
        &wasm_bindgen::memory(),
        &wasm_bindgen::JsValue::from_str("buffer"),
    ) else {
        return false;
    };
    buffer.is_instance_of::<js_sys::SharedArrayBuffer>()
}

fn build_worker_init_message(handle: u32) -> Result<js_sys::Object, wasm_bindgen::JsValue> {
    let msg = js_sys::Object::new();
    js_sys::Reflect::set(
        &msg,
        &wasm_bindgen::JsValue::from_str("type"),
        &wasm_bindgen::JsValue::from_str("init"),
    )?;
    js_sys::Reflect::set(
        &msg,
        &wasm_bindgen::JsValue::from_str("module"),
        &wasm_bindgen::module(),
    )?;
    js_sys::Reflect::set(
        &msg,
        &wasm_bindgen::JsValue::from_str("memory"),
        &wasm_bindgen::memory(),
    )?;
    js_sys::Reflect::set(
        &msg,
        &wasm_bindgen::JsValue::from_str("handle"),
        &wasm_bindgen::JsValue::from_f64(f64::from(handle)),
    )?;
    Ok(msg)
}

/// Spawns a dedicated Web Worker only when linear memory is backed by `SharedArrayBuffer`,
/// terminating the worker immediately if initialization message construction or posting fails.
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn spawn_web_worker(handle: u32) -> Option<web_sys::Worker> {
    if !is_shared_wasm_memory() {
        return None;
    }
    let worker_opts = web_sys::WorkerOptions::new();
    worker_opts.set_type(web_sys::WorkerType::Module);
    let worker = web_sys::Worker::new_with_options("worker.js", &worker_opts).ok()?;
    let Ok(msg) = build_worker_init_message(handle) else {
        worker.terminate();
        return None;
    };
    if worker.post_message(&msg).is_err() {
        worker.terminate();
        return None;
    }
    Some(worker)
}

/// Wakes the suspended simulation task associated with `handle`.
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn wake_worker(handle: u32) {
    let callback = WAKE_REGISTRY.with(|registry| {
        let mut reg = registry.borrow_mut();
        if let Some(cb) = reg.callbacks.remove(&handle) {
            Some(cb)
        } else {
            reg.pending_wakes.insert(handle);
            None
        }
    });
    if let Some(func) = callback {
        let _ = func.call0(&wasm_bindgen::JsValue::UNDEFINED);
    }
}

fn clear_wake_entry(handle: u32) {
    WAKE_REGISTRY.with(|registry| {
        let mut reg = registry.borrow_mut();
        reg.callbacks.remove(&handle);
        reg.pending_wakes.remove(&handle);
    });
}

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

async fn suspend_worker(handle: u32) {
    let already_woken =
        WAKE_REGISTRY.with(|registry| registry.borrow_mut().pending_wakes.remove(&handle));
    if already_woken {
        return;
    }
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let immediate = WAKE_REGISTRY.with(|registry| {
            let mut reg = registry.borrow_mut();
            if reg.pending_wakes.remove(&handle) {
                true
            } else {
                reg.callbacks.insert(handle, resolve.clone());
                false
            }
        });
        if immediate {
            let _ = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    WAKE_REGISTRY.with(|registry| {
        registry.borrow_mut().callbacks.remove(&handle);
    });
}

/// Installs an idempotent `DedicatedWorkerGlobalScope` message listener routing `wake` by `handle`.
pub fn setup_wake_listener() {
    let already_installed = WAKE_LISTENER_INSTALLED.with(|flag| flag.replace(true));
    if already_installed {
        return;
    }
    let global = js_sys::global();
    if let Ok(worker_scope) = global.dyn_into::<web_sys::DedicatedWorkerGlobalScope>() {
        let cb = wasm_bindgen::closure::Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
            let Ok(data) = e.data().dyn_into::<js_sys::Object>() else {
                return;
            };
            let is_wake = js_sys::Reflect::get(&data, &wasm_bindgen::JsValue::from_str("type"))
                .ok()
                .and_then(|v| v.as_string())
                .as_deref()
                == Some("wake");
            if !is_wake {
                return;
            }
            if let Some(handle) =
                js_sys::Reflect::get(&data, &wasm_bindgen::JsValue::from_str("handle"))
                    .ok()
                    .and_then(|v| v.as_f64())
                    .map(|v| v as u32)
            {
                wake_worker(handle);
            }
        })
            as Box<dyn FnMut(web_sys::MessageEvent)>);
        let _ =
            worker_scope.add_event_listener_with_callback("message", cb.as_ref().unchecked_ref());
        cb.forget();
    }
}

async fn poll_next_wasm_command(
    handle: u32,
    state: SimWorkerState,
    cmd_rx: &Receiver<SimCommand>,
) -> Result<Option<SimCommand>, ()> {
    if state.is_running() {
        return match cmd_rx.try_recv() {
            Ok(cmd) => Ok(Some(cmd)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(()),
        };
    }
    match cmd_rx.try_recv() {
        Ok(cmd) => Ok(Some(cmd)),
        Err(TryRecvError::Disconnected) => Err(()),
        Err(TryRecvError::Empty) => {
            suspend_worker(handle).await;
            match cmd_rx.try_recv() {
                Ok(cmd) => Ok(Some(cmd)),
                Err(TryRecvError::Empty) => Ok(None),
                Err(TryRecvError::Disconnected) => Err(()),
            }
        }
    }
}

/// Cooperative simulation worker entry point for WebAssembly.
#[wasm_bindgen::prelude::wasm_bindgen]
#[allow(missing_docs)]
pub async fn run_simulation_worker(handle: u32) {
    let Some(ctx) = take_worker_context(handle) else {
        clear_wake_entry(handle);
        return;
    };
    setup_wake_listener();
    let mut runner = ctx.runner;
    let cmd_rx = ctx.cmd_rx;
    let state_tx = ctx.state_tx;

    let mut state = SimWorkerState::Ready;
    let mut steps_per_frame = runner.get_steps_per_frame();
    if !state_tx.mark_seeded()
        && state_tx
            .send_snapshot_and_status(runner.get_snapshot(), WorkerStatusEvent::Ready)
            .is_err()
    {
        clear_wake_entry(handle);
        return;
    }

    loop {
        let Ok(cmd_opt) = poll_next_wasm_command(handle, state, &cmd_rx).await else {
            break;
        };

        if let Some(first_cmd) = cmd_opt {
            if !drain_worker_commands(
                first_cmd,
                &cmd_rx,
                &mut *runner,
                &mut state,
                &mut steps_per_frame,
                &state_tx,
            ) {
                break;
            }
            if !state.is_running() {
                yield_to_event_loop().await;
            }
        }

        if state.is_running() {
            if matches!(
                advance_running_frame(
                    &mut *runner,
                    steps_per_frame,
                    Some(WASM_FRAME_SLICE_BUDGET),
                    &mut state,
                    &state_tx,
                ),
                FrameTickOutcome::Disconnected
            ) {
                break;
            }
            yield_to_event_loop().await;
        }
    }

    clear_wake_entry(handle);
}
