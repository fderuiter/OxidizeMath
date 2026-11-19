//! WebAssembly worker integration for async simulation.

use super::worker::{advance_running_frame, drain_worker_commands, FrameTickOutcome};
use super::{take_worker_context, SimStateUpdate, SimWorkerState, WorkerStatusEvent};
use wasm_bindgen::JsCast;

fn wake_map_object(global: &wasm_bindgen::JsValue) -> Option<js_sys::Object> {
    let key = wasm_bindgen::JsValue::from_str("__sim_worker_wakers");
    if let Ok(existing) = js_sys::Reflect::get(global, &key) {
        if let Ok(obj) = existing.dyn_into::<js_sys::Object>() {
            return Some(obj);
        }
    }
    let created = js_sys::Object::new();
    js_sys::Reflect::set(global, &key, &created).ok()?;
    Some(created)
}

/// Attempts to spawn a dedicated Web Worker initialized with shared WASM module & memory.
pub(crate) fn spawn_web_worker(handle: u32) -> Option<web_sys::Worker> {
    let has_shared_memory = js_sys::Reflect::get(
        &wasm_bindgen::memory(),
        &wasm_bindgen::JsValue::from_str("buffer"),
    )
    .map(|buf| buf.is_instance_of::<js_sys::SharedArrayBuffer>())
    .unwrap_or(false);
    if !has_shared_memory {
        return None;
    }

    let worker_opts = web_sys::WorkerOptions::new();
    worker_opts.set_type(web_sys::WorkerType::Module);
    let worker = web_sys::Worker::new_with_options("worker.js", &worker_opts).ok()?;
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
        &wasm_bindgen::JsValue::from_f64(f64::from(handle)),
    );
    if worker.post_message(&msg).is_ok() {
        Some(worker)
    } else {
        worker.terminate();
        None
    }
}

/// Wakes up the Web Worker (or any suspended local WASM workers) if suspended.
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
    if let Some(map_obj) = wake_map_object(&global) {
        let values = js_sys::Object::values(&map_obj);
        for idx in 0..values.length() {
            let entry = values.get(idx);
            if entry.is_function() {
                let func = entry.unchecked_into::<js_sys::Function>();
                let _ = func.call0(&wasm_bindgen::JsValue::UNDEFINED);
            }
        }
    }
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
    let handle_key = handle.to_string();
    let is_dedicated_worker = js_sys::global()
        .dyn_into::<web_sys::DedicatedWorkerGlobalScope>()
        .is_ok();
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let global = js_sys::global();
        if let Some(map_obj) = wake_map_object(&global) {
            let _ = js_sys::Reflect::set(
                &map_obj,
                &wasm_bindgen::JsValue::from_str(&handle_key),
                &resolve,
            );
        }
        if is_dedicated_worker {
            let _ = js_sys::Reflect::set(
                &global,
                &wasm_bindgen::JsValue::from_str("__sim_worker_wake"),
                &resolve,
            );
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    let global = js_sys::global();
    if let Some(map_obj) = wake_map_object(&global) {
        let _ = js_sys::Reflect::delete_property(
            &map_obj,
            &wasm_bindgen::JsValue::from_str(&handle_key),
        );
    }
    if is_dedicated_worker {
        let _ = js_sys::Reflect::delete_property(
            &global,
            &wasm_bindgen::JsValue::from_str("__sim_worker_wake"),
        );
    }
}

/// Sets up the event listener for wake messages in DedicatedWorkerGlobalScope.
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
        cb.forget();
    }
}

/// Simulation worker entry point for WebAssembly.
#[wasm_bindgen::prelude::wasm_bindgen]
#[allow(missing_docs)]
pub async fn run_simulation_worker(handle: u32) {
    let Some(ctx) = take_worker_context(handle) else {
        return;
    };
    setup_wake_listener();
    let mut runner = ctx.runner;
    let cmd_rx = ctx.cmd_rx;
    let state_tx = ctx.state_tx;

    let mut state = SimWorkerState::Ready;
    let mut steps_per_frame = runner.get_steps_per_frame();
    let _ = state_tx.try_send(SimStateUpdate::Snapshot(runner.get_snapshot()));
    let _ = state_tx.try_send(SimStateUpdate::Status(WorkerStatusEvent::Ready));

    loop {
        let cmd_opt = match cmd_rx.try_recv() {
            Ok(c) => Some(c),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
            Err(std::sync::mpsc::TryRecvError::Empty) if state.is_running() => None,
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                suspend_worker(handle).await;
                match cmd_rx.try_recv() {
                    Ok(c) => Some(c),
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
                    Err(std::sync::mpsc::TryRecvError::Empty) => None,
                }
            }
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
                advance_running_frame(&mut *runner, steps_per_frame, &mut state, &state_tx),
                FrameTickOutcome::Disconnected
            ) {
                break;
            }
            yield_to_event_loop().await;
        }
    }
}
