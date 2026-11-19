//! WebAssembly worker integration for async simulation.

use super::{
    take_worker_context, SimCommand, SimStateUpdate, SimWorkerState, SimulationRunner,
    WorkerStatusEvent,
};
use std::sync::mpsc::{Receiver, SyncSender};
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

fn process_wasm_commands(
    runner: &mut dyn SimulationRunner,
    first_cmd: SimCommand,
    cmd_rx: &Receiver<SimCommand>,
    state_tx: &SyncSender<SimStateUpdate>,
    state: &mut SimWorkerState,
    steps_per_frame: &mut usize,
) {
    let mut cmd = first_cmd;
    let mut emit_paused_snapshot = false;

    loop {
        match cmd {
            SimCommand::Start => {
                *state = SimWorkerState::Running;
                let _ = state_tx.try_send(SimStateUpdate::Status(WorkerStatusEvent::Running));
            }
            SimCommand::Pause => {
                *state = SimWorkerState::Ready;
                let _ = state_tx.try_send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
            }
            SimCommand::Reset => {
                *state = SimWorkerState::Ready;
                emit_paused_snapshot = false;
                runner.process_command(SimCommand::Reset);
                let snapshot = runner.get_snapshot();
                let _ = state_tx.try_send(SimStateUpdate::Snapshot(snapshot));
                let _ = state_tx.try_send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
            }
            SimCommand::SetSpeed(speed) => {
                *steps_per_frame = speed;
                runner.process_command(SimCommand::SetSpeed(speed));
            }
            SimCommand::TrainEpisodes(_) => {
                *state = SimWorkerState::Running;
                runner.process_command(cmd);
                let _ = state_tx.try_send(SimStateUpdate::Status(WorkerStatusEvent::Running));
            }
            SimCommand::ApplyBrush { .. } | SimCommand::ClearObstacles | SimCommand::Custom(_) => {
                runner.process_command(cmd);
                emit_paused_snapshot = true;
            }
        }
        if let Ok(next_cmd) = cmd_rx.try_recv() {
            cmd = next_cmd;
        } else {
            break;
        }
    }

    if emit_paused_snapshot && *state == SimWorkerState::Ready {
        let snapshot = runner.get_snapshot();
        let _ = state_tx.try_send(SimStateUpdate::Snapshot(snapshot));
    }
}

fn step_wasm_runner(
    runner: &mut dyn SimulationRunner,
    state_tx: &SyncSender<SimStateUpdate>,
    state: &mut SimWorkerState,
    steps_per_frame: usize,
) {
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
        *state = SimWorkerState::Faulted;
        let report = runner.diagnostic_report().unwrap_or_else(|| {
            crate::framework::recovery::DiagnosticReport::new(
                0.0,
                0,
                "state",
                "Non-finite floating-point state encountered",
            )
        });
        let _ = state_tx.try_send(SimStateUpdate::Status(WorkerStatusEvent::Faulted(
            report.message.clone(),
        )));
        let _ = state_tx.try_send(SimStateUpdate::Divergence(report));
        let snapshot = runner.get_snapshot();
        let _ = state_tx.try_send(SimStateUpdate::Snapshot(snapshot));
        return;
    }

    let is_done = runner.is_complete();
    let snapshot = runner.get_snapshot();
    if is_done {
        *state = SimWorkerState::Ready;
        let _ = state_tx.try_send(SimStateUpdate::Snapshot(snapshot));
        let _ = state_tx.try_send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
    } else {
        let _ = state_tx.try_send(SimStateUpdate::Snapshot(snapshot));
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
            process_wasm_commands(
                runner.as_mut(),
                first_cmd,
                &cmd_rx,
                &state_tx,
                &mut state,
                &mut steps_per_frame,
            );
        }

        if state.is_running() {
            step_wasm_runner(runner.as_mut(), &state_tx, &mut state, steps_per_frame);
            yield_to_event_loop().await;
        }
    }
}
