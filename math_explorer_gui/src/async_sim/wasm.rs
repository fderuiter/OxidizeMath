//! WebAssembly worker integration for async simulation.

use super::{take_worker_context, SimCommand, SimStateUpdate, SimWorkerState, WorkerStatusEvent};
use wasm_bindgen::JsCast;

/// Wakes up the Web Worker if it is suspended.
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
    let global = js_sys::global();
    let _ = js_sys::Reflect::delete_property(
        &global,
        &wasm_bindgen::JsValue::from_str("__sim_worker_wake"),
    );
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
#[allow(missing_docs, clippy::too_many_lines, clippy::cognitive_complexity)]
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
    let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));

    loop {
        let mut cmd_opt = None;
        if state.is_running() {
            if let Ok(c) = cmd_rx.try_recv() {
                cmd_opt = Some(c);
            }
        } else if let Ok(c) = cmd_rx.try_recv() {
            cmd_opt = Some(c);
        } else {
            suspend_worker().await;
            if let Ok(c) = cmd_rx.try_recv() {
                cmd_opt = Some(c);
            }
        }

        if let Some(mut cmd) = cmd_opt {
            loop {
                match cmd {
                    SimCommand::Start => {
                        state = SimWorkerState::Running;
                        let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Running));
                    }
                    SimCommand::Pause => {
                        state = SimWorkerState::Ready;
                        let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
                    }
                    SimCommand::Reset => {
                        state = SimWorkerState::Ready;
                        runner.process_command(SimCommand::Reset);
                        let snapshot = runner.get_snapshot();
                        let _ = state_tx.send(SimStateUpdate::Snapshot(snapshot));
                        let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
                    }
                    SimCommand::SetSpeed(speed) => {
                        steps_per_frame = speed;
                        runner.process_command(SimCommand::SetSpeed(speed));
                    }
                    SimCommand::TrainEpisodes(_) => {
                        state = SimWorkerState::Running;
                        runner.process_command(cmd);
                        let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Running));
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
                if runner.is_complete() {
                    break;
                }
            }
            let is_done = runner.is_complete();
            let snapshot = runner.get_snapshot();
            if is_done {
                state = SimWorkerState::Ready;
                let _ = state_tx.send(SimStateUpdate::Snapshot(snapshot));
                let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
            } else {
                let _ = state_tx.try_send(SimStateUpdate::Snapshot(snapshot));
            }
            yield_to_event_loop().await;
        }
    }
}
