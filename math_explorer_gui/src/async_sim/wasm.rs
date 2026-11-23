//! Cooperative WebAssembly simulation scheduler and handle-keyed wake registry.

// theory_verification!

use super::worker::{advance_running_frame, drain_worker_commands, FrameTickOutcome};
use super::{
    take_worker_context, SimCommand, SimWorkerState, WorkerStatusEvent, WASM_FRAME_SLICE_BUDGET,
};
use std::cell::RefCell;
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
}

/// Wakes the suspended simulation task associated with `handle`.
pub(super) fn wake_worker(handle: u32) {
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
        let timeout_scheduled = if let Ok(window) = js_sys::global().dyn_into::<web_sys::Window>() {
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
pub(super) async fn run_simulation_worker(handle: u32) {
    let Some(ctx) = take_worker_context(handle) else {
        clear_wake_entry(handle);
        return;
    };
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
