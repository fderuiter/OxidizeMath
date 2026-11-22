//! Post-boot WebGL context loss and in-place `eframe::WebRunner` restoration coordinator.
//!
//! Intercepts `webglcontextlost` and `webglcontextrestored` on `#the_canvas_id`, extracts and
//! persists [`WorkspaceRecoverySnapshot`], calls `WebRunner::destroy()` to immediately silence
//! input listeners on the lost session, restarts a fresh `WebRunner` + [`MathExplorerApp`] when
//! hardware rendering returns before timeout, and transitions to `#error_container` via
//! `reload.js` (with `.mws` workspace download support) when restoration times out or fails.

mod bridge;

use crate::accessibility;
use crate::app::session::WorkspaceRecoverySnapshot;
use crate::app::MathExplorerApp;
use bridge::{
    bind_fallback_download_listener, notify_js_begin_recovery, notify_recovery_completed,
    notify_recovery_failed, persist_envelope_to_local_storage, register_js_timeout_bridge,
    set_overlay_visible, LOSS_STATUS_MSG,
};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};

const DEFAULT_FALLBACK_RESTORE_TIMEOUT_MS: i32 = 8_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RecoveryPhase {
    Running,
    AwaitingRestore { epoch: u64 },
    Restoring { epoch: u64 },
    Failed,
}

struct RecoveryRuntime {
    canvas: web_sys::HtmlCanvasElement,
    active_runner: Option<eframe::WebRunner>,
    saved_snapshot: Option<WorkspaceRecoverySnapshot>,
    storage_blocked: bool,
    epoch: u64,
    phase: RecoveryPhase,
    fallback_timer_id: Option<i32>,
}

impl RecoveryRuntime {
    fn new(canvas: web_sys::HtmlCanvasElement, runner: eframe::WebRunner) -> Self {
        Self {
            canvas,
            active_runner: Some(runner),
            saved_snapshot: None,
            storage_blocked: false,
            epoch: 0,
            phase: RecoveryPhase::Running,
            fallback_timer_id: None,
        }
    }

    fn snapshot_metadata(&self) -> (String, String, bool) {
        let (envelope_json, filename) = self
            .saved_snapshot
            .as_ref()
            .map(|snap| (snap.envelope_json.clone(), snap.recovery_filename()))
            .unwrap_or_else(|| (String::new(), "math-explorer-recovery.mws".to_string()));
        (envelope_json, filename, self.storage_blocked)
    }
}

fn clear_fallback_timer(window: Option<&web_sys::Window>, runtime: &mut RecoveryRuntime) {
    if let (Some(win), Some(timer_id)) = (window, runtime.fallback_timer_id.take()) {
        win.clear_timeout_with_handle(timer_id);
    }
}

fn bind_download_button(doc: &web_sys::Document, runtime_rc: &Rc<RefCell<RecoveryRuntime>>) {
    let runtime_for_click = Rc::clone(runtime_rc);
    bind_fallback_download_listener(doc, move || {
        let (envelope_json, filename, _) = runtime_for_click.borrow().snapshot_metadata();
        (envelope_json, filename)
    });
}

fn handle_restore_timeout(runtime_rc: &Rc<RefCell<RecoveryRuntime>>, detail: &str) {
    let window = web_sys::window();
    let Some(doc) = window.as_ref().and_then(web_sys::Window::document) else {
        return;
    };
    let (stale_runner, envelope_json, filename, storage_blocked) = {
        let mut rt = runtime_rc.borrow_mut();
        if !rt.canvas.is_connected() {
            clear_fallback_timer(window.as_ref(), &mut rt);
            return;
        }
        if !matches!(
            rt.phase,
            RecoveryPhase::AwaitingRestore { .. } | RecoveryPhase::Restoring { .. }
        ) {
            return;
        }
        clear_fallback_timer(window.as_ref(), &mut rt);
        rt.phase = RecoveryPhase::Failed;
        let (env, name, blocked) = rt.snapshot_metadata();
        (rt.active_runner.take(), env, name, blocked)
    };
    if let Some(runner) = stale_runner {
        runner.destroy();
    }
    notify_recovery_failed(
        window.as_ref(),
        &doc,
        detail,
        &envelope_json,
        &filename,
        storage_blocked,
    );
}

fn arm_fallback_restore_timer(
    window: Option<&web_sys::Window>,
    runtime_rc: &Rc<RefCell<RecoveryRuntime>>,
    epoch: u64,
) {
    let Some(win) = window else {
        return;
    };
    let runtime_for_timer = Rc::clone(runtime_rc);
    let cb = Closure::once(move || {
        let matches_epoch = matches!(
            runtime_for_timer.borrow().phase,
            RecoveryPhase::AwaitingRestore { epoch: cur }
                | RecoveryPhase::Restoring { epoch: cur }
                if cur == epoch
        );
        if matches_epoch {
            handle_restore_timeout(&runtime_for_timer, "WebGL context restoration timed out.");
        }
    });
    if let Ok(timer_id) = win.set_timeout_with_callback_and_timeout_and_arguments_0(
        cb.as_ref().unchecked_ref(),
        DEFAULT_FALLBACK_RESTORE_TIMEOUT_MS,
    ) {
        runtime_rc.borrow_mut().fallback_timer_id = Some(timer_id);
    }
    cb.forget();
}

fn on_context_lost(event: &web_sys::Event, runtime_rc: &Rc<RefCell<RecoveryRuntime>>) {
    event.prevent_default();
    let window = web_sys::window();
    let Some(doc) = window.as_ref().and_then(web_sys::Window::document) else {
        return;
    };
    if crate::is_error_boundary_active(&doc) {
        return;
    }
    crate::audio::silence_all_audio_on_webgl_loss();

    let (epoch, runner_to_destroy, envelope_json, filename, storage_blocked) = {
        let mut rt = runtime_rc.borrow_mut();
        if rt.phase == RecoveryPhase::Failed {
            return;
        }
        clear_fallback_timer(window.as_ref(), &mut rt);
        rt.epoch = rt.epoch.wrapping_add(1);
        let epoch = rt.epoch;
        let runner = rt.active_runner.take();
        if let Some(ref active) = runner {
            if let Some(mut app) = active.app_mut::<MathExplorerApp>() {
                let snap = app.capture_recovery_snapshot();
                let blocked =
                    !persist_envelope_to_local_storage(window.as_ref(), &snap.envelope_json);
                rt.saved_snapshot = Some(snap);
                rt.storage_blocked = blocked;
            }
        }
        rt.phase = RecoveryPhase::AwaitingRestore { epoch };
        let (env, name, blocked) = rt.snapshot_metadata();
        (epoch, runner, env, name, blocked)
    };

    if let Some(runner) = runner_to_destroy {
        runner.destroy();
    }

    bind_download_button(&doc, runtime_rc);
    if !notify_js_begin_recovery(window.as_ref(), &envelope_json, &filename, storage_blocked) {
        set_overlay_visible(&doc, true);
        accessibility::announce_status_with_priority(LOSS_STATUS_MSG, "polite");
        arm_fallback_restore_timer(window.as_ref(), runtime_rc, epoch);
    }
}

fn spawn_runner_restart(
    canvas: web_sys::HtmlCanvasElement,
    runtime_rc: Rc<RefCell<RecoveryRuntime>>,
    snapshot: Option<WorkspaceRecoverySnapshot>,
    epoch: u64,
) {
    wasm_bindgen_futures::spawn_local(async move {
        let new_runner = eframe::WebRunner::new();
        let ctx_slot: Rc<RefCell<Option<eframe::egui::Context>>> = Rc::new(RefCell::new(None));
        let ctx_slot_init = Rc::clone(&ctx_slot);

        let start_res = new_runner
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(move |cc| {
                    cc.egui_ctx.options_mut(|o| o.screen_reader = true);
                    let mut app = MathExplorerApp::new(cc);
                    if let Some(ref snap) = snapshot {
                        app.restore_recovery_snapshot(snap, &cc.egui_ctx);
                    }
                    *ctx_slot_init.borrow_mut() = Some(cc.egui_ctx.clone());
                    Ok(Box::new(app))
                }),
            )
            .await;

        finish_runner_restart(new_runner, start_res, &runtime_rc, &ctx_slot, epoch);
    });
}

fn finish_runner_restart(
    new_runner: eframe::WebRunner,
    start_res: Result<(), JsValue>,
    runtime_rc: &Rc<RefCell<RecoveryRuntime>>,
    ctx_slot: &Rc<RefCell<Option<eframe::egui::Context>>>,
    epoch: u64,
) {
    let window = web_sys::window();
    let Some(doc) = window.as_ref().and_then(web_sys::Window::document) else {
        new_runner.destroy();
        return;
    };
    let still_current = runtime_rc.borrow().phase == RecoveryPhase::Restoring { epoch }
        && !crate::is_error_boundary_active(&doc);
    if !still_current {
        new_runner.destroy();
        return;
    }

    match start_res {
        Ok(()) => {
            {
                let mut rt = runtime_rc.borrow_mut();
                clear_fallback_timer(window.as_ref(), &mut rt);
                rt.active_runner = Some(new_runner);
                rt.phase = RecoveryPhase::Running;
            }
            notify_recovery_completed(window.as_ref(), &doc);
            if let Some(ctx) = ctx_slot.borrow().as_ref() {
                ctx.request_repaint();
            }
        }
        Err(err) => {
            new_runner.destroy();
            let (envelope_json, filename, storage_blocked) = {
                let mut rt = runtime_rc.borrow_mut();
                clear_fallback_timer(window.as_ref(), &mut rt);
                rt.phase = RecoveryPhase::Failed;
                rt.snapshot_metadata()
            };
            let detail = format!("Failed to restart WebRunner after WebGL restoration: {err:?}");
            notify_recovery_failed(
                window.as_ref(),
                &doc,
                &detail,
                &envelope_json,
                &filename,
                storage_blocked,
            );
        }
    }
}

fn on_context_restored(
    canvas: &web_sys::HtmlCanvasElement,
    runtime_rc: &Rc<RefCell<RecoveryRuntime>>,
) {
    let window = web_sys::window();
    let Some(doc) = window.as_ref().and_then(web_sys::Window::document) else {
        return;
    };
    if crate::is_error_boundary_active(&doc) {
        return;
    }

    let (epoch, snapshot) = {
        let mut rt = runtime_rc.borrow_mut();
        let RecoveryPhase::AwaitingRestore { epoch } = rt.phase else {
            return;
        };
        rt.phase = RecoveryPhase::Restoring { epoch };
        (epoch, rt.saved_snapshot.clone())
    };

    spawn_runner_restart(canvas.clone(), Rc::clone(runtime_rc), snapshot, epoch);
}

/// Retains `runner` and attaches `webglcontextlost` and `webglcontextrestored` handlers to `canvas`.
pub(crate) fn install_webgl_recovery(
    canvas: &web_sys::HtmlCanvasElement,
    doc: &web_sys::Document,
    window: Option<&web_sys::Window>,
    runner: eframe::WebRunner,
) {
    let runtime_rc = Rc::new(RefCell::new(RecoveryRuntime::new(canvas.clone(), runner)));
    bind_download_button(doc, &runtime_rc);
    let timeout_rt = Rc::clone(&runtime_rc);
    register_js_timeout_bridge(window, move |msg| {
        handle_restore_timeout(&timeout_rt, &msg);
    });

    let lost_rt = Rc::clone(&runtime_rc);
    let on_lost = Closure::wrap(Box::new(move |event: web_sys::Event| {
        on_context_lost(&event, &lost_rt);
    }) as Box<dyn FnMut(web_sys::Event)>);
    let _ = canvas
        .add_event_listener_with_callback("webglcontextlost", on_lost.as_ref().unchecked_ref());
    on_lost.forget();

    let restored_canvas = canvas.clone();
    let restored_rt = Rc::clone(&runtime_rc);
    let on_restored = Closure::wrap(Box::new(move |_event: web_sys::Event| {
        on_context_restored(&restored_canvas, &restored_rt);
    }) as Box<dyn FnMut(web_sys::Event)>);
    let _ = canvas.add_event_listener_with_callback(
        "webglcontextrestored",
        on_restored.as_ref().unchecked_ref(),
    );
    on_restored.forget();
}
