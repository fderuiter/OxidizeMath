//! Frame-paced `requestAnimationFrame` coordinator and shared 8 ms slice budget for WASM workers.

// theory_verification!

use super::WASM_FRAME_SLICE_BUDGET;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Duration;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

const FALLBACK_FRAME_INTERVAL_MS: i32 = 16;
const RAF_WATCHDOG_INTERVAL_MS: i32 = 48;
const IDLE_EPOCH_REFRESH_THRESHOLD: Duration = Duration::from_millis(16);
const MIN_WORKER_SLICE_FLOOR: Duration = Duration::from_micros(100);

struct FrameCoordinator {
    frame_epoch: u64,
    epoch_elapsed: Duration,
    epoch_pending_workers: u32,
    epoch_started_at: Option<crate::web_io::PlatformInstant>,
    frame_scheduled: bool,
    scheduled_raf_id: Option<i32>,
    scheduled_timeout_id: Option<i32>,
    live_handles: HashSet<u32>,
    active_running: HashSet<u32>,
    suspended: HashSet<u32>,
    frame_waiters: VecDeque<(u32, js_sys::Function)>,
    idle_callbacks: HashMap<u32, js_sys::Function>,
    pending_wakes: HashSet<u32>,
    raf_closure: Option<Closure<dyn FnMut(f64)>>,
    timeout_closure: Option<Closure<dyn FnMut()>>,
}

impl Default for FrameCoordinator {
    fn default() -> Self {
        Self {
            frame_epoch: 1,
            epoch_elapsed: Duration::ZERO,
            epoch_pending_workers: 1,
            epoch_started_at: None,
            frame_scheduled: false,
            scheduled_raf_id: None,
            scheduled_timeout_id: None,
            live_handles: HashSet::new(),
            active_running: HashSet::new(),
            suspended: HashSet::new(),
            frame_waiters: VecDeque::new(),
            idle_callbacks: HashMap::new(),
            pending_wakes: HashSet::new(),
            raf_closure: None,
            timeout_closure: None,
        }
    }
}

thread_local! {
    static COORDINATOR: RefCell<FrameCoordinator> = RefCell::new(FrameCoordinator::default());
}

fn refresh_idle_epoch(coord: &mut FrameCoordinator) {
    if coord.frame_scheduled {
        return;
    }
    let stale = coord
        .epoch_started_at
        .is_none_or(|started| started.elapsed() >= IDLE_EPOCH_REFRESH_THRESHOLD);
    if stale {
        coord.frame_epoch = coord.frame_epoch.wrapping_add(1).max(1);
        coord.epoch_elapsed = Duration::ZERO;
        coord.epoch_pending_workers = (coord.active_running.len().max(1)) as u32;
        coord.epoch_started_at = Some(crate::web_io::PlatformInstant::now());
    }
}

pub(super) fn register_wasm_worker(handle: u32, suspended: bool) {
    COORDINATOR.with(|c| {
        let mut coord = c.borrow_mut();
        coord.live_handles.insert(handle);
        if suspended {
            coord.suspended.insert(handle);
        } else {
            coord.suspended.remove(&handle);
        }
    });
}

pub(super) fn acknowledge_wake(handle: u32) {
    COORDINATOR.with(|c| {
        c.borrow_mut().pending_wakes.remove(&handle);
    });
}

pub(super) fn current_frame_epoch() -> u64 {
    COORDINATOR.with(|c| {
        let mut coord = c.borrow_mut();
        refresh_idle_epoch(&mut coord);
        coord.frame_epoch
    })
}

pub(super) fn is_worker_suspended(handle: u32) -> bool {
    COORDINATOR.with(|c| {
        let coord = c.borrow();
        !coord.live_handles.contains(&handle) || coord.suspended.contains(&handle)
    })
}

pub(super) fn sync_controller_state(handle: u32, running: bool, suspended: bool) {
    let is_live = COORDINATOR.with(|c| {
        let mut coord = c.borrow_mut();
        if !coord.live_handles.contains(&handle) {
            return false;
        }
        if suspended {
            coord.suspended.insert(handle);
            coord.active_running.remove(&handle);
        } else {
            coord.suspended.remove(&handle);
            if running {
                coord.active_running.insert(handle);
            } else {
                coord.active_running.remove(&handle);
            }
        }
        true
    });
    if is_live {
        wake_worker(handle);
    }
}

#[cfg(test)]
pub(in crate::async_sim) fn test_active_running_count() -> usize {
    COORDINATOR.with(|c| c.borrow().active_running.len())
}

#[cfg(test)]
pub(in crate::async_sim) fn test_worker_slice_budget(handle: u32) -> Duration {
    COORDINATOR.with(|c| compute_worker_slice_budget(&mut c.borrow_mut(), handle, false))
}

#[cfg(test)]
pub(in crate::async_sim) fn test_current_epoch_elapsed() -> Duration {
    COORDINATOR.with(|c| c.borrow().epoch_elapsed)
}

pub(super) fn set_worker_running(handle: u32, running: bool) {
    COORDINATOR.with(|c| {
        let mut coord = c.borrow_mut();
        if running && coord.live_handles.contains(&handle) && !coord.suspended.contains(&handle) {
            coord.active_running.insert(handle);
        } else {
            coord.active_running.remove(&handle);
        }
    });
}

fn compute_worker_slice_budget(
    coord: &mut FrameCoordinator,
    handle: u32,
    consume: bool,
) -> Duration {
    refresh_idle_epoch(coord);
    if consume && coord.live_handles.contains(&handle) && !coord.suspended.contains(&handle) {
        coord.active_running.insert(handle);
    }
    let active_count = (coord.active_running.len().max(1)) as u32;
    let fair_share = WASM_FRAME_SLICE_BUDGET / active_count;
    let workers_left = coord.epoch_pending_workers.max(1);
    if consume {
        coord.epoch_pending_workers = coord.epoch_pending_workers.saturating_sub(1);
    }
    let remaining = WASM_FRAME_SLICE_BUDGET.saturating_sub(coord.epoch_elapsed);
    let dynamic_share = remaining / workers_left;
    fair_share.min(dynamic_share).max(MIN_WORKER_SLICE_FLOOR)
}

pub(super) fn allocate_worker_slice_budget(handle: u32) -> Duration {
    COORDINATOR.with(|c| compute_worker_slice_budget(&mut c.borrow_mut(), handle, true))
}

pub(in crate::async_sim) fn allocate_inline_slice_budget() -> (Duration, bool) {
    COORDINATOR.with(|c| {
        let mut coord = c.borrow_mut();
        if coord.active_running.is_empty() && !coord.frame_scheduled {
            return (WASM_FRAME_SLICE_BUDGET, false);
        }
        refresh_idle_epoch(&mut coord);
        let active_count = (coord.active_running.len().saturating_add(1)) as u32;
        let fair_share = WASM_FRAME_SLICE_BUDGET / active_count;
        let remaining = WASM_FRAME_SLICE_BUDGET.saturating_sub(coord.epoch_elapsed);
        (fair_share.min(remaining).max(MIN_WORKER_SLICE_FLOOR), true)
    })
}

pub(in crate::async_sim) fn record_worker_slice_elapsed(elapsed: Duration) {
    COORDINATOR.with(|c| {
        let mut coord = c.borrow_mut();
        coord.epoch_elapsed = coord.epoch_elapsed.saturating_add(elapsed);
    });
}

/// Wakes a suspended or frame-waiting simulation worker immediately (e.g. on command or drop).
pub(super) fn wake_worker(handle: u32) {
    let callback = COORDINATOR.with(|c| {
        let mut coord = c.borrow_mut();
        if let Some(cb) = coord.idle_callbacks.remove(&handle) {
            return Some(cb);
        }
        if let Some(pos) = coord.frame_waiters.iter().position(|(h, _)| *h == handle) {
            return coord.frame_waiters.remove(pos).map(|(_, cb)| cb);
        }
        if coord.live_handles.contains(&handle) {
            coord.pending_wakes.insert(handle);
        }
        None
    });
    if let Some(func) = callback {
        let _ = func.call0(&wasm_bindgen::JsValue::UNDEFINED);
    }
}

pub(super) fn clear_worker_entries(handle: u32) {
    COORDINATOR.with(|c| {
        let mut coord = c.borrow_mut();
        coord.live_handles.remove(&handle);
        coord.active_running.remove(&handle);
        coord.suspended.remove(&handle);
        coord.idle_callbacks.remove(&handle);
        coord.pending_wakes.remove(&handle);
        coord.frame_waiters.retain(|(h, _)| *h != handle);
    });
}

pub(super) async fn suspend_idle_worker(handle: u32) {
    let skip = COORDINATOR.with(|c| {
        let mut coord = c.borrow_mut();
        !coord.live_handles.contains(&handle) || coord.pending_wakes.remove(&handle)
    });
    if skip {
        return;
    }
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let immediate = COORDINATOR.with(|c| {
            let mut coord = c.borrow_mut();
            if !coord.live_handles.contains(&handle) || coord.pending_wakes.remove(&handle) {
                true
            } else {
                coord.idle_callbacks.insert(handle, resolve.clone());
                false
            }
        });
        if immediate {
            let _ = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    COORDINATOR.with(|c| {
        c.borrow_mut().idle_callbacks.remove(&handle);
    });
}

pub(super) async fn wait_for_next_animation_frame(handle: u32) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let immediate = COORDINATOR.with(|c| {
            let mut coord = c.borrow_mut();
            if !coord.live_handles.contains(&handle)
                || coord.pending_wakes.remove(&handle)
                || coord.suspended.contains(&handle)
            {
                true
            } else {
                coord.frame_waiters.retain(|(h, _)| *h != handle);
                coord.frame_waiters.push_back((handle, resolve.clone()));
                false
            }
        });
        if immediate {
            let _ = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
        } else {
            ensure_next_frame_scheduled();
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

fn ensure_next_frame_scheduled() {
    let need_schedule = COORDINATOR.with(|c| {
        let mut coord = c.borrow_mut();
        if coord.frame_scheduled || coord.frame_waiters.is_empty() {
            return false;
        }
        if coord.raf_closure.is_none() {
            coord.raf_closure = Some(Closure::wrap(Box::new(|_ts: f64| {
                on_animation_frame_fired(true);
            }) as Box<dyn FnMut(f64)>));
        }
        if coord.timeout_closure.is_none() {
            coord.timeout_closure = Some(Closure::wrap(Box::new(|| {
                on_animation_frame_fired(false);
            }) as Box<dyn FnMut()>));
        }
        coord.frame_scheduled = true;
        true
    });
    if !need_schedule {
        return;
    }
    schedule_browser_frame_callbacks();
}

fn schedule_browser_frame_callbacks() {
    let Ok(window) = js_sys::global().dyn_into::<web_sys::Window>() else {
        on_animation_frame_fired(false);
        return;
    };
    let doc_hidden = window.document().is_some_and(|doc| doc.hidden());
    let mut scheduled_any = false;
    COORDINATOR.with(|c| {
        let mut coord = c.borrow_mut();
        if !doc_hidden {
            if let Some(raf_cb) = coord.raf_closure.as_ref() {
                if let Ok(id) = window.request_animation_frame(raf_cb.as_ref().unchecked_ref()) {
                    coord.scheduled_raf_id = Some(id);
                    scheduled_any = true;
                }
            }
        }
        let delay = if scheduled_any {
            RAF_WATCHDOG_INTERVAL_MS
        } else {
            FALLBACK_FRAME_INTERVAL_MS
        };
        if let Some(timer_cb) = coord.timeout_closure.as_ref() {
            if let Ok(tid) = window.set_timeout_with_callback_and_timeout_and_arguments_0(
                timer_cb.as_ref().unchecked_ref(),
                delay,
            ) {
                coord.scheduled_timeout_id = Some(tid);
                scheduled_any = true;
            }
        }
    });
    if !scheduled_any {
        on_animation_frame_fired(false);
    }
}

fn on_animation_frame_fired(from_raf: bool) {
    let waiters: Vec<js_sys::Function> = COORDINATOR.with(|c| {
        let mut coord = c.borrow_mut();
        if !coord.frame_scheduled {
            return Vec::new();
        }
        coord.frame_scheduled = false;
        let raf_id = coord.scheduled_raf_id.take();
        let timeout_id = coord.scheduled_timeout_id.take();
        if let Ok(window) = js_sys::global().dyn_into::<web_sys::Window>() {
            if from_raf {
                if let Some(tid) = timeout_id {
                    window.clear_timeout_with_handle(tid);
                }
            } else if let Some(rid) = raf_id {
                let _ = window.cancel_animation_frame(rid);
            }
        }
        coord.frame_epoch = coord.frame_epoch.wrapping_add(1).max(1);
        coord.epoch_elapsed = Duration::ZERO;
        coord.epoch_pending_workers = (coord
            .frame_waiters
            .len()
            .max(coord.active_running.len())
            .max(1)) as u32;
        coord.epoch_started_at = Some(crate::web_io::PlatformInstant::now());
        if coord.frame_waiters.len() > 1 {
            coord.frame_waiters.rotate_left(1);
        }
        coord.frame_waiters.drain(..).map(|(_, cb)| cb).collect()
    });
    for func in waiters {
        let _ = func.call0(&wasm_bindgen::JsValue::UNDEFINED);
    }
}
