#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! WAI-ARIA twin-slot polite/assertive live-region bridge and startup readiness announcer.

use std::sync::atomic::{AtomicBool, Ordering};

/// Canonical polite status message emitted once when the interactive workspace finishes booting.
pub const READY_ANNOUNCEMENT: &str =
    "Math Explorer ready. Interactive simulation workspace loaded.";

#[cfg(target_arch = "wasm32")]
const SR_ONLY_STYLE: &str =
    "position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap;";

static STARTUP_READY_ANNOUNCED: AtomicBool = AtomicBool::new(false);

#[cfg(target_arch = "wasm32")]
static POLITE_SLOT_TOGGLE: AtomicBool = AtomicBool::new(false);

#[cfg(target_arch = "wasm32")]
static ASSERTIVE_SLOT_TOGGLE: AtomicBool = AtomicBool::new(false);

/// State container modeling WAI-ARIA twin-slot ping-pong live-region alternation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TwinSlotLiveRegion {
    /// Active slot index (`0` or `1`).
    pub active_slot: usize,
    /// Text contents of slot `0` and slot `1`.
    pub slots: [String; 2],
}

impl TwinSlotLiveRegion {
    /// Creates a new empty twin-slot live region.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Alternates to the sibling slot, clears the previous slot, and populates the new active slot.
    pub fn dispatch(&mut self, message: &str) -> usize {
        let next_slot = 1 - self.active_slot;
        self.slots[self.active_slot].clear();
        self.slots[next_slot] = message.to_string();
        self.active_slot = next_slot;
        next_slot
    }

    /// Synchronizes a specific slot index (`0` or `1`) with DOM slot alternation state.
    pub fn set_active_slot(&mut self, slot_idx: usize, message: &str) {
        let clamped = slot_idx.min(1);
        self.slots[1 - clamped].clear();
        self.slots[clamped] = message.to_string();
        self.active_slot = clamped;
    }
}

#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn next_live_slot(toggle: &AtomicBool) -> usize {
    usize::from(toggle.fetch_xor(true, Ordering::SeqCst))
}

#[cfg(target_arch = "wasm32")]
fn mount_live_container(
    document: &web_sys::Document,
    body: &web_sys::HtmlElement,
    id: &str,
    role: &str,
    politeness: &str,
) {
    if document.get_element_by_id(id).is_some() {
        return;
    }
    let Ok(container) = document.create_element("div") else {
        return;
    };
    container.set_id(id);
    let _ = container.set_attribute("role", role);
    let _ = container.set_attribute("aria-live", politeness);
    let _ = container.set_attribute("aria-atomic", "true");
    let _ = container.set_attribute("class", "sr-only");
    let _ = container.set_attribute("style", SR_ONLY_STYLE);

    for slot_idx in [0_usize, 1] {
        if let Ok(slot) = document.create_element("span") {
            slot.set_id(&format!("{id}-slot-{slot_idx}"));
            let _ = container.append_child(&slot);
        }
    }
    let _ = body.append_child(&container);
}

/// Mounts `#aria-live-polite` and `#aria-live-assertive` twin-slot live regions and binds `egui_plot`.
#[cfg(target_arch = "wasm32")]
pub fn init_accessibility_bridge() {
    if let Some(document) = web_sys::window().and_then(|w| w.document()) {
        if let Some(body) = document.body() {
            mount_live_container(&document, &body, "aria-live-polite", "status", "polite");
            mount_live_container(
                &document,
                &body,
                "aria-live-assertive",
                "alert",
                "assertive",
            );
        }
    }
    ensure_plot_live_bridge();
}

/// Initializes the host accessibility bridge and binds `egui_plot` live announcements.
#[cfg(not(target_arch = "wasm32"))]
pub fn init_accessibility_bridge() {
    ensure_plot_live_bridge();
}

/// Ensures `egui_plot` live messages route directly into [`TwinSlotLiveRegion`] and root `ctx` data.
pub fn ensure_plot_live_bridge() {
    egui_plot::set_live_region_announcer(|ctx, message, assertive| {
        if assertive {
            announce_live_assertive_status(ctx, message);
        } else {
            announce_live_status(ctx, message);
        }
    });
}

/// Synchronizes `aria_live_message` and `aria_live_assertive_message` temp keys on `ctx`
/// with [`TwinSlotLiveRegion`], optionally clearing previous-frame keys after dispatch.
pub fn sync_context_live_messages(ctx: &eframe::egui::Context, clear_keys: bool) {
    for (msg_key, flag_key, priority) in [
        ("aria_live_message", "aria_live_dispatched", "polite"),
        (
            "aria_live_assertive_message",
            "aria_live_assertive_dispatched",
            "assertive",
        ),
    ] {
        let pending = ctx.data_mut(|d| {
            let m_id = eframe::egui::Id::new(msg_key);
            let f_id = eframe::egui::Id::new(flag_key);
            let msg = d.get_temp::<String>(m_id);
            let dispatched = d.get_temp::<bool>(f_id).unwrap_or(false);
            if clear_keys {
                d.remove::<String>(m_id);
                d.remove::<bool>(f_id);
            } else if msg.is_some() && !dispatched {
                d.insert_temp(f_id, true);
            }
            msg.filter(|_| !dispatched)
        });
        if let Some(msg) = pending {
            announce_status_with_priority(&msg, priority);
            if !clear_keys {
                let slot = with_live_region(|r| r.active_slot);
                ctx.data_mut(|d| d.insert_temp(eframe::egui::Id::new("aria_live_slot"), slot));
            }
        }
    }
}

thread_local! {
    static LIVE_REGION: std::cell::RefCell<TwinSlotLiveRegion> =
        std::cell::RefCell::new(TwinSlotLiveRegion::new());
}

#[cfg(test)]
thread_local! {
    static TEST_ANNOUNCEMENTS: std::cell::RefCell<Vec<(String, String)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Inspects the thread-local [`TwinSlotLiveRegion`] state backing screen-reader announcements.
pub fn with_live_region<R>(f: impl FnOnce(&TwinSlotLiveRegion) -> R) -> R {
    LIVE_REGION.with(|region| f(&region.borrow()))
}

#[cfg(target_arch = "wasm32")]
fn resolve_dom_live_slot(
    document: &web_sys::Document,
    container_id: &str,
    toggle: &AtomicBool,
) -> usize {
    let text0 = document
        .get_element_by_id(&format!("{container_id}-slot-0"))
        .and_then(|el| el.text_content())
        .unwrap_or_default();
    let text1 = document
        .get_element_by_id(&format!("{container_id}-slot-1"))
        .and_then(|el| el.text_content())
        .unwrap_or_default();
    let slot_idx = if !text0.trim().is_empty() && text1.trim().is_empty() {
        toggle.store(false, Ordering::SeqCst);
        1
    } else if text0.trim().is_empty() && !text1.trim().is_empty() {
        toggle.store(true, Ordering::SeqCst);
        0
    } else {
        next_live_slot(toggle)
    };
    slot_idx
}

/// Announces a polite status message to `#aria-live-polite`.
pub fn announce_status(message: &str) {
    announce_status_with_priority(message, "polite");
}

/// Announces a status message with `"polite"` or `"assertive"` priority, synchronized with `reload.js` twin slots.
#[cfg(target_arch = "wasm32")]
pub fn announce_status_with_priority(message: &str, priority: &str) {
    #[cfg(test)]
    TEST_ANNOUNCEMENTS.with(|a| {
        a.borrow_mut()
            .push((message.to_string(), priority.to_string()));
    });

    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        LIVE_REGION.with(|region| {
            let _ = region.borrow_mut().dispatch(message);
        });
        return;
    };

    let is_assertive =
        priority.eq_ignore_ascii_case("assertive") || priority.eq_ignore_ascii_case("alert");
    let container_id = if is_assertive {
        "aria-live-assertive"
    } else {
        "aria-live-polite"
    };
    if document.get_element_by_id(container_id).is_none() {
        init_accessibility_bridge();
    }
    let toggle = if is_assertive {
        &ASSERTIVE_SLOT_TOGGLE
    } else {
        &POLITE_SLOT_TOGGLE
    };
    let slot_idx = resolve_dom_live_slot(&document, container_id, toggle);
    let sibling_idx = 1 - slot_idx;

    if let Some(sibling_slot) =
        document.get_element_by_id(&format!("{container_id}-slot-{sibling_idx}"))
    {
        sibling_slot.set_text_content(Some(""));
    }
    if let Some(active_slot) =
        document.get_element_by_id(&format!("{container_id}-slot-{slot_idx}"))
    {
        active_slot.set_text_content(Some(message));
    }
    LIVE_REGION.with(|region| {
        region.borrow_mut().set_active_slot(slot_idx, message);
    });
}

/// Announces a status message with `"polite"` or `"assertive"` priority on native targets.
#[cfg(not(target_arch = "wasm32"))]
pub fn announce_status_with_priority(message: &str, priority: &str) {
    LIVE_REGION.with(|region| {
        let _ = region.borrow_mut().dispatch(message);
    });
    #[cfg(test)]
    TEST_ANNOUNCEMENTS.with(|a| {
        a.borrow_mut()
            .push((message.to_string(), priority.to_string()));
    });
    #[cfg(not(test))]
    let _ = priority;
}

#[cfg(target_arch = "wasm32")]
fn dom_has_ready_announcement(document: &web_sys::Document) -> Option<usize> {
    for idx in [0_usize, 1] {
        let text = document
            .get_element_by_id(&format!("aria-live-polite-slot-{idx}"))
            .and_then(|el| el.text_content())
            .unwrap_or_default();
        if text.trim() == READY_ANNOUNCEMENT {
            return Some(idx);
        }
    }
    None
}

/// Emits [`READY_ANNOUNCEMENT`] to `#aria-live-polite` at most once across JS `markBootComplete` and Rust startup.
#[cfg(target_arch = "wasm32")]
pub fn announce_startup_ready_once() -> bool {
    if let Some(document) = web_sys::window().and_then(|w| w.document()) {
        if let Some(slot_idx) = dom_has_ready_announcement(&document) {
            STARTUP_READY_ANNOUNCED.store(true, Ordering::SeqCst);
            POLITE_SLOT_TOGGLE.store(slot_idx == 0, Ordering::SeqCst);
            LIVE_REGION.with(|region| {
                region
                    .borrow_mut()
                    .set_active_slot(slot_idx, READY_ANNOUNCEMENT);
            });
            return false;
        }
    }
    if STARTUP_READY_ANNOUNCED.swap(true, Ordering::SeqCst) {
        return false;
    }
    announce_status(READY_ANNOUNCEMENT);
    true
}

/// Emits [`READY_ANNOUNCEMENT`] once per startup lifecycle on native targets.
#[cfg(not(target_arch = "wasm32"))]
pub fn announce_startup_ready_once() -> bool {
    if STARTUP_READY_ANNOUNCED.swap(true, Ordering::SeqCst) {
        return false;
    }
    announce_status(READY_ANNOUNCEMENT);
    true
}

/// Announces a polite status message to the live region and stores it in `ctx` temp data.
pub fn announce_live_status(ctx: &eframe::egui::Context, message: impl Into<String>) {
    let message = message.into();
    announce_status(&message);
    let active_slot = with_live_region(|r| r.active_slot);
    ctx.data_mut(|d| {
        d.insert_temp(eframe::egui::Id::new("aria_live_message"), message);
        d.insert_temp(eframe::egui::Id::new("aria_live_slot"), active_slot);
        d.insert_temp(eframe::egui::Id::new("aria_live_dispatched"), true);
    });
}

/// Announces an assertive alert message to the live region and stores it in `ctx` temp data.
pub fn announce_live_assertive_status(ctx: &eframe::egui::Context, message: impl Into<String>) {
    let message = message.into();
    announce_status_with_priority(&message, "assertive");
    let active_slot = with_live_region(|r| r.active_slot);
    ctx.data_mut(|d| {
        d.insert_temp(
            eframe::egui::Id::new("aria_live_assertive_message"),
            message,
        );
        d.insert_temp(eframe::egui::Id::new("aria_live_slot"), active_slot);
        d.insert_temp(
            eframe::egui::Id::new("aria_live_assertive_dispatched"),
            true,
        );
    });
}

/// Returns all recorded `(message, priority)` status announcements for the current test thread.
#[cfg(test)]
pub(crate) fn all_announcements() -> Vec<(String, String)> {
    TEST_ANNOUNCEMENTS.with(|a| a.borrow().clone())
}

/// Clears all recorded status announcements and resets the twin-slot live region for the current test thread.
#[cfg(test)]
pub(crate) fn clear_test_announcements() {
    LIVE_REGION.with(|region| {
        *region.borrow_mut() = TwinSlotLiveRegion::new();
    });
    TEST_ANNOUNCEMENTS.with(|a| a.borrow_mut().clear());
}

/// Resets the single-shot startup readiness flag between unit and browser tests.
#[cfg(test)]
pub(crate) fn reset_startup_ready_flag() {
    STARTUP_READY_ANNOUNCED.store(false, Ordering::SeqCst);
}

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "live_region_wasm_tests.rs"]
mod wasm_tests;
