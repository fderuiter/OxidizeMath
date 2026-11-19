use eframe::egui::{Response, WidgetText};
use scientific_metadata::theory::TheoryDescribable;

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
}

#[cfg(any(target_arch = "wasm32", test))]
fn next_live_slot(toggle: &std::sync::atomic::AtomicBool) -> usize {
    usize::from(toggle.fetch_xor(true, std::sync::atomic::Ordering::SeqCst))
}

#[cfg(target_arch = "wasm32")]
static POLITE_SLOT_TOGGLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(target_arch = "wasm32")]
static ASSERTIVE_SLOT_TOGGLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(target_arch = "wasm32")]
// theory_verification!
pub fn init_accessibility_bridge() {
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(body) = document.body() {
                // Technique ARIA22: Polite container with role="status", aria-live="polite", and aria-atomic="true"
                if document.get_element_by_id("aria-live-polite").is_none() {
                    if let Ok(container) = document.create_element("div") {
                        container.set_id("aria-live-polite");
                        let _ = container.set_attribute("role", "status");
                        let _ = container.set_attribute("aria-live", "polite");
                        let _ = container.set_attribute("aria-atomic", "true");
                        let _ = container.set_attribute("class", "sr-only");
                        let _ = container.set_attribute(
                            "style",
                            "position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap;",
                        );

                        if let Ok(slot0) = document.create_element("span") {
                            slot0.set_id("aria-live-polite-slot-0");
                            let _ = container.append_child(&slot0);
                        }
                        if let Ok(slot1) = document.create_element("span") {
                            slot1.set_id("aria-live-polite-slot-1");
                            let _ = container.append_child(&slot1);
                        }

                        let _ = body.append_child(&container);
                    }
                }

                // Technique ARIA19: Assertive container with role="alert", aria-live="assertive", and aria-atomic="true"
                if document.get_element_by_id("aria-live-assertive").is_none() {
                    if let Ok(container) = document.create_element("div") {
                        container.set_id("aria-live-assertive");
                        let _ = container.set_attribute("role", "alert");
                        let _ = container.set_attribute("aria-live", "assertive");
                        let _ = container.set_attribute("aria-atomic", "true");
                        let _ = container.set_attribute("class", "sr-only");
                        let _ = container.set_attribute(
                            "style",
                            "position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap;",
                        );

                        if let Ok(slot0) = document.create_element("span") {
                            slot0.set_id("aria-live-assertive-slot-0");
                            let _ = container.append_child(&slot0);
                        }
                        if let Ok(slot1) = document.create_element("span") {
                            slot1.set_id("aria-live-assertive-slot-1");
                            let _ = container.append_child(&slot1);
                        }

                        let _ = body.append_child(&container);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
thread_local! {
    static TEST_ANNOUNCEMENTS: std::cell::RefCell<Vec<(String, String)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(missing_docs)]
pub fn init_accessibility_bridge() {}

thread_local! {
    static LIVE_REGION: std::cell::RefCell<TwinSlotLiveRegion> =
        std::cell::RefCell::new(TwinSlotLiveRegion::new());
}

/// Inspects the thread-local [`TwinSlotLiveRegion`] state backing screen-reader announcements.
pub fn with_live_region<R>(f: impl FnOnce(&TwinSlotLiveRegion) -> R) -> R {
    LIVE_REGION.with(|region| f(&region.borrow()))
}

#[cfg(target_arch = "wasm32")]
// theory_verification!
pub fn announce_status(message: &str) {
    announce_status_with_priority(message, "polite");
}

#[cfg(target_arch = "wasm32")]
pub fn announce_status_with_priority(message: &str, priority: &str) {
    #[cfg(test)]
    TEST_ANNOUNCEMENTS.with(|a| {
        a.borrow_mut()
            .push((message.to_string(), priority.to_string()));
    });

    LIVE_REGION.with(|region| {
        let mut r = region.borrow_mut();
        let active = r.dispatch(message);
        let _ = &r.slots[active];
    });

    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            let is_assertive = priority.eq_ignore_ascii_case("assertive")
                || priority.eq_ignore_ascii_case("alert");
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
            let slot_idx = next_live_slot(toggle);
            let sibling_idx = 1 - slot_idx;

            let active_id = format!("{container_id}-slot-{slot_idx}");
            let sibling_id = format!("{container_id}-slot-{sibling_idx}");

            if let Some(sibling_slot) = document.get_element_by_id(&sibling_id) {
                sibling_slot.set_text_content(Some(""));
            }

            if let Some(active_slot) = document.get_element_by_id(&active_id) {
                active_slot.set_text_content(Some(message));
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(missing_docs)]
pub fn announce_status(message: &str) {
    announce_status_with_priority(message, "polite");
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(missing_docs)]
pub fn announce_status_with_priority(message: &str, priority: &str) {
    LIVE_REGION.with(|region| {
        let mut r = region.borrow_mut();
        let active = r.dispatch(message);
        let _ = &r.slots[active];
    });
    #[cfg(test)]
    TEST_ANNOUNCEMENTS.with(|a| {
        a.borrow_mut()
            .push((message.to_string(), priority.to_string()));
    });
    #[cfg(not(test))]
    let _ = priority;
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

#[allow(missing_docs)]
pub trait AccessibleHoverText {
    #[allow(missing_docs)]
    fn accessible_hover_text(self, text: impl Into<WidgetText>) -> Self;
}

impl AccessibleHoverText for Response {
    fn accessible_hover_text(self, text: impl Into<WidgetText>) -> Self {
        let is_hovered = if self.enabled() {
            self.hovered()
        } else {
            self.contains_pointer()
        };
        let dismiss_id = self.id.with("tooltip_dismissed");
        let mut dismissed = self
            .ctx
            .data(|d| d.get_temp::<bool>(dismiss_id))
            .unwrap_or(false);

        let escape_pressed = self.ctx.input(|i| i.key_pressed(eframe::egui::Key::Escape));
        if escape_pressed && !dismissed && self.lost_focus() {
            self.request_focus();
            dismissed = true;
            self.ctx.data_mut(|d| d.insert_temp(dismiss_id, true));
        }

        let is_focused = self.has_focus() || self.ctx.memory(|mem| mem.has_focus(self.id));
        let is_active = is_hovered || is_focused;

        if is_active {
            if escape_pressed && !dismissed {
                dismissed = true;
                self.ctx.data_mut(|d| d.insert_temp(dismiss_id, true));
            }
        } else {
            if dismissed {
                self.ctx.data_mut(|d| d.insert_temp(dismiss_id, false));
            }
            return if self.enabled() {
                self.on_hover_text(text)
            } else {
                self.on_disabled_hover_text(text)
            };
        }

        if !dismissed {
            let text_into = text.into();
            let text_str = text_into.text().to_string();
            announce_status(&text_str);
            let active_slot = with_live_region(|r| r.active_slot);
            self.ctx.data_mut(|d| {
                d.insert_temp(eframe::egui::Id::new("aria_live_message"), text_str);
                d.insert_temp(eframe::egui::Id::new("aria_live_slot"), active_slot);
            });
            if self.enabled() {
                self.on_hover_text(text_into)
            } else {
                self.on_disabled_hover_text(text_into)
            }
        } else {
            self
        }
    }
}

#[allow(missing_docs)]
pub trait AccessibleTheoryHover {
    #[allow(missing_docs)]
    fn accessible_theory_hover(self, theory: &impl TheoryDescribable) -> Self;
}

impl AccessibleTheoryHover for Response {
    fn accessible_theory_hover(self, theory: &impl TheoryDescribable) -> Self {
        let is_hovered = if self.enabled() {
            self.hovered()
        } else {
            self.contains_pointer()
        };
        let dismiss_id = self.id.with("tooltip_dismissed");
        let mut dismissed = self
            .ctx
            .data(|d| d.get_temp::<bool>(dismiss_id))
            .unwrap_or(false);

        let escape_pressed = self.ctx.input(|i| i.key_pressed(eframe::egui::Key::Escape));
        if escape_pressed && !dismissed && self.lost_focus() {
            self.request_focus();
            dismissed = true;
            self.ctx.data_mut(|d| d.insert_temp(dismiss_id, true));
        }

        let is_focused = self.has_focus() || self.ctx.memory(|mem| mem.has_focus(self.id));
        let is_active = is_hovered || is_focused;

        if is_active {
            if escape_pressed && !dismissed {
                dismissed = true;
                self.ctx.data_mut(|d| d.insert_temp(dismiss_id, true));
            }
        } else {
            if dismissed {
                self.ctx.data_mut(|d| d.insert_temp(dismiss_id, false));
            }
            if self.ctx.memory(|mem| mem.everything_is_visible()) {
                let raw_text = theory.theory_description();
                return if self.enabled() {
                    self.on_hover_text(raw_text)
                } else {
                    self.on_disabled_hover_text(raw_text)
                };
            }
            return self;
        }

        if !dismissed {
            let raw_text = theory.theory_description();
            let speech_text = theory.phonetic_description();
            announce_status(&speech_text);
            let active_slot = with_live_region(|r| r.active_slot);
            self.ctx.data_mut(|d| {
                d.insert_temp(eframe::egui::Id::new("aria_live_message"), speech_text);
                d.insert_temp(eframe::egui::Id::new("aria_live_slot"), active_slot);
            });
            if self.enabled() {
                self.on_hover_text(raw_text)
            } else {
                self.on_disabled_hover_text(raw_text)
            }
        } else {
            self
        }
    }
}
// theory_verification!

#[cfg(test)]
#[path = "accessibility_tests.rs"]
mod tests;
