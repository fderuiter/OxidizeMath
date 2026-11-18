use eframe::egui::{Response, WidgetText};
use scientific_metadata::theory::TheoryDescribable;

/// Helper modeling the twin-slot ping-pong alternation logic.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TwinSlotLiveRegion {
    /// Active slot index (0 or 1).
    pub active_slot: usize,
    /// Text contents of slot 0 and slot 1.
    pub slots: [String; 2],
}

impl TwinSlotLiveRegion {
    /// Creates a new empty twin-slot live region.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Alternates to the sibling slot, clears the previous slot, and populates the new active slot.
    pub fn dispatch(&mut self, message: &str) {
        let next_slot = 1 - self.active_slot;
        self.slots[self.active_slot].clear();
        self.slots[next_slot] = message.to_string();
        self.active_slot = next_slot;
    }
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

#[cfg(any(not(target_arch = "wasm32"), test))]
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

    let slot_idx = LIVE_REGION.with(|region| {
        let mut r = region.borrow_mut();
        r.dispatch(message);
        r.active_slot
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

            let _ = (&ASSERTIVE_SLOT_TOGGLE, &POLITE_SLOT_TOGGLE);
            let sibling_idx = 1 - slot_idx;

            let active_id = format!("{container_id}-slot-{slot_idx}");
            let sibling_id = format!("{container_id}-slot-{sibling_idx}");

            // Clear the sibling slot
            if let Some(sibling_slot) = document.get_element_by_id(&sibling_id) {
                sibling_slot.set_text_content(Some(""));
            }

            // Populate the active slot
            if let Some(active_slot) = document.get_element_by_id(&active_id) {
                active_slot.set_text_content(Some(message));
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
static LAST_ANNOUNCED: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

#[cfg(not(target_arch = "wasm32"))]
#[allow(missing_docs)]
pub fn announce_status(message: &str) {
    announce_status_with_priority(message, "polite");
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(missing_docs)]
pub fn announce_status_with_priority(message: &str, priority: &str) {
    LIVE_REGION.with(|region| {
        region.borrow_mut().dispatch(message);
    });
    if let Ok(mut lock) = LAST_ANNOUNCED.lock() {
        *lock = Some(message.to_string());
    }
    TEST_ANNOUNCEMENTS.with(|a| {
        a.borrow_mut()
            .push((message.to_string(), priority.to_string()));
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(missing_docs)]
pub fn get_last_announced() -> Option<String> {
    LAST_ANNOUNCED.lock().ok().and_then(|lock| lock.clone())
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(missing_docs)]
pub fn clear_last_announced() {
    if let Ok(mut lock) = LAST_ANNOUNCED.lock() {
        *lock = None;
    }
    clear_test_announcements();
}

#[cfg(any(not(target_arch = "wasm32"), test))]
#[allow(dead_code, missing_docs)]
pub fn last_announcement() -> Option<(String, String)> {
    TEST_ANNOUNCEMENTS.with(|a| a.borrow().last().cloned())
}

#[cfg(any(not(target_arch = "wasm32"), test))]
#[allow(dead_code, missing_docs)]
pub fn all_announcements() -> Vec<(String, String)> {
    TEST_ANNOUNCEMENTS.with(|a| a.borrow().clone())
}

#[cfg(any(not(target_arch = "wasm32"), test))]
#[allow(dead_code, missing_docs)]
pub fn clear_test_announcements() {
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(mut lock) = LAST_ANNOUNCED.lock() {
        *lock = None;
    }
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
        let text_into = text.into();
        let text_str = text_into.text().to_string();

        let is_active = self.hovered() || self.has_focus();
        let dismiss_id = self.id.with("tooltip_dismissed");
        let mut dismissed = self
            .ctx
            .data(|d| d.get_temp::<bool>(dismiss_id))
            .unwrap_or(false);

        if is_active {
            if self.ctx.input(|i| i.key_pressed(eframe::egui::Key::Escape)) {
                dismissed = true;
                self.ctx.data_mut(|d| d.insert_temp(dismiss_id, true));
            }
        } else if dismissed {
            dismissed = false;
            self.ctx.data_mut(|d| d.insert_temp(dismiss_id, false));
        }

        if is_active && !dismissed {
            announce_status(&text_str);
            self.ctx.data_mut(|d| {
                d.insert_temp(eframe::egui::Id::new("aria_live_message"), text_str);
            });
        }

        if !dismissed {
            self.on_hover_text(text_into)
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
        let raw_text = theory.theory_description();
        let speech_text = theory.phonetic_description();

        let is_active = self.hovered() || self.has_focus();
        let dismiss_id = self.id.with("tooltip_dismissed");
        let mut dismissed = self
            .ctx
            .data(|d| d.get_temp::<bool>(dismiss_id))
            .unwrap_or(false);

        if is_active {
            if self.ctx.input(|i| i.key_pressed(eframe::egui::Key::Escape)) {
                dismissed = true;
                self.ctx.data_mut(|d| d.insert_temp(dismiss_id, true));
            }
        } else if dismissed {
            dismissed = false;
            self.ctx.data_mut(|d| d.insert_temp(dismiss_id, false));
        }

        if is_active && !dismissed {
            announce_status(&speech_text);
            self.ctx.data_mut(|d| {
                d.insert_temp(eframe::egui::Id::new("aria_live_message"), speech_text);
            });
        }

        if !dismissed {
            self.on_hover_text(raw_text)
        } else {
            self
        }
    }
}
// theory_verification!

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twin_slot_alternation() {
        let mut region = TwinSlotLiveRegion::new();
        assert_eq!(region.active_slot, 0);
        assert_eq!(region.slots[0], "");
        assert_eq!(region.slots[1], "");

        // Dispatch 1
        region.dispatch("First message");
        assert_eq!(region.active_slot, 1);
        assert_eq!(region.slots[1], "First message");
        assert_eq!(region.slots[0], "");

        // Dispatch 2 with identical message (prevents dropped announcements)
        region.dispatch("First message");
        assert_eq!(region.active_slot, 0);
        assert_eq!(region.slots[0], "First message");
        assert_eq!(region.slots[1], "");

        // Dispatch 3 with new message
        region.dispatch("Second message");
        assert_eq!(region.active_slot, 1);
        assert_eq!(region.slots[1], "Second message");
        assert_eq!(region.slots[0], "");
    }

    #[test]
    fn test_announcement_dispatch_on_native() {
        clear_test_announcements();
        announce_status("Simulation paused");
        announce_status_with_priority("Syntax Error: missing operand", "assertive");

        let announcements = all_announcements();
        assert_eq!(announcements.len(), 2);
        assert_eq!(
            announcements[0],
            ("Simulation paused".to_string(), "polite".to_string())
        );
        assert_eq!(
            announcements[1],
            (
                "Syntax Error: missing operand".to_string(),
                "assertive".to_string()
            )
        );

        #[cfg(not(target_arch = "wasm32"))]
        {
            assert_eq!(
                get_last_announced().as_deref(),
                Some("Syntax Error: missing operand")
            );
            clear_last_announced();
            assert_eq!(get_last_announced(), None);
            assert_eq!(all_announcements().len(), 0);
        }
    }
}
