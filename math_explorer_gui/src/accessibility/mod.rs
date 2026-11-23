#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Accessibility subsystem: WAI-ARIA twin-slot live regions, `AccessKit` DOM reconciliation,
//! DOM action routing, opt-in `SpeechSynthesis` narration, and canvas keyboard/focus helpers.

#[cfg(target_arch = "wasm32")]
pub mod accesskit_dom;
#[cfg(target_arch = "wasm32")]
pub mod accesskit_events;
pub mod accesskit_model;
mod canvas_a11y;
mod hover;
mod live_region;
pub mod speech;

#[cfg(target_arch = "wasm32")]
pub use accesskit_dom::{
    clear_accesskit_dom_root, reconcile_accesskit_output, reconcile_tree_update,
    ACCESSKIT_DOM_ROOT_ID,
};
#[cfg(target_arch = "wasm32")]
pub use accesskit_events::attach_web_runner_accesskit;
pub use accesskit_model::{AccessKitTreeStore, CssBounds, DomNodeSpec};
pub use canvas_a11y::{
    lock_canvas_arrow_keys, paint_canvas_focus_ring, register_canvas_accessibility,
};
pub use hover::{AccessibleHoverText, AccessibleTheoryHover};
#[cfg(test)]
pub(crate) use live_region::{all_announcements, clear_test_announcements, next_live_slot};
pub use live_region::{
    announce_live_assertive_status, announce_live_status, announce_startup_ready_once,
    announce_status, announce_status_with_priority, ensure_plot_live_bridge,
    init_accessibility_bridge, sync_context_live_messages, with_live_region, TwinSlotLiveRegion,
    READY_ANNOUNCEMENT,
};
pub use speech::{
    is_speech_enabled, speak_if_enabled, speech_volume, stop_speech, sync_speech_settings,
    DEFAULT_SPEECH_VOLUME,
};

#[cfg(test)]
#[path = "../accessibility_tests.rs"]
mod tests;
