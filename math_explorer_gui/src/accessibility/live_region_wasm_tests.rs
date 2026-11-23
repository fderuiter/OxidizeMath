#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use wasm_bindgen_test::wasm_bindgen_test;

fn slot_pair_texts(doc: &web_sys::Document, prefix: &str) -> (String, String) {
    let slot0 = doc
        .get_element_by_id(&format!("{prefix}-slot-0"))
        .and_then(|el| el.text_content())
        .unwrap_or_default();
    let slot1 = doc
        .get_element_by_id(&format!("{prefix}-slot-1"))
        .and_then(|el| el.text_content())
        .unwrap_or_default();
    (slot0, slot1)
}

#[wasm_bindgen_test]
fn wasm_accessibility_bridge_creates_and_alternates_dom_live_regions() {
    let doc = web_sys::window()
        .and_then(|w| w.document())
        .expect("browser document should exist");

    if let Some(existing) = doc.get_element_by_id("aria-live-polite") {
        existing.remove();
    }
    if let Some(existing) = doc.get_element_by_id("aria-live-assertive") {
        existing.remove();
    }
    reset_startup_ready_flag();

    init_accessibility_bridge();

    let polite = doc
        .get_element_by_id("aria-live-polite")
        .expect("polite live container should be mounted");
    assert_eq!(polite.get_attribute("role").as_deref(), Some("status"));
    assert_eq!(polite.get_attribute("aria-live").as_deref(), Some("polite"));

    let assertive = doc
        .get_element_by_id("aria-live-assertive")
        .expect("assertive live container should be mounted");
    assert_eq!(assertive.get_attribute("role").as_deref(), Some("alert"));
    assert_eq!(
        assertive.get_attribute("aria-live").as_deref(),
        Some("assertive")
    );

    assert!(announce_startup_ready_once());
    assert!(!announce_startup_ready_once());
    let ready_slots = slot_pair_texts(&doc, "aria-live-polite");
    assert!(
        ready_slots == (READY_ANNOUNCEMENT.to_string(), String::new())
            || ready_slots == (String::new(), READY_ANNOUNCEMENT.to_string())
    );

    announce_status("Polite announcement B");
    let second_polite = slot_pair_texts(&doc, "aria-live-polite");
    if ready_slots.0.is_empty() {
        assert_eq!(
            second_polite,
            ("Polite announcement B".to_string(), String::new())
        );
    } else {
        assert_eq!(
            second_polite,
            (String::new(), "Polite announcement B".to_string())
        );
    }

    announce_status_with_priority("Critical startup error", "assertive");
    let assertive_slots = slot_pair_texts(&doc, "aria-live-assertive");
    assert!(
        assertive_slots == ("Critical startup error".to_string(), String::new())
            || assertive_slots == (String::new(), "Critical startup error".to_string())
    );
}
