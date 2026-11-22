//! Browser `wasm32` integration tests for [`super::MriKSpaceScannerTool`].

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{MriKSpaceScannerTool, KSPACE_CENTER, KSPACE_DIM, KSPACE_LEN};
use crate::accessibility::init_accessibility_bridge;
use crate::framework::InteractiveTool;
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use wasm_bindgen_test::wasm_bindgen_test;

fn wasm_input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1200.0, 820.0),
        )),
        ..Default::default()
    }
}

fn find_kspace_canvas_id(output: &egui::FullOutput) -> Option<egui::Id> {
    let update = output.platform_output.accesskit_update.as_ref()?;
    for (nid, node) in &update.nodes {
        if node.label() == Some("Interactive 2D k-Space Frequency Canvas") {
            return Some(egui::Id::new(nid.0));
        }
    }
    None
}

#[wasm_bindgen_test]
fn wasm_mri_kspace_full_reconstruction_and_lorenz_preset() {
    let mut tool = MriKSpaceScannerTool::default();
    assert_eq!(tool.recon.sampled_bins, KSPACE_LEN);
    assert!(tool.recon.normalized_l2_error < 1e-5);

    tool.import_preset_json(crate::starter_gallery::PRESET_MRI_KSPACE_LORENZ_CS)
        .expect("Lorenz compressed sensing preset should import on wasm32");
    assert!(tool.recon.sampled_bins > 0 && tool.recon.sampled_bins < KSPACE_LEN);
    assert!(tool.export_raster_image().is_some());
}

#[wasm_bindgen_test]
fn wasm_mri_kspace_keyboard_reticle_updates_dom_aria_live_polite() {
    let doc = web_sys::window()
        .and_then(|w| w.document())
        .expect("browser document should exist");
    init_accessibility_bridge();

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = MriKSpaceScannerTool::default();

    let out = ctx.run(wasm_input(), |ctx| tool.show(ctx));
    assert!(find_kspace_canvas_id(&out).is_some());
    let canvas_id = tool.kspace_canvas_id.expect("kspace_canvas_id recorded");
    ctx.memory_mut(|m| m.request_focus(canvas_id));
    let _ = ctx.run(wasm_input(), |ctx| tool.show(ctx));

    let mut key_in = wasm_input();
    key_in.events = vec![
        egui::Event::Key {
            key: egui::Key::ArrowDown,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = ctx.run(key_in, |ctx| tool.show(ctx));

    assert_eq!(tool.reticle_row, KSPACE_CENTER + 1);
    assert!(!tool.active_mask[flatten_2d_index(KSPACE_CENTER, KSPACE_CENTER + 1, KSPACE_DIM)]);

    let s0 = doc
        .get_element_by_id("aria-live-polite-slot-0")
        .and_then(|e| e.text_content())
        .unwrap_or_default();
    let s1 = doc
        .get_element_by_id("aria-live-polite-slot-1")
        .and_then(|e| e.text_content())
        .unwrap_or_default();
    let combined = format!("{s0}{s1}");
    assert!(
        combined.contains("k-space bin") && combined.contains("PSNR"),
        "Expected polite ARIA live region to announce toggled k-space bin and PSNR, got '{combined}'"
    );
}
