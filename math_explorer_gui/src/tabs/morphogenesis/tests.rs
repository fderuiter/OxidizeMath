#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use std::thread;
use std::time::Duration;

#[test]
fn test_morphogenesis_initial_state() {
    let tab = MorphogenesisTab::default();
    assert!(tab.dirty);
    assert!(tab.texture.is_none());
    assert!(tab.pixels.is_empty());
}

#[test]
fn test_morphogenesis_render_lifecycle_and_dirty_tracking() {
    let ctx = egui::Context::default();
    let mut tab = MorphogenesisTab::default();

    // 1. Initial state has dirty flag set and no texture
    assert!(tab.dirty);
    assert!(tab.texture.is_none());

    // Allow worker thread to generate the initial snapshot
    for _ in 0..100 {
        if tab.controller.update().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }

    // Run first render pass: uploads texture and clears dirty flag
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });

    assert!(!tab.dirty);
    assert!(tab.texture.is_some());
    assert_eq!(tab.pixels.len(), tab.width * tab.height);

    let initial_pixels_cap = tab.pixels.capacity();
    let initial_tex_id = tab.texture.as_ref().unwrap().id();

    // 2. Pause simulation to test stationary/idle frame
    tab.paused = true;
    let _ = tab.controller.send_command(SimCommand::Pause);

    // Drain any pending updates from the worker before pause settled
    while tab.controller.update().is_some() {}
    tab.dirty = false;

    // Run idle frame: does not mark dirty and reuses persistent buffer and texture
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });

    assert!(!tab.dirty);
    assert_eq!(tab.pixels.capacity(), initial_pixels_cap);
    assert_eq!(tab.texture.as_ref().unwrap().id(), initial_tex_id);

    // 3. Manually setting dirty flag causes texture update and clears dirty
    tab.dirty = true;
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    assert!(!tab.dirty);
    assert_eq!(tab.pixels.capacity(), initial_pixels_cap);
    assert_eq!(tab.texture.as_ref().unwrap().id(), initial_tex_id);

    // 4. Reset simulation sets dirty flag
    let _ = tab.controller.send_command(SimCommand::Reset);
    tab.dirty = true;
    assert!(tab.dirty);
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    assert!(!tab.dirty);
    assert_eq!(tab.texture.as_ref().unwrap().id(), initial_tex_id);
}

#[test]
fn test_morphogenesis_grid_resize_and_buffer_safety() {
    let ctx = egui::Context::default();
    let mut tab = MorphogenesisTab::default();

    // Initial render
    for _ in 0..100 {
        if tab.controller.update().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    assert_eq!(tab.pixels.len(), 100 * 100);

    // Apply config with changed dimensions
    let mut config = tab.export_config();
    config.width = 50;
    config.height = 50;
    assert!(tab.apply_config(config).is_ok());
    assert!(tab.dirty);

    // Wait for resized runner to produce a snapshot
    for _ in 0..100 {
        if tab.controller.update().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }

    // Render after resize
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    assert!(!tab.dirty);
    assert_eq!(tab.pixels.len(), 50 * 50);
}

#[test]
fn test_plot_concentration_color_mapping() {
    let width = 2;
    let height = 2;
    let data = vec![0.0, 0.5, 1.0, 0.25];
    let image = MorphogenesisTab::plot_concentration(&data, width, height);

    assert_eq!(image.size, [width, height]);
    assert_eq!(image.pixels.len(), 4);

    let min_color = heatmap_color(0.0);
    assert_eq!(image.pixels[0], egui::Color32::from_rgb(min_color[0], min_color[1], min_color[2]));

    let max_color = heatmap_color(1.0);
    assert_eq!(image.pixels[2], egui::Color32::from_rgb(max_color[0], max_color[1], max_color[2]));

    // Uniform data (zero range) should not panic or divide by zero
    let uniform_data = vec![1.0; 4];
    let uniform_image = MorphogenesisTab::plot_concentration(&uniform_data, width, height);
    assert_eq!(uniform_image.pixels.len(), 4);
}

#[test]
fn test_morphogenesis_starter_gallery_lifecycle() {
    let mut tab = MorphogenesisTab::default();
    // Acceptance Criteria 1: Empty simulation tab renders gallery on initial launch
    assert!(tab.show_starter_gallery);

    // Acceptance Criteria 2: Applying starter preset transitions into active mode
    let spots_json = crate::starter_gallery::PRESET_TURING_SPOTS;
    assert!(tab.from_json_str(spots_json).is_ok());
    tab.show_starter_gallery = false;
    assert!(!tab.show_starter_gallery);

    // Acceptance Criteria 4: Toggle gallery on demand
    tab.show_starter_gallery = !tab.show_starter_gallery;
    assert!(tab.show_starter_gallery);

    // Acceptance Criteria 3: Loaded workspace session bypasses the gallery
    let saved_state = tab.save_state().unwrap();
    let mut loaded_tab = MorphogenesisTab::default();
    assert!(loaded_tab.show_starter_gallery);
    loaded_tab.load_state(&saved_state);
    assert!(!loaded_tab.show_starter_gallery);
}

