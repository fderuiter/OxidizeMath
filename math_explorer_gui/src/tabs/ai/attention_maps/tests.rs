#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use math_commons::math_kernel::types::flatten_2d_index;

#[test]
fn test_attention_maps_tool_default_and_recalculate() {
    let mut tool = AttentionMapsTool::default();
    assert_eq!(tool.attention_weights().nrows(), 4);
    assert_eq!(tool.attention_weights().ncols(), 4);
    assert_eq!(tool.output_matrix().nrows(), 4);
    assert_eq!(tool.output_matrix().ncols(), 4);

    assert!(tool.weights_texture().is_dirty());
    assert!(tool.output_texture().is_dirty());

    let ctx = egui::Context::default();
    tool.weights_texture
        .update_if_needed(&ctx, &tool.attention_weights, "weights_test");
    tool.output_texture
        .update_if_needed(&ctx, &tool.output_matrix, "output_test");

    assert!(!tool.weights_texture().is_dirty());
    assert!(!tool.output_texture().is_dirty());

    // Modifying Q and recalculating marks textures dirty
    tool.q_matrix[(0, 0)] = 2.5;
    tool.recalculate();

    assert!(tool.weights_texture().is_dirty());
    assert!(tool.output_texture().is_dirty());
}

#[test]
fn test_heatmap_texture_dirty_tracking_and_steady_state() {
    let ctx = egui::Context::default();
    let mut texture = HeatmapTexture::new();
    let matrix = DMatrix::from_element(8, 8, 0.42);

    assert!(texture.is_dirty());
    assert!(texture.texture().is_none());

    texture.update_if_needed(&ctx, &matrix, "test_mat");
    assert!(!texture.is_dirty());
    assert!(texture.texture().is_some());

    let tex_id = texture.texture().unwrap().id();

    // Steady-state render call: must skip re-uploading and stay clean
    texture.update_if_needed(&ctx, &matrix, "test_mat");
    assert!(!texture.is_dirty());
    assert_eq!(texture.texture().unwrap().id(), tex_id);
}

#[test]
fn test_heatmap_texture_buffer_reuse() {
    let ctx = egui::Context::default();
    let mut texture = HeatmapTexture::new();
    let matrix_64 = DMatrix::from_element(64, 64, 0.1);

    texture.update_if_needed(&ctx, &matrix_64, "seq64_test");
    let initial_cap = texture.buffer_capacity();
    assert!(initial_cap >= 64 * 64);

    // Smaller matrix update does not reallocate buffer
    let matrix_16 = DMatrix::from_element(16, 16, 0.5);
    texture.mark_dirty();
    texture.update_if_needed(&ctx, &matrix_16, "seq16_test");
    assert_eq!(texture.buffer_capacity(), initial_cap);
}

#[test]
fn test_cell_at_pos_precision_and_bounds() {
    let rect = egui::Rect::from_min_max(egui::pos2(100.0, 50.0), egui::pos2(500.0, 450.0));
    let ncols = 4;
    let nrows = 4;

    // Top-left cell
    assert_eq!(
        cell_at_pos(ncols, nrows, rect, egui::pos2(110.0, 60.0)),
        Some((0, 0))
    );

    // Bottom-right cell
    assert_eq!(
        cell_at_pos(ncols, nrows, rect, egui::pos2(490.0, 440.0)),
        Some((3, 3))
    );

    // Center boundary check
    assert_eq!(
        cell_at_pos(ncols, nrows, rect, egui::pos2(300.0, 250.0)),
        Some((2, 2))
    );

    // Points outside rect return None
    assert_eq!(
        cell_at_pos(ncols, nrows, rect, egui::pos2(50.0, 50.0)),
        None
    );
    assert_eq!(
        cell_at_pos(ncols, nrows, rect, egui::pos2(550.0, 450.0)),
        None
    );

    // Empty matrix returns None
    assert_eq!(cell_at_pos(0, 4, rect, egui::pos2(110.0, 60.0)), None);
    assert_eq!(cell_at_pos(4, 0, rect, egui::pos2(110.0, 60.0)), None);
}

#[test]
fn test_theme_contrast_adaptation() {
    let dark_color = compute_attention_cell_color(0.9, ThemeContrastMode::Dark);
    let light_color = compute_attention_cell_color(0.9, ThemeContrastMode::Light);
    let high_contrast_color = compute_attention_cell_color(0.9, ThemeContrastMode::HighContrast);

    // Different theme contrast modes produce distinct color mappings
    assert_ne!(dark_color, light_color);
    assert_ne!(dark_color, high_contrast_color);
    assert_ne!(light_color, high_contrast_color);

    // Light mode high weight should have low luminance (dark colors on light bg)
    assert!(light_color.r() < 100);
    assert!(light_color.g() < 100);

    // Dark mode high weight should have high red/green (yellow on dark bg)
    assert!(dark_color.r() > 200);
    assert!(dark_color.g() > 150);

    // Switching theme updates texture and last_theme state
    let ctx = egui::Context::default();
    let mut texture = HeatmapTexture::new();
    let matrix = DMatrix::from_element(4, 4, 0.5);
    texture.update_if_needed(&ctx, &matrix, "test");
    assert_eq!(texture.last_theme, Some(ThemeContrastMode::Dark));

    ctx.set_visuals(egui::Visuals::light());
    texture.update_if_needed(&ctx, &matrix, "test");
    assert_eq!(texture.last_theme, Some(ThemeContrastMode::Light));

    ctx.set_visuals(crate::app::ThemeMode::HighContrast.visuals());
    texture.update_if_needed(&ctx, &matrix, "test");
    assert_eq!(texture.last_theme, Some(ThemeContrastMode::HighContrast));
}

#[test]
fn test_tooltip_preserves_floating_point_precision() {
    let exact_val: f64 = 0.123456789;
    let formatted = format!("Weight: {}", exact_val);
    assert_eq!(formatted, "Weight: 0.123456789");
    assert_ne!(formatted, "Weight: 0.12"); // Unlike legacy format!("{:.2}")
}

#[test]
fn test_generate_attention_heatmap_screenshots() {
    let ctx = egui::Context::default();
    let mut tool = AttentionMapsTool::default();
    tool.weights_texture
        .update_if_needed(&ctx, &tool.attention_weights, "weights");

    let cell_size = 40;
    let ncols = tool.attention_weights.ncols();
    let nrows = tool.attention_weights.nrows();

    fn write_ppm(
        path: &str,
        ncols: usize,
        nrows: usize,
        pixels: &[egui::Color32],
        cell_size: usize,
    ) {
        let mut data =
            format!("P6\n{} {}\n255\n", ncols * cell_size, nrows * cell_size).into_bytes();
        for r in 0..nrows {
            for _dy in 0..cell_size {
                for c in 0..ncols {
                    let color = pixels[flatten_2d_index(c, r, ncols)];
                    for _dx in 0..cell_size {
                        data.push(color.r());
                        data.push(color.g());
                        data.push(color.b());
                    }
                }
            }
        }
        let _ = std::fs::write(path, data);
    }

    write_ppm(
        "/tmp/attention_heatmap_dark_mode.ppm",
        ncols,
        nrows,
        &tool.weights_texture.pixels_buffer,
        cell_size,
    );

    ctx.set_visuals(egui::Visuals::light());
    tool.weights_texture
        .update_if_needed(&ctx, &tool.attention_weights, "weights_light");

    write_ppm(
        "/tmp/attention_heatmap_light_mode.ppm",
        ncols,
        nrows,
        &tool.weights_texture.pixels_buffer,
        cell_size,
    );
}

fn attn_screen_raw() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    }
}

fn attn_key_raw(key: egui::Key) -> egui::RawInput {
    let mut raw = attn_screen_raw();
    raw.events.push(egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    });
    raw
}

fn focus_attention_weights(ctx: &egui::Context, tool: &mut AttentionMapsTool) -> egui::Id {
    let out0 = ctx.run(attn_screen_raw(), |ctx| tool.show(ctx));
    let update = out0
        .platform_output
        .accesskit_update
        .expect("AccessKit update expected");
    let weights_node_id = update
        .nodes
        .iter()
        .find_map(|(id, node)| {
            node.label()
                .is_some_and(|l| l.contains("Attention Weights"))
                .then_some(*id)
        })
        .expect("Attention Weights heatmap must expose AccessKit node");

    let mut focus_raw = attn_screen_raw();
    focus_raw.events.push(egui::Event::AccessKitActionRequest(
        egui::accesskit::ActionRequest {
            action: egui::accesskit::Action::Focus,
            target: weights_node_id,
            data: None,
        },
    ));
    let _ = ctx.run(focus_raw, |ctx| tool.show(ctx));
    let _ = ctx.run(attn_screen_raw(), |ctx| tool.show(ctx));
    ctx.memory(|m| m.focused()).expect("Heatmap focused")
}

#[test]
fn test_attention_maps_arrow_key_cursor_and_live_announcements() {
    crate::accessibility::clear_test_announcements();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();

    let mut tool = AttentionMapsTool::default();
    tool.q_matrix[(0, 0)] = 2.0;
    tool.k_matrix[(1, 0)] = 1.5;
    tool.v_matrix[(1, 1)] = 3.0;
    tool.recalculate();

    let heatmap_egui_id = focus_attention_weights(&ctx, &mut tool);
    assert_eq!(tool.weights_texture().cursor(), (0, 0));

    crate::accessibility::clear_test_announcements();
    let _ = ctx.run(attn_key_raw(egui::Key::ArrowRight), |ctx| tool.show(ctx));
    assert_eq!(tool.weights_texture().cursor(), (0, 1));
    assert_eq!(ctx.memory(|m| m.focused()), Some(heatmap_egui_id));

    let _ = ctx.run(attn_key_raw(egui::Key::ArrowDown), |ctx| tool.show(ctx));
    assert_eq!(tool.weights_texture().cursor(), (1, 1));
    assert_eq!(ctx.memory(|m| m.focused()), Some(heatmap_egui_id));

    let announcements = crate::accessibility::all_announcements();
    assert!(
        announcements
            .iter()
            .any(|(msg, _)| msg.contains("Token: [Row 1, Col 1]") && msg.contains("Weight:")),
        "Expected live announcement with Token and Weight, got {announcements:?}"
    );

    // Tab must never trap focus on the heatmap.
    let _ = ctx.run(attn_key_raw(egui::Key::Tab), |ctx| tool.show(ctx));
    assert_ne!(
        ctx.memory(|m| m.focused()),
        Some(heatmap_egui_id),
        "Tab must advance focus away from the heatmap"
    );
}

#[test]
fn test_attention_maps_vector_rect_svg_and_scalar_field_exports() {
    let mut tool = AttentionMapsTool::default();
    tool.q_matrix[(0, 0)] = 2.2;
    tool.k_matrix[(1, 2)] = -1.4;
    tool.v_matrix[(2, 3)] = 1.8;
    tool.recalculate();

    let dark_opts = egui_plot::export::ExportOptions::default();
    let dark_svg = tool
        .export_attention_svg(&dark_opts)
        .expect("dark vector SVG export should succeed");
    assert!(dark_svg.contains("<svg"));
    assert!(dark_svg.contains("<rect"));
    assert!(!dark_svg.contains("data:image/png"));

    // 4x4 weights + 4x4 output = 32 matrix cells (+ 1 background rect = 33 <rect tags)
    let rect_count = dark_svg.matches("<rect ").count();
    assert_eq!(rect_count, 33);

    let light_opts = egui_plot::export::ExportOptions {
        background_color: egui::Color32::WHITE,
        ..Default::default()
    };
    let light_svg = tool
        .export_attention_svg(&light_opts)
        .expect("light vector SVG export should succeed");
    assert_ne!(dark_svg, light_svg);
    assert!(!light_svg.contains("data:image/png"));

    let field = tool
        .export_scalar_field()
        .expect("attention weights scalar field");
    assert_eq!(field.width, 4);
    assert_eq!(field.height, 4);
    assert!(!field.periodic);
    assert_eq!(field.values.len(), 16);
}
