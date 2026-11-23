#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::presets::{PRESET_PHYSARUM_MAZE, PRESET_PHYSARUM_STEINER, PRESET_PHYSARUM_TOKYO_RAIL};
use super::*;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool, SimulationFramework};
use crate::tabs::morphogenesis::MorphogenesisTab;
use crate::tabs::ExplorerTab;
use crate::url_state::MAX_DECODED_URL_STATE_BYTES;
#[cfg(not(target_arch = "wasm32"))]
use math_commons::math_kernel::types::flatten_2d_index;

fn screen_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        focused: true,
        events,
        ..Default::default()
    }
}

fn key_press(key: egui::Key) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }
}

fn primary_click_events(target: egui::Pos2) -> Vec<egui::Event> {
    vec![
        egui::Event::PointerMoved(target),
        egui::Event::PointerButton {
            pos: target,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::PointerButton {
            pos: target,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        },
    ]
}

#[test]
fn test_morphogenesis_tab_switches_across_turing_lenia_and_physarum_tools() {
    let mut tab = MorphogenesisTab::default();
    assert_eq!(tab.active_tool_id(), Some("MorphogenesisTool"));

    assert!(tab.select_tool_by_name("LeniaTool"));
    assert_eq!(tab.active_tool_id(), Some("LeniaTool"));

    assert!(tab.select_tool_by_name("PhysarumTool"));
    assert_eq!(tab.active_tool_id(), Some("PhysarumTool"));

    for (title, raw) in [
        ("Physarum Steiner Y-Junction", PRESET_PHYSARUM_STEINER),
        ("Physarum Shortest-Path Maze", PRESET_PHYSARUM_MAZE),
        ("Physarum Tokyo-Rail Network", PRESET_PHYSARUM_TOKYO_RAIL),
    ] {
        tab.framework
            .apply_starter_card(title, Some("PhysarumTool"), Some(raw))
            .expect("Physarum starter card must apply cleanly");
        let phys = tab
            .framework
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut())
            .and_then(|a| a.downcast_mut::<PhysarumTool>())
            .expect("active tool should be PhysarumTool");
        phys.step_once();
        assert!(FiniteStateGuard::is_finite_state(phys));
        assert!(phys.engine.telemetry().total_trail_mass > 1.0);
    }

    // Verify split-viewport / secondary framework instance runs alongside Lenia/Turing cleanly
    let mut right_pane = MorphogenesisTab::default();
    assert!(right_pane.select_tool_by_name("LeniaTool"));
    assert_eq!(tab.active_tool_id(), Some("PhysarumTool"));
    assert_eq!(right_pane.active_tool_id(), Some("LeniaTool"));
}

fn physarum_canvas_point(out: &egui::FullOutput, frac: f32) -> egui::Pos2 {
    let ak = out
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit update");
    let bounds = ak
        .nodes
        .iter()
        .find(|(_, n)| {
            n.label()
                .is_some_and(|l| l.contains("Physarum Network Lab Chemotactic Canvas"))
        })
        .and_then(|(_, n)| n.bounds())
        .expect("Physarum canvas bounds");
    egui::pos2(
        (bounds.x0 + (bounds.x1 - bounds.x0) * f64::from(frac)) as f32,
        (bounds.y0 + (bounds.y1 - bounds.y0) * f64::from(frac)) as f32,
    )
}

#[test]
fn test_physarum_mouse_city_wall_sever_brushes_and_keyboard_accessibility() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = PhysarumTool {
        brush_mode: PhysarumBrushMode::PlaceCity,
        ..Default::default()
    };
    let initial_cities = tool.engine.terminal_count();

    let out = ctx.run(screen_input(Vec::new()), |ctx| tool.show(ctx));
    let target = physarum_canvas_point(&out, 0.25);

    let _ = ctx.run(screen_input(primary_click_events(target)), |ctx| {
        tool.show(ctx)
    });
    assert_eq!(tool.engine.terminal_count(), initial_cities + 1);
    assert!(tool.focused_canvas_id().is_some());

    // Keyboard wall painting (W) and tube severing (Minus)
    let key_events = vec![
        key_press(egui::Key::ArrowRight),
        key_press(egui::Key::ArrowDown),
        key_press(egui::Key::W),
    ];
    let _ = ctx.run(screen_input(key_events), |ctx| tool.show(ctx));
    assert!(!tool.obstacle_strokes.is_empty());
    if let Some(cid) = tool
        .focused_canvas_id()
        .or_else(|| ctx.memory(|m| m.focused()))
    {
        ctx.memory_mut(|m| m.request_focus(cid));
    }

    let mass_before_sever = tool.engine.telemetry().total_trail_mass;
    tool.cursor = [tool.grid_size / 2, tool.grid_size / 2];
    let _ = ctx.run(screen_input(vec![key_press(egui::Key::Minus)]), |ctx| {
        tool.show(ctx)
    });
    let mass_after_sever = tool.engine.telemetry().total_trail_mass;
    assert!(
        mass_after_sever < mass_before_sever,
        "Expected keyboard sever to reduce trail mass: after={mass_after_sever:.2}, before={mass_before_sever:.2}, focused={:?}",
        tool.focused_canvas_id()
    );
}

#[test]
fn test_morphogenesis_tab_propagates_physarum_focused_canvas_id_and_modal_guard() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tab = MorphogenesisTab::default();
    assert!(tab.select_tool_by_name("PhysarumTool"));
    assert!(tab.focused_canvas_id().is_none());

    let out = ctx.run(screen_input(Vec::new()), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            tab.framework.show_inside(ui, "biology");
        });
    });
    let target = physarum_canvas_point(&out, 0.5);
    let _ = ctx.run(screen_input(primary_click_events(target)), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            tab.framework.show_inside(ui, "biology");
        });
    });
    let focused = tab.focused_canvas_id().expect("focused Physarum canvas");
    let rev_before_modal = tab.state_revision();
    ctx.memory_mut(|m| m.request_focus(focused));
    ctx.data_mut(|d| d.insert_temp(egui::Id::new("SHELL_MODAL_OR_SASH_FOCUSED"), true));
    let _ = ctx.run(
        screen_input(vec![key_press(egui::Key::C), key_press(egui::Key::W)]),
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                tab.framework.show_inside(ui, "biology");
            });
        },
    );
    ctx.data_mut(|d| d.remove::<bool>(egui::Id::new("SHELL_MODAL_OR_SASH_FOCUSED")));
    assert_eq!(tab.state_revision(), rev_before_modal);
}

#[test]
fn test_physarum_finite_state_guard_divergence_and_recovery() {
    let mut tool = PhysarumTool {
        step_size: 9.0,
        sensor_angle: -0.5,
        playing: true,
        ..Default::default()
    };
    tool.step_once();
    assert!(FiniteStateGuard::diverged(&tool));
    assert!(!tool.playing);
    assert!(tool.diagnostic_report().is_some());

    FiniteStateGuard::auto_scale_step_size(&mut tool);
    assert!(FiniteStateGuard::is_finite_state(&tool));
    assert!((0.20..=1.75).contains(&tool.step_size));
    assert!((0.10..=1.50).contains(&tool.sensor_angle));
}

#[test]
fn test_physarum_exports_png_gif_csv_json_and_renders_screenshots() {
    let ctx = egui::Context::default();
    let mut framework = SimulationFramework::new("biology");
    framework.export_options.width = 320;
    framework.export_options.height = 240;
    framework.export_gif_frames = 4;
    assert!(framework.select_tool_by_name("PhysarumTool"));
    let fast_steiner = PRESET_PHYSARUM_STEINER
        .replace("\"grid_size\": 128", "\"grid_size\": 64")
        .replace("\"agent_count\": 6144", "\"agent_count\": 1024");
    framework
        .apply_starter_card(
            "Physarum Steiner Y-Junction",
            Some("PhysarumTool"),
            Some(&fast_steiner),
        )
        .expect("Steiner 64x64 starter");

    for fmt in [
        ExportFormat::Png,
        ExportFormat::Gif,
        ExportFormat::Csv,
        ExportFormat::Json,
    ] {
        let _ = ctx.run(screen_input(Vec::new()), |ctx| {
            framework.show_export_modal = true;
            framework.queue_export(fmt);
            framework.show(ctx, "biology");
        });
        if fmt == ExportFormat::Gif {
            for _ in 0..250 {
                if framework.active_gif_progress().is_none() {
                    break;
                }
                #[cfg(not(target_arch = "wasm32"))]
                std::thread::sleep(std::time::Duration::from_millis(10));
                let _ = ctx.run(screen_input(Vec::new()), |ctx| {
                    framework.show(ctx, "biology")
                });
            }
            let bytes = framework.export_queue.last_exported_bytes.as_deref();
            assert!(bytes.is_some_and(|b| b.starts_with(b"GIF89a")));
        }
        assert_eq!(
            framework.export_status_severity,
            ExportStatusSeverity::Success,
            "Export {fmt:?} failed: {:?}",
            framework.export_status_message
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var_os("WRITE_PHYSARUM_SCREENSHOTS").is_some() {
        write_verification_screenshots();
    }
}

#[test]
fn test_physarum_compact_url_share_roundtrip_and_bounds_rejection() {
    let mut framework = SimulationFramework::new("biology");
    assert!(framework.select_tool_by_name("PhysarumTool"));
    framework
        .apply_starter_card(
            "Physarum Tokyo-Rail Network",
            Some("PhysarumTool"),
            Some(PRESET_PHYSARUM_TOKYO_RAIL),
        )
        .expect("Tokyo-Rail starter");

    let preset_json = framework
        .export_url_preset_json()
        .expect("URL preset export")
        .expect("non-empty URL preset");
    assert!(
        preset_json.len() < MAX_DECODED_URL_STATE_BYTES,
        "Physarum URL JSON {} bytes must stay < 4 KiB",
        preset_json.len()
    );
    let share_url = framework.encode_share_url().expect("encode share URL");
    let hash_idx = share_url.find('#').expect("fragment separator");
    let fragment = &share_url[hash_idx..];

    let mut app = crate::app::MathExplorerApp::default();
    assert!(app
        .apply_share_location_strings(fragment, "", None)
        .unwrap());
    let bio_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Morphogenesis")
        .unwrap();
    assert_eq!(app.selected_tab, bio_idx);
    assert_eq!(app.tabs[bio_idx].active_tool_id(), Some("PhysarumTool"));
    assert_eq!(
        app.tabs[bio_idx].export_url_preset_json().unwrap(),
        Some(preset_json)
    );

    // Reject out-of-bounds grid_size and agent_count
    let mut tool = PhysarumTool::default();
    for idx in 0..=super::presets::MAX_SAVED_OBSTACLE_STROKES {
        tool.apply_wall_stroke(8 + idx * 3, 12 + idx * 2, true);
    }
    let rev_at_cap = tool.url_state_revision();
    tool.apply_wall_stroke(96, 96, false);
    assert_ne!(tool.url_state_revision(), rev_at_cap);

    let bad_grid = tool
        .save_state()
        .unwrap()
        .replace("\"grid_size\":128", "\"grid_size\":512");
    assert!(tool.try_load_state(&bad_grid).is_err());
    let bad_agents = tool
        .save_state()
        .unwrap()
        .replace("\"agent_count\":6144", "\"agent_count\":65536");
    assert!(tool.try_load_state(&bad_agents).is_err());
}

#[cfg(not(target_arch = "wasm32"))]
fn write_verification_screenshots() {
    let specs = [
        (
            PhysarumPreset::SteinerTriangle,
            340usize,
            "/tmp/physarum_steiner_y_junction.png",
        ),
        (
            PhysarumPreset::AsymmetricMaze,
            400usize,
            "/tmp/physarum_asymmetric_maze.png",
        ),
        (
            PhysarumPreset::TokyoRail,
            240usize,
            "/tmp/physarum_tokyo_rail.png",
        ),
    ];
    for (preset, steps, path) in specs {
        let mut tool = PhysarumTool::default();
        tool.apply_preset(preset);
        for _ in 0..steps {
            tool.step_once();
        }
        tool.refresh_benchmark_and_record();
        tool.refresh_pixels_in_place();
        save_annotated_tool_png(&tool, 4, path);
        if preset == PhysarumPreset::SteinerTriangle {
            if let Ok(plot_png) = egui_plot::export::export_custom_png(
                &tool.export_datasets(),
                &egui_plot::export::ExportOptions::default(),
            ) {
                let _ = std::fs::write("/tmp/physarum_benchmark_telemetry.png", plot_png);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn save_annotated_tool_png(tool: &PhysarumTool, scale: usize, out_path: &str) {
    let w = tool.engine.width();
    let h = tool.engine.height();
    let sw = w * scale;
    let sh = h * scale;
    let mut rgba = vec![0u8; sw * sh * 4];
    for y in 0..sh {
        let gy = y / scale;
        for x in 0..sw {
            let gx = x / scale;
            let c = tool.pixels[flatten_2d_index(gx, gy, w)];
            let base = flatten_2d_index(x, y, sw) * 4;
            rgba[base] = c.r();
            rgba[base + 1] = c.g();
            rgba[base + 2] = c.b();
            rgba[base + 3] = 255;
        }
    }
    for &(p0, p1) in &tool.latest_benchmark.mst_segments {
        draw_line_rgba(&mut rgba, sw, sh, p0, p1, scale, [132, 140, 152, 220]);
    }
    for edge in &tool.latest_benchmark.edges {
        draw_line_rgba(
            &mut rgba,
            sw,
            sh,
            edge.p0,
            edge.p1,
            scale,
            [74, 214, 122, 255],
        );
    }
    for &junc in &tool.latest_benchmark.steiner_junctions {
        draw_disc_rgba(&mut rgba, sw, sh, junc, scale, 6, [245, 194, 66, 255]);
    }
    for term in tool.engine.terminals() {
        draw_disc_rgba(
            &mut rgba,
            sw,
            sh,
            [term.x, term.y],
            scale,
            6,
            [250, 204, 56, 255],
        );
    }
    let mut buf = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut buf, sw as u32, sh as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        if let Ok(mut writer) = enc.write_header() {
            let _ = writer.write_image_data(&rgba);
        }
    }
    let _ = std::fs::write(out_path, buf);
}

#[cfg(not(target_arch = "wasm32"))]
fn draw_line_rgba(
    rgba: &mut [u8],
    sw: usize,
    sh: usize,
    p0: [f64; 2],
    p1: [f64; 2],
    scale: usize,
    col: [u8; 4],
) {
    let x0 = (p0[0] + 0.5) * (scale as f64);
    let y0 = (p0[1] + 0.5) * (scale as f64);
    let x1 = (p1[0] + 0.5) * (scale as f64);
    let y1 = (p1[1] + 0.5) * (scale as f64);
    let steps = ((x1 - x0).hypot(y1 - y0).ceil() as usize).max(1);
    for s in 0..=steps {
        let t = (s as f64) / (steps as f64);
        let px = (x0 + t * (x1 - x0)).round() as isize;
        let py = (y0 + t * (y1 - y0)).round() as isize;
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                let nx = px + dx;
                let ny = py + dy;
                if nx >= 0 && nx < sw as isize && ny >= 0 && ny < sh as isize {
                    let base = flatten_2d_index(nx as usize, ny as usize, sw) * 4;
                    rgba[base..base + 4].copy_from_slice(&col);
                }
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn draw_disc_rgba(
    rgba: &mut [u8],
    sw: usize,
    sh: usize,
    center: [f64; 2],
    scale: usize,
    rad_px: isize,
    col: [u8; 4],
) {
    let cx = ((center[0] + 0.5) * (scale as f64)).round() as isize;
    let cy = ((center[1] + 0.5) * (scale as f64)).round() as isize;
    for dy in -rad_px..=rad_px {
        for dx in -rad_px..=rad_px {
            let d2 = dx * dx + dy * dy;
            if d2 <= rad_px * rad_px {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx >= 0 && nx < sw as isize && ny >= 0 && ny < sh as isize {
                    let base = flatten_2d_index(nx as usize, ny as usize, sw) * 4;
                    rgba[base..base + 4].copy_from_slice(&col);
                }
            }
        }
    }
}
