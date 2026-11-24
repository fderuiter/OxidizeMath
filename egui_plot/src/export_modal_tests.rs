use super::*;
use crate::export::sample_progressive_frames;
use std::cell::Cell;

const MAX_SHAPE_NODES: usize = 8192;

fn rendered_texts(output: &egui::FullOutput) -> Vec<(String, egui::Pos2)> {
    let mut out = Vec::new();
    let mut stack: Vec<&egui::epaint::Shape> = output.shapes.iter().map(|c| &c.shape).collect();
    let mut visited = 0usize;

    while let Some(shape) = stack.pop() {
        if visited >= MAX_SHAPE_NODES {
            break;
        }
        visited += 1;
        match shape {
            egui::epaint::Shape::Text(text_shape) => {
                let center = text_shape.pos + text_shape.galley.rect.center().to_vec2();
                out.push((text_shape.galley.job.text.clone(), center));
            }
            egui::epaint::Shape::Vec(shapes) => {
                for inner in shapes.iter().rev() {
                    stack.push(inner);
                }
            }
            _ => {}
        }
    }
    out
}

fn find_text_center(output: &egui::FullOutput, needle: &str) -> Option<egui::Pos2> {
    rendered_texts(output)
        .into_iter()
        .find_map(|(text, center)| text.contains(needle).then_some(center))
}

fn screen_input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    }
}

fn click_input(pos: egui::Pos2) -> (egui::RawInput, egui::RawInput) {
    let mut down = screen_input();
    down.events = vec![
        egui::Event::PointerMoved(pos),
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let mut up = screen_input();
    up.events = vec![egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed: false,
        modifiers: egui::Modifiers::NONE,
    }];
    (down, up)
}

#[test]
fn test_export_modal_state_defaults() {
    let state = ExportModalState::default();
    assert!(!state.open);
    assert_eq!(state.options.width, 1920);
    assert_eq!(state.options.height, 1080);
    assert_eq!(state.options.scale, 1.0);
    assert_eq!(state.options.aspect_ratio, AspectRatioOption::SixteenNine);
    assert!(!state.options.log_x);
    assert!(!state.options.log_y);
    assert!(state.bg_dark);
    assert_eq!(state.gif_fps, 15);
    assert_eq!(state.gif_frames, 20);
    assert!(state.status_message.is_none());
}

#[test]
fn test_calculate_target_dimensions_and_max_cap() {
    let default_opts = ExportOptions::default();
    assert_eq!(calculate_target_dimensions(&default_opts), (1920, 1080));
    assert!(!exceeds_max_dimension(1920, 1080));

    let custom_free = ExportOptions {
        width: 1200,
        height: 900,
        scale: 2.0,
        aspect_ratio: AspectRatioOption::Free,
        ..Default::default()
    };
    assert_eq!(calculate_target_dimensions(&custom_free), (2400, 1800));

    let square_opts = ExportOptions {
        width: 1000,
        height: 600,
        scale: 1.0,
        aspect_ratio: AspectRatioOption::OneOne,
        ..Default::default()
    };
    assert_eq!(calculate_target_dimensions(&square_opts), (1000, 1000));

    let non_finite_scale = ExportOptions {
        width: 800,
        height: 600,
        scale: f32::NAN,
        aspect_ratio: AspectRatioOption::Free,
        ..Default::default()
    };
    assert_eq!(calculate_target_dimensions(&non_finite_scale), (800, 600));

    let oversized = ExportOptions {
        scale: 4.0,
        ..Default::default()
    };
    let (over_w, over_h) = calculate_target_dimensions(&oversized);
    assert_eq!((over_w, over_h), (7680, 4320));
    assert!(exceeds_max_dimension(over_w, over_h));
}

#[test]
fn test_poll_clipboard_status_scoped_and_when_closed() {
    let ctx = egui::Context::default();
    let id_a = egui::Id::new("plot_a");
    let id_b = egui::Id::new("plot_b");

    ctx.data_mut(|d| {
        d.insert_temp(
            clipboard_status_id(id_a),
            "Copied plot image to clipboard!".to_string(),
        );
        d.insert_temp(clipboard_status_id(id_b), "Other plot status".to_string());
    });

    let mut state_a = ExportModalState::default();
    let config_a = ExportModalConfig {
        window_title: "Export High-Res Plot - A",
        id: id_a,
        png_filename: "plot.png",
        svg_filename: "plot.svg",
        gif_filename: "plot.gif",
    };
    let mut source = CallbackExportSource {
        datasets_fn: Vec::new,
        sample_frames_fn: |_| Vec::new(),
    };

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        show_export_modal(ctx, state_a.bindings(), &config_a, &mut source);
    });

    assert!(!state_a.open);
    assert_eq!(
        state_a.status_message.as_deref(),
        Some("Copied plot image to clipboard!")
    );
    assert!(
        ctx.data(|d| d.get_temp::<String>(clipboard_status_id(id_a)))
            .is_none()
    );
    assert_eq!(
        ctx.data(|d| d.get_temp::<String>(clipboard_status_id(id_b)))
            .as_deref(),
        Some("Other plot status")
    );
}

#[test]
fn test_modal_header_gif_callback_and_close_button() {
    let ctx = egui::Context::default();
    let mut state = ExportModalState {
        open: true,
        options: ExportOptions {
            width: 320,
            height: 240,
            aspect_ratio: AspectRatioOption::Free,
            ..Default::default()
        },
        gif_frames: 6,
        ..Default::default()
    };
    let config = ExportModalConfig {
        window_title: "Export High-Res Plot - UnitTest",
        id: egui::Id::new("modal_unit_test"),
        png_filename: "plot.png",
        svg_filename: "plot.svg",
        gif_filename: "plot.gif",
    };
    let sampled_arg = Cell::new(0usize);
    let sample_data = vec![("Wave".to_string(), vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])];
    let run_frame = |raw: egui::RawInput, state: &mut ExportModalState| {
        let mut source = CallbackExportSource {
            datasets_fn: || sample_data.clone(),
            sample_frames_fn: |n| {
                sampled_arg.set(n);
                sample_progressive_frames(&sample_data, n)
            },
        };
        ctx.run(raw, |ctx| {
            show_export_modal(ctx, state.bindings(), &config, &mut source);
        })
    };

    let _ = run_frame(screen_input(), &mut state);
    let out1 = run_frame(screen_input(), &mut state);
    assert!(find_text_center(&out1, MODAL_HEADER_TEXT).is_some());
    assert!(find_text_center(&out1, "Target Canvas Size: 320 x 240 px").is_some());

    let gif_pos = find_text_center(&out1, "Save GIF").expect("Save GIF button rendered");
    let (down, up) = click_input(gif_pos);
    let _ = run_frame(down, &mut state);
    let _ = run_frame(up, &mut state);
    assert_eq!(sampled_arg.get(), 6);
    assert!(
        state.gif_job.is_active(),
        "Save GIF must start an asynchronous/incremental job without finishing inline"
    );
    assert_eq!(
        state.status_message.as_deref(),
        Some("Encoding GIF frame 0/6...")
    );

    for _ in 0..250 {
        if !state.gif_job.is_active() {
            break;
        }
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::sleep(std::time::Duration::from_millis(10));
        let _ = run_frame(screen_input(), &mut state);
    }
    assert!(!state.gif_job.is_active());
    assert_eq!(
        state.status_message.as_deref(),
        Some("GIF animation saved successfully.")
    );

    let out_settled = run_frame(screen_input(), &mut state);
    let close_pos = find_text_center(&out_settled, "Close").expect("Close button rendered");
    let (down_close, up_close) = click_input(close_pos);
    let _ = run_frame(down_close, &mut state);
    let _ = run_frame(up_close, &mut state);
    assert!(!state.open);
}

#[test]
fn test_plot_integration_opens_unified_modal_and_polls_clipboard() {
    let ctx = egui::Context::default();
    let plot_id = egui::Id::new("plot_unified_export_test");

    let show_plot = |ctx: &egui::Context| {
        egui::CentralPanel::default().show(ctx, |ui| {
            crate::Plot::new(plot_id).id(plot_id).show(ui, |plot_ui| {
                plot_ui.line(crate::Line::new("Series", vec![[1.0, 1.0], [10.0, 100.0]]));
            });
        });
    };

    let out1 = ctx.run(screen_input(), show_plot);
    let mut mem = crate::PlotMemory::load(&ctx, plot_id).expect("PlotMemory initialized");
    mem.log_x = true;
    mem.log_y = true;
    mem.store(&ctx, plot_id);

    let save_png_btn =
        find_text_center(&out1, "Save PNG").expect("Plot overlay Save PNG button must render");
    let (down, up) = click_input(save_png_btn);
    let _ = ctx.run(down, show_plot);
    let _ = ctx.run(up, show_plot);
    let out_open = ctx.run(screen_input(), show_plot);

    assert!(find_text_center(&out_open, MODAL_HEADER_TEXT).is_some());
    assert!(find_text_center(&out_open, "Target Canvas Size: 1920 x 1080 px").is_some());

    let svg_btn = find_text_center(&out_open, "Save SVG").expect("Modal Save SVG button");
    let (down_svg, up_svg) = click_input(svg_btn);
    let _ = ctx.run(down_svg, show_plot);
    let _ = ctx.run(up_svg, show_plot);
    let out_saved = ctx.run(screen_input(), show_plot);
    assert!(find_text_center(&out_saved, "SVG image saved successfully.").is_some());

    let close_pos = find_text_center(&out_saved, "Close").expect("Plot modal Close button");
    let (down_close, up_close) = click_input(close_pos);
    let _ = ctx.run(down_close, show_plot);
    let _ = ctx.run(up_close, show_plot);

    ctx.data_mut(|d| {
        d.insert_temp(
            clipboard_status_id(plot_id),
            "Async clipboard ready".to_string(),
        );
    });
    let _ = ctx.run(screen_input(), show_plot);
    assert!(
        ctx.data(|d| d.get_temp::<String>(clipboard_status_id(plot_id)))
            .is_none()
    );
}

fn srgb_luminance(c: egui::Color32) -> f64 {
    let ch = |v: u8| {
        let s = f64::from(v) / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * ch(c.r()) + 0.7152 * ch(c.g()) + 0.0722 * ch(c.b())
}

fn wcag_contrast(a: egui::Color32, b: egui::Color32) -> f64 {
    let (l1, l2) = (srgb_luminance(a), srgb_luminance(b));
    (l1.max(l2) + 0.05) / (l1.min(l2) + 0.05)
}

fn find_text_color(output: &egui::FullOutput, needle: &str) -> Option<egui::Color32> {
    let mut stack: Vec<&egui::epaint::Shape> = output.shapes.iter().map(|c| &c.shape).collect();
    while let Some(shape) = stack.pop() {
        match shape {
            egui::epaint::Shape::Text(t) if t.galley.job.text.contains(needle) => {
                return t
                    .galley
                    .job
                    .sections
                    .first()
                    .map(|s| s.format.color)
                    .or(Some(t.fallback_color));
            }
            egui::epaint::Shape::Vec(v) => stack.extend(v.iter()),
            _ => {}
        }
    }
    None
}

#[test]
fn test_export_modal_warning_and_status_colors_meet_wcag_aa() {
    for visuals in [egui::Visuals::dark(), egui::Visuals::light()] {
        let ctx = egui::Context::default();
        ctx.set_visuals(visuals.clone());
        let bg = visuals.window_fill();
        let mut state = ExportModalState {
            open: true,
            options: ExportOptions {
                scale: 4.0,
                ..Default::default()
            },
            status_message: Some("PNG image saved successfully.".to_string()),
            ..Default::default()
        };
        let config = ExportModalConfig {
            window_title: "Export Contrast Test",
            id: egui::Id::new(("export_contrast", visuals.dark_mode)),
            png_filename: "plot.png",
            svg_filename: "plot.svg",
            gif_filename: "plot.gif",
        };
        let mut source = CallbackExportSource {
            datasets_fn: Vec::new,
            sample_frames_fn: |_| Vec::new(),
        };
        let _ = ctx.run(screen_input(), |ctx| {
            show_export_modal(ctx, state.bindings(), &config, &mut source);
        });
        let out = ctx.run(screen_input(), |ctx| {
            show_export_modal(ctx, state.bindings(), &config, &mut source);
        });
        let warn_c = find_text_color(&out, "Capped at max dimension").expect("warn text");
        let ok_c = find_text_color(&out, "PNG image saved successfully.").expect("status text");
        assert!(wcag_contrast(warn_c, bg) >= 4.5, "warn contrast < 4.5");
        assert!(wcag_contrast(ok_c, bg) >= 4.5, "status contrast < 4.5");
    }
}

#[test]
fn test_export_modal_scalar_field_controls_and_intercept_actions() {
    struct ScalarStubSource {
        actions: Vec<ExportModalAction>,
    }
    impl ExportDataSource for ScalarStubSource {
        fn export_datasets(&mut self) -> Vec<ExportDataset> {
            Vec::new()
        }
        fn sample_gif_frames(&mut self, _num_frames: usize) -> Vec<DatasetFrame> {
            Vec::new()
        }
        fn scalar_field_capability(&self) -> (bool, &'static str) {
            (true, "")
        }
        fn intercept_action(&mut self, action: ExportModalAction) -> bool {
            self.actions.push(action);
            true
        }
    }

    let ctx = egui::Context::default();
    let mut state = ExportModalState {
        open: true,
        ..Default::default()
    };
    let config = ExportModalConfig {
        window_title: "Export Scalar Field Test",
        id: egui::Id::new("modal_scalar_test"),
        png_filename: "field.png",
        svg_filename: "field.svg",
        gif_filename: "field.gif",
    };
    let mut source = ScalarStubSource {
        actions: Vec::new(),
    };
    let _ = ctx.run(screen_input(), |ctx| {
        show_export_modal(ctx, state.bindings(), &config, &mut source);
    });
    let out = ctx.run(screen_input(), |ctx| {
        show_export_modal(ctx, state.bindings(), &config, &mut source);
    });
    assert!(find_text_center(&out, "Seamless 1:1 Tile").is_some());
    assert!(find_text_center(&out, "SVG Contour Levels:").is_some());
    assert!(find_text_center(&out, "Normal Map Strength:").is_some());

    let hm_pos =
        find_text_center(&out, "Save 16-bit Heightmap").expect("heightmap button rendered");
    let (down_hm, up_hm) = click_input(hm_pos);
    let _ = ctx.run(down_hm, |ctx| {
        show_export_modal(ctx, state.bindings(), &config, &mut source);
    });
    let _ = ctx.run(up_hm, |ctx| {
        show_export_modal(ctx, state.bindings(), &config, &mut source);
    });
    assert!(
        source
            .actions
            .contains(&ExportModalAction::SaveHeightmapPng)
    );

    let out2 = ctx.run(screen_input(), |ctx| {
        show_export_modal(ctx, state.bindings(), &config, &mut source);
    });
    let nm_pos = find_text_center(&out2, "Save Normal Map").expect("normal map button rendered");
    let (down_nm, up_nm) = click_input(nm_pos);
    let _ = ctx.run(down_nm, |ctx| {
        show_export_modal(ctx, state.bindings(), &config, &mut source);
    });
    let _ = ctx.run(up_nm, |ctx| {
        show_export_modal(ctx, state.bindings(), &config, &mut source);
    });
    assert!(
        source
            .actions
            .contains(&ExportModalAction::SaveNormalMapPng)
    );
}

#[test]
fn test_standalone_modal_gif_progress_duplicate_click_guard_and_cancel() {
    let ctx = egui::Context::default();
    let mut state = ExportModalState {
        open: true,
        options: ExportOptions {
            width: 1280,
            height: 720,
            ..Default::default()
        },
        gif_frames: 20,
        ..Default::default()
    };
    let config = ExportModalConfig {
        window_title: "Export Cancel Test",
        id: egui::Id::new("modal_cancel_test"),
        png_filename: "plot.png",
        svg_filename: "plot.svg",
        gif_filename: "plot.gif",
    };
    let sample_calls = Cell::new(0usize);
    let pts: Vec<[f64; 2]> = (0..500)
        .map(|i| [f64::from(i), (f64::from(i) * 0.1).sin()])
        .collect();
    let sample_data = vec![("DenseWave".to_string(), pts)];
    let run_frame = |raw: egui::RawInput, state: &mut ExportModalState| {
        let mut source = CallbackExportSource {
            datasets_fn: || sample_data.clone(),
            sample_frames_fn: |n| {
                sample_calls.set(sample_calls.get() + 1);
                sample_progressive_frames(&sample_data, n)
            },
        };
        ctx.run(raw, |ctx| {
            show_export_modal(ctx, state.bindings(), &config, &mut source);
        })
    };

    let _ = run_frame(screen_input(), &mut state);
    let out1 = run_frame(screen_input(), &mut state);
    let gif_pos = find_text_center(&out1, "Save GIF").expect("Save GIF button");
    let (down, up) = click_input(gif_pos);
    let _ = run_frame(down, &mut state);
    let _ = run_frame(up, &mut state);
    assert_eq!(sample_calls.get(), 1);
    assert!(state.gif_job.is_active());

    // Duplicate click on Save GIF while active must be ignored
    let (dup_down, dup_up) = click_input(gif_pos);
    let _ = run_frame(dup_down, &mut state);
    let out_active = run_frame(dup_up, &mut state);
    assert_eq!(
        sample_calls.get(),
        1,
        "Duplicate Save GIF click must be ignored"
    );
    assert!(find_text_center(&out_active, "GIF Progress:").is_some());

    let cancel_pos = find_text_center(&out_active, "Cancel GIF").expect("Cancel GIF button");
    let (c_down, c_up) = click_input(cancel_pos);
    let _ = run_frame(c_down, &mut state);
    let _ = run_frame(c_up, &mut state);
    assert!(!state.gif_job.is_active());
    assert_eq!(
        state.status_message.as_deref(),
        Some("GIF export canceled.")
    );
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn wasm_export_modal_uses_async_clipboard_initial_status() {
    assert_eq!(
        INITIAL_CLIPBOARD_STATUS,
        "Copying plot image to clipboard..."
    );
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn wasm_standalone_modal_encodes_at_most_one_gif_frame_per_repaint_tick() {
    let ctx = egui::Context::default();
    let mut state = ExportModalState {
        open: true,
        options: ExportOptions {
            width: 160,
            height: 120,
            aspect_ratio: AspectRatioOption::Free,
            ..Default::default()
        },
        gif_frames: 4,
        ..Default::default()
    };
    let config = ExportModalConfig {
        window_title: "WASM Incremental GIF",
        id: egui::Id::new("wasm_inc_gif"),
        png_filename: "p.png",
        svg_filename: "p.svg",
        gif_filename: "p.gif",
    };
    let data = vec![(
        "S".to_string(),
        vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 1.5]],
    )];
    let step = |raw: egui::RawInput, state: &mut ExportModalState| {
        let mut src = CallbackExportSource {
            datasets_fn: || data.clone(),
            sample_frames_fn: |n| sample_progressive_frames(&data, n),
        };
        ctx.run(raw, |ctx| {
            show_export_modal(ctx, state.bindings(), &config, &mut src)
        })
    };

    let _ = step(screen_input(), &mut state);
    let out = step(screen_input(), &mut state);
    let gif_pos = find_text_center(&out, "Save GIF").expect("Save GIF");
    let (down, up) = click_input(gif_pos);
    let _ = step(down, &mut state);
    let _ = step(up, &mut state);
    assert_eq!(state.gif_job.progress(), Some((0, 4)));

    let _ = step(screen_input(), &mut state);
    assert_eq!(state.gif_job.progress(), Some((1, 4)));
    let _ = step(screen_input(), &mut state);
    assert_eq!(state.gif_job.progress(), Some((2, 4)));
    let _ = step(screen_input(), &mut state);
    assert_eq!(state.gif_job.progress(), Some((3, 4)));
    let _ = step(screen_input(), &mut state);
    assert_eq!(state.gif_job.progress(), None);
    assert_eq!(
        state.status_message.as_deref(),
        Some("GIF animation saved successfully.")
    );
}
