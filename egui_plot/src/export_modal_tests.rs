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
