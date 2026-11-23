//! Acceptance tests for the docked plot header, corner overlay insets, axis/hover separation,
//! pre-legend curve fit injection, failed-fit diagnostic banner, and scientific/logarithmic formatting.

use egui::{Context, Event, Id, PointerButton, Pos2, RawInput, Rect, accesskit::Role, pos2, vec2};

use crate::{
    AccessState, Corner, Legend, Line, Plot, PlotMemory,
    a11y::format_critical_point_live_summary,
    fit::FitType,
    format::format_axis_tick,
    items::{SavedCallout, format_value_tooltip_content},
    poi::{CriticalPoint, CriticalPointKind},
};

fn raw_screen(size: egui::Vec2, events: Vec<Event>) -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, size)),
        events,
        ..Default::default()
    }
}

fn click_events(pos: Pos2, button: PointerButton) -> [Vec<Event>; 2] {
    [
        vec![
            Event::PointerMoved(pos),
            Event::PointerButton {
                pos,
                button,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        vec![Event::PointerButton {
            pos,
            button,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    ]
}

fn collect_rendered_texts(shapes: &[egui::epaint::ClippedShape]) -> Vec<(String, Rect)> {
    fn walk(s: &egui::epaint::Shape, out: &mut Vec<(String, Rect)>) {
        match s {
            egui::epaint::Shape::Text(t) => {
                out.push((
                    t.galley.text().to_owned(),
                    Rect::from_min_size(t.pos, t.galley.size()),
                ));
            }
            egui::epaint::Shape::Vec(v) => v.iter().for_each(|inner| walk(inner, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    for cs in shapes {
        walk(&cs.shape, &mut out);
    }
    out
}

fn ak_nodes_by_role(out: &egui::FullOutput, role: Role) -> Vec<(String, Rect)> {
    let Some(ak) = out.platform_output.accesskit_update.as_ref() else {
        return Vec::new();
    };
    ak.nodes
        .iter()
        .filter(|(_, n)| n.role() == role)
        .filter_map(|(_, n)| {
            let b = n.bounds()?;
            let rect = Rect::from_min_max(
                pos2(b.x0 as f32, b.y0 as f32),
                pos2(b.x1 as f32, b.y1 as f32),
            );
            Some((n.label().unwrap_or_default().to_owned(), rect))
        })
        .collect()
}

fn overlap_area(a: Rect, b: Rect) -> f32 {
    let inter = a.intersect(b);
    if inter.is_positive() {
        inter.area()
    } else {
        0.0
    }
}

#[test]
fn test_docked_header_collapses_and_context_menu_restores() {
    let ctx = Context::default();
    ctx.enable_accesskit();
    let plot_id = Id::new("docked_collapse_plot");
    let show = |c: &Context| -> Rect {
        let mut frame_rect = Rect::NOTHING;
        egui::CentralPanel::default().show(c, |ui| {
            let resp = Plot::new(plot_id)
                .id(plot_id)
                .legend(Legend::default())
                .show(ui, |p| {
                    p.line(Line::new("Wave", vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]));
                });
            frame_rect = *resp.transform.frame();
        });
        frame_rect
    };

    let mut expanded_frame = Rect::NOTHING;
    let out1 = ctx.run(raw_screen(vec2(800.0, 600.0), Vec::new()), |c| {
        expanded_frame = show(c);
    });
    let buttons = ak_nodes_by_role(&out1, Role::Button);
    let (_, collapse_rect) = buttons
        .iter()
        .find(|(lbl, _)| lbl == "Collapse plot toolbar")
        .expect("Collapse plot toolbar button must render in docked header");
    assert!(collapse_rect.max.y <= expanded_frame.min.y + 0.5);
    assert_eq!(overlap_area(*collapse_rect, expanded_frame), 0.0);

    for evs in click_events(collapse_rect.center(), PointerButton::Primary) {
        let _ = ctx.run(raw_screen(vec2(800.0, 600.0), evs), |c| {
            let _ = show(c);
        });
    }
    let mut collapsed_frame = Rect::NOTHING;
    let out_collapsed = ctx.run(raw_screen(vec2(800.0, 600.0), Vec::new()), |c| {
        collapsed_frame = show(c);
    });
    assert!(
        (expanded_frame.min.y - collapsed_frame.min.y - 30.0).abs() < 1.0,
        "Collapsing docked header must reclaim the 30 px row ({expanded_frame:?} vs {collapsed_frame:?})"
    );
    assert!(
        !ak_nodes_by_role(&out_collapsed, Role::Button)
            .iter()
            .any(|(l, _)| l == "Collapse plot toolbar")
    );

    let [down, up] = click_events(collapsed_frame.center(), PointerButton::Secondary);
    let _ = ctx.run(raw_screen(vec2(800.0, 600.0), down), |c| {
        let _ = show(c);
    });
    let _ = ctx.run(raw_screen(vec2(800.0, 600.0), up), |c| {
        let _ = show(c);
    });
    let out_menu = ctx.run(raw_screen(vec2(800.0, 600.0), Vec::new()), |c| {
        let _ = show(c);
    });
    let menu_texts: Vec<String> = collect_rendered_texts(&out_menu.shapes)
        .into_iter()
        .map(|(t, _)| t)
        .collect();
    assert!(menu_texts.iter().any(|t| t.contains("Show Toolbar")));
    assert!(menu_texts.iter().any(|t| t.contains("Save PNG")));
    assert!(menu_texts.iter().any(|t| t.contains("Fit Curve")));
}

#[test]
fn test_show_toolbar_and_settings_button_flags_suppress_2d_chrome() {
    let ctx = Context::default();
    ctx.enable_accesskit();
    let out = ctx.run(raw_screen(vec2(800.0, 600.0), Vec::new()), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            Plot::new("suppressed_3d_like_plot")
                .show_toolbar(false)
                .show_settings_button(false)
                .show_navigation_controls(false)
                .show(ui, |p| {
                    p.line(Line::new("Orbit", vec![[-1.0, -1.0], [1.0, 1.0]]));
                });
        });
    });
    let btn_labels: Vec<String> = ak_nodes_by_role(&out, Role::Button)
        .into_iter()
        .map(|(l, _)| l)
        .collect();
    assert!(
        btn_labels.is_empty(),
        "3D viewport flags must suppress all 2D plot toolbar, wrench, and zoom buttons; got {btn_labels:?}"
    );
}

#[test]
fn test_legend_and_fit_badge_zero_overlap_across_corners_and_narrow_reflow() {
    let corners = [
        Corner::RightTop,
        Corner::LeftTop,
        Corner::LeftBottom,
        Corner::RightBottom,
    ];
    for size in [vec2(800.0, 600.0), vec2(320.0, 480.0)] {
        for corner in corners {
            let ctx = Context::default();
            ctx.enable_accesskit();
            let plot_id = Id::new(("corner_fit_check", corner as u8, size.x as i32));
            let st = AccessState {
                fit_model: Some(FitType::Quartic),
                ..Default::default()
            };
            ctx.data_mut(|d| d.insert_temp(plot_id, st));

            let run = |c: &Context| {
                egui::CentralPanel::default().show(c, |ui| {
                    Plot::new(plot_id)
                        .id(plot_id)
                        .legend(Legend::default().position(corner))
                        .show(ui, |p| {
                            let pts: Vec<[f64; 2]> = (1880..=1950)
                                .map(|yr| [yr as f64, 0.001 * ((yr - 1900) as f64).powi(2) - 0.2])
                                .collect();
                            p.line(Line::new("Historical Anomaly", pts));
                            p.line(Line::new(
                                "Reference Baseline",
                                vec![[1880.0, 0.0], [1950.0, 0.0]],
                            ));
                        });
                });
            };
            let _ = ctx.run(raw_screen(size, Vec::new()), run);
            let out = ctx.run(raw_screen(size, Vec::new()), run);

            let checkboxes = ak_nodes_by_role(&out, Role::CheckBox);
            let buttons = ak_nodes_by_role(&out, Role::Button);
            assert!(
                checkboxes.iter().any(|(l, _)| l.starts_with("Fit (")),
                "Fitted series must be injected before LegendWidget::try_new ({corner:?}, {size:?})"
            );
            for (cl, cr) in &checkboxes {
                if size.x >= 800.0 {
                    assert!(
                        cr.width() >= 23.5 && cr.height() >= 23.5,
                        "Legend checkbox '{cl}' must meet 24x24 WCAG 2.5.8 target: {cr:?}"
                    );
                }
                for (bl, br) in &buttons {
                    assert_eq!(
                        overlap_area(*cr, *br),
                        0.0,
                        "Legend '{cl}' ({cr:?}) overlapped button '{bl}' ({br:?}) at {corner:?} {size:?}"
                    );
                }
            }
        }
    }
}

fn run_hover_frames(
    ctx: &Context,
    pos: Pos2,
    mut render: impl FnMut(&Context) -> Rect,
) -> Vec<(String, Rect)> {
    let _ = ctx.run(
        raw_screen(vec2(800.0, 600.0), vec![Event::PointerMoved(pos)]),
        |c| {
            let _ = render(c);
        },
    );
    let out = ctx.run(
        raw_screen(vec2(800.0, 600.0), vec![Event::PointerMoved(pos)]),
        |c| {
            let _ = render(c);
        },
    );
    collect_rendered_texts(&out.shapes)
}

#[test]
fn test_axis_label_toggles_hide_ticks_without_disabling_hover_or_pois() {
    let ctx = Context::default();
    let plot_id = Id::new("axis_vs_hover_plot");
    let curve: Vec<[f64; 2]> = (-20..=20)
        .map(|i| {
            let x = i as f64 * 0.25;
            [x, x * x - 1.0]
        })
        .collect();
    let render = |c: &Context| -> Rect {
        let mut frame = Rect::NOTHING;
        egui::CentralPanel::default().show(c, |ui| {
            let r = Plot::new(plot_id).id(plot_id).show(ui, |p| {
                p.line(Line::new("Parabola", curve.clone()));
            });
            frame = *r.transform.frame();
        });
        frame
    };

    let mut frame_with_labels = Rect::NOTHING;
    let _ = ctx.run(raw_screen(vec2(800.0, 600.0), Vec::new()), |c| {
        frame_with_labels = render(c);
    });
    let mut mem = PlotMemory::load(&ctx, plot_id).unwrap();
    mem.show_x_labels = false;
    mem.show_y_labels = false;
    mem.show_hover = true;
    mem.store(&ctx, plot_id);

    let mut frame_no_labels = Rect::NOTHING;
    let _ = ctx.run(raw_screen(vec2(800.0, 600.0), Vec::new()), |c| {
        frame_no_labels = render(c);
    });
    assert!(
        frame_no_labels.width() > frame_with_labels.width()
            && frame_no_labels.height() > frame_with_labels.height(),
        "Hiding X/Y tick labels must collapse extra axis margin"
    );

    let root_screen = PlotMemory::load(&ctx, plot_id)
        .unwrap()
        .transform()
        .position_from_point(&crate::PlotPoint::new(1.0, 0.0));
    let hover_texts = run_hover_frames(&ctx, root_screen, render);
    assert!(
        hover_texts
            .iter()
            .any(|(t, _)| t.contains("Root") || t.contains("Parabola")),
        "Critical-point/curve hover tooltips must remain active: {hover_texts:?}"
    );

    let mut mem_off = PlotMemory::load(&ctx, plot_id).unwrap();
    mem_off.show_hover = false;
    mem_off.store(&ctx, plot_id);
    let off_texts = run_hover_frames(&ctx, root_screen, render);
    assert!(
        !off_texts
            .iter()
            .any(|(t, _)| t.contains("Root") || t.contains("x =")),
        "Turning off Show Hover Readouts must suppress hover tooltips: {off_texts:?}"
    );
}

#[test]
fn test_failed_fit_hides_hide_fit_shows_clear_fit_and_announces_error() {
    let ctx = Context::default();
    ctx.enable_accesskit();
    let plot_id = Id::new("failed_exp_fit_plot");
    let st = AccessState {
        fit_model: Some(FitType::PowerLaw),
        ..Default::default()
    };
    ctx.data_mut(|d| d.insert_temp(plot_id, st));

    let render = |c: &Context| {
        egui::CentralPanel::default().show(c, |ui| {
            Plot::new(plot_id)
                .id(plot_id)
                .legend(Legend::default())
                .show(ui, |p| {
                    p.line(Line::new(
                        "Signed Signal",
                        vec![[-2.0, -2.0], [-1.0, -1.0], [0.0, 0.5]],
                    ));
                });
        });
    };
    let out = ctx.run(raw_screen(vec2(800.0, 600.0), Vec::new()), render);
    let texts: Vec<String> = collect_rendered_texts(&out.shapes)
        .into_iter()
        .map(|(t, _)| t)
        .collect();
    assert!(
        !texts.iter().any(|t| t.contains("Hide Fit")),
        "Failed fit must hide 'Hide Fit': {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("Clear Fit")),
        "Failed fit must show 'Clear Fit': {texts:?}"
    );
    assert!(
        texts
            .iter()
            .any(|t| t.contains("strictly positive X values")),
        "Failed power-law fit must explain why fitting failed: {texts:?}"
    );
    let live_err = ctx
        .data(|d| d.get_temp::<String>(Id::new("aria_live_assertive_message")))
        .unwrap_or_default();
    assert!(
        live_err.to_lowercase().contains("fit failed"),
        "Failed fit must emit an assertive live announcement, got '{live_err}'"
    );
}

#[test]
fn test_scientific_readout_formatting_and_5409_log_ticks_zero_noise() {
    let ctx = Context::default();
    let out = ctx.run(
        raw_screen(
            vec2(800.0, 600.0),
            vec![Event::PointerMoved(pos2(400.0, 300.0))],
        ),
        |c| {
            egui::CentralPanel::default().show(c, |ui| {
                let plot_cfg = crate::items::PlotConfig {
                    ui,
                    transform: &crate::PlotTransform::new(
                        Rect::from_min_size(pos2(0.0, 0.0), vec2(400.0, 300.0)),
                        crate::PlotBounds::from_min_max([0.0, -1e9], [1e-6, 1e9]),
                        egui::Vec2b::FALSE,
                    ),
                    show_x: true,
                    show_y: true,
                    sidecar_3d: None,
                    world_point: None,
                };
                let (_, lines) = format_value_tooltip_content(
                    crate::PlotPoint::new(4.9e-7, -3.2e8),
                    "Trace",
                    &plot_cfg,
                    &None,
                );
                assert_eq!(lines[0].0, "x = 4.9000e-7");
                assert_eq!(lines[1].0, "y = -3.2000e8");
            });
        },
    );
    let _ = out;

    let poi = CriticalPoint {
        kind: CriticalPointKind::Root,
        point: [4.9e-7, 0.0],
        primary_series: "Trace".to_owned(),
        secondary_series: None,
        item_id: Id::new("trace"),
    };
    let summary = format_critical_point_live_summary(&poi, 0, 1);
    assert!(summary.contains("4.9000e-7") && !summary.contains("0.000000"));
    let callout_lbl = poi.callout_label();
    assert!(
        callout_lbl.contains("(4.9000e-7, 0.0000)"),
        "got '{callout_lbl}'"
    );
    let saved = SavedCallout::new([4.9e-7, 1.25], "", [10.0, -10.0]);
    assert!(saved.label.contains("4.9000e-7"), "got '{}'", saved.label);
}

#[test]
fn test_5409_logarithmic_ticks_zero_ieee754_noise() {
    assert_eq!(format_axis_tick(3.0 * 10_f64.powi(-8), 100.0), "3e-8");

    let mut checked = 0_usize;
    for exp in -300..=300 {
        let base = 10_f64.powi(exp);
        for m in 1..=9 {
            let val = (m as f64) * base;
            if !val.is_finite() || val == 0.0 {
                continue;
            }
            let formatted = format_axis_tick(val, 100.0);
            assert!(
                !formatted.contains("00000000") && !formatted.contains("99999999"),
                "Log tick m={m} exp={exp} ({val:e}) showed IEEE-754 rounding noise: '{formatted}'"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 5409);
}
