#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use egui_plot::{Line, PlotPoints};

fn screen_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(800.0, 600.0),
        )),
        events,
        ..Default::default()
    }
}

#[test]
fn test_camera_snapping() {
    let mut camera = Camera3D::default();

    camera.snap_to_xy();
    assert_eq!(camera.pitch, 0.0_f32);
    assert_eq!(camera.yaw, 0.0_f32);

    camera.snap_to_xz();
    assert_eq!(camera.pitch, -std::f32::consts::FRAC_PI_2);
    assert_eq!(camera.yaw, 0.0_f32);

    camera.snap_to_yz();
    assert_eq!(camera.pitch, -std::f32::consts::FRAC_PI_2);
    assert_eq!(camera.yaw, -std::f32::consts::FRAC_PI_2);

    camera.snap_to_iso();
    assert_eq!(camera.pitch, 0.5_f32);
    assert_eq!(camera.yaw, 0.5_f32);
}

#[test]
fn test_camera_qe_zoom_directions_and_r_shortcut_reset() {
    let ctx = egui::Context::default();
    let mut camera = Camera3D::new(0.25, -0.35, 1.0);
    let resp_id = egui::Id::new("cam_kbd_focus");

    let run_key = |cam: &mut Camera3D, key: egui::Key| {
        for pressed in [true, false] {
            let raw = screen_input(vec![egui::Event::Key {
                key,
                physical_key: None,
                pressed,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }]);
            let _ = ctx.run(raw, |sub_ctx| {
                sub_ctx.memory_mut(|m| m.request_focus(resp_id));
                egui::CentralPanel::default().show(sub_ctx, |ui| {
                    let rect =
                        egui::Rect::from_min_size(egui::pos2(20.0, 20.0), egui::vec2(300.0, 300.0));
                    let resp = ui.interact(rect, resp_id, egui::Sense::click_and_drag());
                    cam.handle_interaction(&resp, ui);
                });
            });
        }
    };

    run_key(&mut camera, egui::Key::Q);
    assert!(camera.zoom < 1.0_f32, "Q (Zoom Out) must divide zoom");
    let zoomed_out = camera.zoom;

    run_key(&mut camera, egui::Key::E);
    assert!(camera.zoom > zoomed_out, "E (Zoom In) must multiply zoom");
    run_key(&mut camera, egui::Key::E);
    assert!(camera.zoom > 1.0_f32, "Repeated E must zoom past 1.0");

    camera.pitch = 1.2_f32;
    camera.yaw = -0.9_f32;
    run_key(&mut camera, egui::Key::R);
    assert_eq!(camera.pitch, 0.25_f32);
    assert_eq!(camera.yaw, -0.35_f32);
    assert_eq!(camera.zoom, 1.0_f32);
}

fn drag_frames(
    start: egui::Pos2,
    end: egui::Pos2,
    button: egui::PointerButton,
) -> [Vec<egui::Event>; 3] {
    [
        vec![
            egui::Event::PointerMoved(start),
            egui::Event::PointerButton {
                pos: start,
                button,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        vec![egui::Event::PointerMoved(end)],
        vec![egui::Event::PointerButton {
            pos: end,
            button,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    ]
}

#[test]
fn test_camera_gesture_filtering_secondary_and_callout_drags() {
    let ctx = egui::Context::default();
    let mut camera = Camera3D::default();
    let mut plot_rect = egui::Rect::NOTHING;

    let _ = ctx.run(screen_input(Vec::new()), |sub_ctx| {
        egui::CentralPanel::default().show(sub_ctx, |ui| {
            plot_rect = Plot3DViewport::new("filter_plot", 5.0)
                .show(ui, &mut camera, |plot_ui| {
                    plot_ui.line(Line::new(
                        "axis",
                        PlotPoints::new(vec![[-1.0, 0.0], [1.0, 0.0]]),
                    ));
                })
                .response
                .rect;
        });
    });

    let c = plot_rect.center();
    for evs in drag_frames(
        c,
        c + egui::vec2(50.0, 30.0),
        egui::PointerButton::Secondary,
    ) {
        let _ = ctx.run(screen_input(evs), |sub_ctx| {
            egui::CentralPanel::default().show(sub_ctx, |ui| {
                Plot3DViewport::new("filter_plot", 5.0).show(ui, &mut camera, |_| {});
            });
        });
    }
    assert_eq!(camera, Camera3D::default());

    for evs in drag_frames(c, c + egui::vec2(45.0, -25.0), egui::PointerButton::Primary) {
        let _ = ctx.run(screen_input(evs), |sub_ctx| {
            egui::CentralPanel::default().show(sub_ctx, |ui| {
                Plot3DViewport::new("filter_plot", 5.0).show(ui, &mut camera, |plot_ui| {
                    let pid = plot_ui.plot_id();
                    plot_ui.ctx().data_mut(|d| {
                        d.insert_temp(pid.with("dragging_callout"), true);
                    });
                });
            });
        });
    }
    assert_eq!(camera, Camera3D::default());
}

#[test]
fn test_gizmo_drag_ignored_while_canvas_primary_drag_rotates_with_locked_bounds() {
    let ctx = egui::Context::default();
    let mut camera = Camera3D::default();
    let mut plot_rect = egui::Rect::NOTHING;
    let mut settled_bounds = egui_plot::PlotBounds::NOTHING;

    for _ in 0..2 {
        let _ = ctx.run(screen_input(Vec::new()), |sub_ctx| {
            egui::CentralPanel::default().show(sub_ctx, |ui| {
                let pt = camera.project(&[2.0, 1.0, 0.5]);
                let pr = Plot3DViewport::new("bounds_lock_plot", 4.0).show(ui, &mut camera, |p| {
                    p.line(Line::new("seg", PlotPoints::new(vec![[0.0, 0.0], pt])));
                });
                plot_rect = pr.response.rect;
                settled_bounds = *pr.transform.bounds();
            });
        });
    }

    let palette = crate::app::ThemeMode::Dark.palette();
    let (gizmo_center, _, _, _) = camera.gizmo_geometry(plot_rect, &palette);
    for evs in drag_frames(
        gizmo_center,
        gizmo_center + egui::vec2(-40.0, 40.0),
        egui::PointerButton::Primary,
    ) {
        let _ = ctx.run(screen_input(evs), |sub_ctx| {
            egui::CentralPanel::default().show(sub_ctx, |ui| {
                Plot3DViewport::new("bounds_lock_plot", 4.0).show(ui, &mut camera, |_| {});
            });
        });
    }
    assert_eq!(camera, Camera3D::default());

    let c = plot_rect.center();
    let mut after_bounds = settled_bounds;
    let mut frames = drag_frames(c, c + egui::vec2(60.0, -30.0), egui::PointerButton::Primary);
    frames[2].push(egui::Event::MouseWheel {
        unit: egui::MouseWheelUnit::Point,
        delta: egui::vec2(0.0, 120.0),
        modifiers: egui::Modifiers::NONE,
    });
    for evs in frames {
        let _ = ctx.run(screen_input(evs), |sub_ctx| {
            egui::CentralPanel::default().show(sub_ctx, |ui| {
                let pt = camera.project(&[2.0, 1.0, 0.5]);
                let pr = Plot3DViewport::new("bounds_lock_plot", 4.0).show(ui, &mut camera, |p| {
                    p.line(Line::new("seg", PlotPoints::new(vec![[0.0, 0.0], pt])));
                });
                after_bounds = *pr.transform.bounds();
            });
        });
    }
    assert!((camera.yaw - 0.5_f32).abs() > 0.1_f32 && (camera.pitch - 0.5_f32).abs() > 0.1_f32);
    assert!(camera.zoom > 1.0_f32);
    assert_eq!(after_bounds.min(), settled_bounds.min());
    assert_eq!(after_bounds.max(), settled_bounds.max());
}

#[test]
fn test_camera_reset_view_button_restores_tool_defaults() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut camera = Camera3D::new(0.0, 0.0, 1.0);
    camera.pitch = 0.9;
    camera.yaw = -1.1;
    camera.zoom = 2.4;

    let mut btn_center = egui::Pos2::ZERO;
    let out = ctx.run(screen_input(Vec::new()), |sub_ctx| {
        egui::CentralPanel::default().show(sub_ctx, |ui| camera.ui(ui));
    });
    if let Some(ak) = &out.platform_output.accesskit_update {
        let node = ak
            .nodes
            .iter()
            .find(|(_, n)| {
                n.role() == egui::accesskit::Role::Button
                    && n.label().unwrap_or("").contains("Reset View")
            })
            .map(|(_, n)| n)
            .expect("Reset View button must exist in Camera3D::ui");
        let b = node.bounds().expect("button bounds");
        btn_center = egui::pos2(((b.x0 + b.x1) * 0.5) as f32, ((b.y0 + b.y1) * 0.5) as f32);
    }

    for pressed in [true, false] {
        let evs = vec![
            egui::Event::PointerMoved(btn_center),
            egui::Event::PointerButton {
                pos: btn_center,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            },
        ];
        let _ = ctx.run(screen_input(evs), |sub_ctx| {
            egui::CentralPanel::default().show(sub_ctx, |ui| camera.ui(ui));
        });
    }
    assert_eq!(camera, Camera3D::new(0.0, 0.0, 1.0));
}

#[test]
fn test_camera_double_click_canvas_restores_tool_defaults() {
    let ctx = egui::Context::default();
    let mut camera = Camera3D::new(0.0, 0.0, 1.0);
    camera.pitch = 0.7;
    camera.yaw = 0.8;
    camera.zoom = 1.9;

    let mut warmup = screen_input(Vec::new());
    warmup.time = Some(0.05);
    let _ = ctx.run(warmup, |sub_ctx| {
        egui::CentralPanel::default().show(sub_ctx, |ui| {
            Plot3DViewport::new("dbl_click_plot", 5.0).show(ui, &mut camera, |_| {});
        });
    });

    let click_pos = egui::pos2(300.0, 300.0);
    for (t, pressed) in [(0.10, true), (0.12, false), (0.18, true), (0.20, false)] {
        let mut raw = screen_input(vec![
            egui::Event::PointerMoved(click_pos),
            egui::Event::PointerButton {
                pos: click_pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            },
        ]);
        raw.time = Some(t);
        let _ = ctx.run(raw, |sub_ctx| {
            egui::CentralPanel::default().show(sub_ctx, |ui| {
                Plot3DViewport::new("dbl_click_plot", 5.0).show(ui, &mut camera, |_| {});
            });
        });
    }
    assert_eq!(camera, Camera3D::new(0.0, 0.0, 1.0));
}

#[test]
fn test_shared_zoom_range_across_pinch_scroll_and_slider_ui() {
    assert_eq!(CAMERA_3D_ZOOM_MIN, 0.01_f32);
    assert_eq!(CAMERA_3D_ZOOM_MAX, 100.0_f32);

    let ctx = egui::Context::default();
    let mut camera = Camera3D::default();
    let mut plot_center = egui::pos2(500.0, 300.0);
    let step = |events: Vec<egui::Event>, cam: &mut Camera3D, center: &mut egui::Pos2| {
        let _ = ctx.run(screen_input(events), |sub_ctx| {
            egui::SidePanel::left("cam_zoom_panel").show(sub_ctx, |ui| cam.ui(ui));
            egui::CentralPanel::default().show(sub_ctx, |ui| {
                *center = Plot3DViewport::new("pinch_plot", 4.0)
                    .show(ui, cam, |_| {})
                    .response
                    .rect
                    .center();
            });
        });
    };

    step(Vec::new(), &mut camera, &mut plot_center);

    // Trackpad pinch Event::Zoom scales past legacy 5.0 ceiling while Camera3D::ui is mounted
    let pinch = |factor: f32, pos: egui::Pos2| {
        vec![egui::Event::PointerMoved(pos), egui::Event::Zoom(factor)]
    };
    step(pinch(25.0, plot_center), &mut camera, &mut plot_center);
    step(Vec::new(), &mut camera, &mut plot_center);
    assert!((camera.zoom - 25.0_f32).abs() < 1e-4);

    // Deep zoom-out below legacy 0.1 floor stays reachable and un-clamped above 0.01
    step(pinch(0.001, plot_center), &mut camera, &mut plot_center);
    step(Vec::new(), &mut camera, &mut plot_center);
    assert!((camera.zoom - 0.025_f32).abs() < 1e-5);

    // Extreme pinch clamps cleanly to [0.01, 100.0]
    step(pinch(0.01, plot_center), &mut camera, &mut plot_center);
    assert_eq!(camera.zoom, CAMERA_3D_ZOOM_MIN);
    step(pinch(50_000.0, plot_center), &mut camera, &mut plot_center);
    assert_eq!(camera.zoom, CAMERA_3D_ZOOM_MAX);
}

#[test]
fn test_first_interaction_zoom_scales_geometry_without_auto_bounds_cancellation() {
    let ctx = egui::Context::default();
    let mut camera = Camera3D::default();
    let mut bounds_f0 = egui_plot::PlotBounds::NOTHING;
    let mut span_f0 = 0.0_f32;

    let _ = ctx.run(screen_input(Vec::new()), |sub_ctx| {
        egui::CentralPanel::default().show(sub_ctx, |ui| {
            let p_a = camera.project(&[-1.5, 0.0, 0.0]);
            let p_b = camera.project(&[1.5, 0.0, 0.0]);
            let pr = Plot3DViewport::new("first_zoom_plot", 5.0).show(ui, &mut camera, |plot_ui| {
                plot_ui.line(Line::new("span", PlotPoints::new(vec![p_a, p_b])));
            });
            bounds_f0 = *pr.transform.bounds();
            let s_a = pr
                .transform
                .position_from_point(&egui_plot::PlotPoint::new(p_a[0], p_a[1]));
            let s_b = pr
                .transform
                .position_from_point(&egui_plot::PlotPoint::new(p_b[0], p_b[1]));
            span_f0 = s_a.distance(s_b);
        });
    });

    // Immediately change Zoom on the very first interaction of a newly opened 3D viewport
    camera.zoom = 2.5_f32;
    let mut bounds_f1 = egui_plot::PlotBounds::NOTHING;
    let mut span_f1 = 0.0_f32;
    let _ = ctx.run(screen_input(Vec::new()), |sub_ctx| {
        egui::CentralPanel::default().show(sub_ctx, |ui| {
            let p_a = camera.project(&[-1.5, 0.0, 0.0]);
            let p_b = camera.project(&[1.5, 0.0, 0.0]);
            let pr = Plot3DViewport::new("first_zoom_plot", 5.0).show(ui, &mut camera, |plot_ui| {
                plot_ui.line(Line::new("span", PlotPoints::new(vec![p_a, p_b])));
            });
            bounds_f1 = *pr.transform.bounds();
            let s_a = pr
                .transform
                .position_from_point(&egui_plot::PlotPoint::new(p_a[0], p_a[1]));
            let s_b = pr
                .transform
                .position_from_point(&egui_plot::PlotPoint::new(p_b[0], p_b[1]));
            span_f1 = s_a.distance(s_b);
        });
    });

    assert_eq!(bounds_f1.min()[0], bounds_f0.min()[0]);
    assert_eq!(bounds_f1.max()[0], bounds_f0.max()[0]);
    assert!((bounds_f1.min()[1] - bounds_f0.min()[1]).abs() < 0.05);
    assert!(((span_f1 / span_f0) - 2.5_f32).abs() < 0.02_f32);
}

#[test]
fn test_all_six_3d_tools_expose_reset_view_and_helper_on_initial_open() {
    let tool_ids: &[&str] = &[
        #[cfg(feature = "pure_math")]
        "SurfaceViewer",
        #[cfg(feature = "physics")]
        "SpinVisualizer",
        #[cfg(feature = "physics")]
        "AttractorPlotter",
        #[cfg(feature = "physics")]
        "CrystalViewer",
        #[cfg(feature = "pure_math")]
        "CurvatureHeatmap",
        #[cfg(feature = "applied")]
        "ImplicitCadStudio",
        #[cfg(feature = "ai")]
        "Gaussian Splatting",
    ];
    for &id in tool_ids {
        let meta = inventory::iter::<crate::framework::ToolMetadata>
            .into_iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("Missing 3D ToolMetadata for {id}"));
        let mut tool = (meta.build)();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let out = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280.0, 800.0),
                )),
                ..Default::default()
            },
            |sub_ctx| tool.show(sub_ctx),
        );
        let ak = out
            .platform_output
            .accesskit_update
            .expect("AccessKit update expected");
        let has_reset = ak.nodes.iter().any(|(_, n)| {
            n.role() == egui::accesskit::Role::Button
                && n.label().unwrap_or("").contains("Reset View")
        });
        assert!(
            has_reset,
            "3D tool '{id}' must expose Reset View button on initial open"
        );
        let has_helper = ak.nodes.iter().any(|(_, n)| {
            [n.label(), n.value()]
                .into_iter()
                .flatten()
                .any(|s| s.contains("Drag canvas or use WASD to orbit"))
        });
        assert!(
            has_helper,
            "3D tool '{id}' must expose shared 3D control helper caption"
        );
    }
}

#[test]
fn test_camera_ui_and_validation_reject_non_finite_and_overflow_angles() {
    assert!(!Camera3D::new(1e38_f32, 0.0, 1.0).is_valid());
    assert!(!Camera3D::new(0.0, -1e38_f32, 1.0).is_valid());
    assert!(!Camera3D::new(0.0, 0.0, f32::NAN).is_valid());

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut cam = Camera3D::new(0.4, -0.3, 1.5);
    let run_ui = |events, cam: &mut Camera3D| {
        ctx.run(screen_input(events), |c| {
            egui::CentralPanel::default().show(c, |ui| cam.ui(ui));
        })
    };
    let out = run_ui(Vec::new(), &mut cam);
    let spins: Vec<egui::Pos2> = out
        .platform_output
        .accesskit_update
        .expect("accesskit")
        .nodes
        .iter()
        .filter(|(_, n)| n.role() == egui::accesskit::Role::SpinButton)
        .filter_map(|(_, n)| n.bounds())
        .map(|b| egui::pos2(((b.x0 + b.x1) * 0.5) as f32, ((b.y0 + b.y1) * 0.5) as f32))
        .collect();
    assert_eq!(spins.len(), 3);

    for &pos in &spins {
        let [down, _, up] = drag_frames(pos, pos, egui::PointerButton::Primary);
        for bad in ["NaN", "1e400", "1e38"] {
            let _ = run_ui([down.clone(), up.clone()].concat(), &mut cam);
            let enter = egui::Event::Key {
                key: egui::Key::Enter,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            };
            let _ = run_ui(vec![egui::Event::Text(bad.into()), enter], &mut cam);
            let _ = run_ui(Vec::new(), &mut cam);
            assert!(cam.is_valid(), "Camera3D must remain valid after '{bad}'");
        }
    }
}
