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
