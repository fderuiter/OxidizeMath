#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::gizmo::GizmoAxis;
use super::*;
use eframe::egui;
use egui_plot::{Line, ModalLifecycle, PlotMemory, PlotPoints};

fn viewport_input(events: Vec<egui::Event>, dt: f32) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        predicted_dt: dt,
        events,
        ..Default::default()
    }
}

fn click_events(pos: egui::Pos2) -> Vec<egui::Event> {
    vec![
        egui::Event::PointerMoved(pos),
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        },
    ]
}

fn assert_snapped_knob_selects_axis(setup: fn(&mut Camera3D), expected_axis: GizmoAxis) {
    let rect = egui::Rect::from_min_size(egui::pos2(40.0, 40.0), egui::vec2(600.0, 480.0));
    let palette = crate::app::ThemeMode::Dark.palette();
    let ctx = egui::Context::default();
    let resp_id = egui::Id::new(("gizmo_snap", expected_axis));

    let mut cam = Camera3D::default();
    setup(&mut cam);
    let (_, _, _, handles) = cam.gizmo_geometry(rect, &palette);
    let target = handles
        .iter()
        .find(|h| h.axis == expected_axis)
        .expect("axis handle")
        .screen_pos;

    let mut probe = Camera3D::new(0.23, -0.41, 1.0);
    let _ = ctx.run(viewport_input(Vec::new(), 1.0 / 60.0), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            let r = ui.interact(rect, resp_id, egui::Sense::click_and_drag());
            let _ = probe.draw_gizmo_overlay(&r, ui);
        });
    });
    setup(&mut probe);
    let mut clicked = false;
    let _ = ctx.run(viewport_input(click_events(target), 1.0 / 60.0), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            let r = ui.interact(rect, resp_id, egui::Sense::click_and_drag());
            clicked = probe.draw_gizmo_overlay(&r, ui);
        });
    });
    assert!(clicked, "Expected gizmo click on {expected_axis:?}");
    assert_eq!((probe.pitch, probe.yaw), (cam.pitch, cam.yaw));
    assert_ne!((probe.pitch, probe.yaw), (0.5_f32, 0.5_f32));
}

#[test]
fn test_snapped_gizmo_clicks_select_frontmost_axis_and_match_spherical_poses() {
    assert_snapped_knob_selects_axis(Camera3D::snap_to_yz, GizmoAxis::X);
    assert_snapped_knob_selects_axis(Camera3D::snap_to_xz, GizmoAxis::Y);
    assert_snapped_knob_selects_axis(Camera3D::snap_to_xy, GizmoAxis::Z);

    let rect = egui::Rect::from_min_size(egui::pos2(40.0, 40.0), egui::vec2(600.0, 480.0));
    let palette = crate::app::ThemeMode::Dark.palette();
    for p_idx in -6..=6 {
        for y_idx in -12..=12 {
            let pitch = (p_idx as f32) * (std::f32::consts::FRAC_PI_2 / 6.0);
            let yaw = (y_idx as f32) * (std::f32::consts::PI / 12.0);
            let cam = Camera3D::new(pitch, yaw, 1.0);
            let (_, _, knob_r, handles) = cam.gizmo_geometry(rect, &palette);
            for h in &handles {
                let picked = Camera3D::pick_frontmost_gizmo_axis(&handles, h.screen_pos, knob_r)
                    .expect("handle center must hit a gizmo axis");
                let expected = handles
                    .iter()
                    .filter(|cand| cand.screen_pos.distance(h.screen_pos) <= knob_r + 4.0)
                    .max_by(|a, b| a.depth.total_cmp(&b.depth))
                    .expect("at least self within hit disk")
                    .axis;
                assert_eq!(picked, expected, "Mismatch at pitch={pitch}, yaw={yaw}");
            }
        }
    }
}

#[test]
fn test_gizmo_accesskit_buttons_target_size_and_toolbar_clearance() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut cam = Camera3D::default();
    let mut plot_rect = egui::Rect::NOTHING;

    let out = ctx.run(viewport_input(Vec::new(), 1.0 / 60.0), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            let pr = Plot3DViewport::new("a11y_gizmo_vp", 4.0).show(ui, &mut cam, |plot_ui| {
                plot_ui.line(Line::new(
                    "axis",
                    PlotPoints::new(vec![[-1.0, -1.0], [1.0, 1.0]]),
                ));
            });
            plot_rect = pr.response.rect;
        });
    });

    let ak = out
        .platform_output
        .accesskit_update
        .expect("AccessKit update");
    for token in [
        "Orient camera to X",
        "Orient camera to Y",
        "Orient camera to Z",
        "Orient camera to ISO",
    ] {
        let (_, node) = ak
            .nodes
            .iter()
            .find(|(_, n)| n.role() == egui::accesskit::Role::Button && n.label() == Some(token))
            .unwrap_or_else(|| panic!("Missing AccessKit Role::Button for '{token}'"));
        assert!(node.supports_action(egui::accesskit::Action::Focus));
        let bounds = node.bounds().expect("gizmo button bounds");
        assert!((bounds.x1 - bounds.x0) >= 24.0 && (bounds.y1 - bounds.y0) >= 24.0);
        assert!(bounds.y0 >= f64::from(plot_rect.min.y + 30.0) - 0.5);
    }
}

#[test]
fn test_gizmo_modal_lifecycle_blocks_background_clicks() {
    let ctx = egui::Context::default();
    let mut cam = Camera3D::default();
    let mut plot_rect = egui::Rect::NOTHING;
    let _ = ctx.run(viewport_input(Vec::new(), 1.0 / 60.0), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            let pr = Plot3DViewport::new("modal_occ_vp", 4.0).show(ui, &mut cam, |_| {});
            plot_rect = pr.response.rect;
        });
    });

    let palette = crate::app::ThemeMode::Dark.palette();
    let (_, _, _, handles) = cam.gizmo_geometry(plot_rect, &palette);
    let x_pos = handles
        .iter()
        .find(|h| h.axis == GizmoAxis::X)
        .expect("X handle")
        .screen_pos;
    let mut modal_open = true;
    for evs in [Vec::new(), click_events(x_pos)] {
        let _ = ctx.run(viewport_input(evs, 1.0 / 60.0), |c| {
            egui::CentralPanel::default().show(c, |ui| {
                let _ = Plot3DViewport::new("modal_occ_vp", 4.0).show(ui, &mut cam, |_| {});
            });
            ModalLifecycle::new(egui::Id::new("blocking_test_modal"))
                .title("Blocking Modal")
                .show(c, &mut modal_open, |ui, _| {
                    ui.label("Foreground modal content");
                });
        });
    }
    assert_eq!((cam.pitch, cam.yaw), (0.5_f32, 0.5_f32));
}

#[test]
fn test_gizmo_keyboard_enter_activation_and_numeric_snap_shortcuts() {
    let ctx = egui::Context::default();
    let mut cam = Camera3D::default();
    let resp_id = egui::Id::new("kbd_gizmo_canvas");
    let rect = egui::Rect::from_min_size(egui::pos2(20.0, 20.0), egui::vec2(400.0, 320.0));
    let x_btn_id = resp_id.with(("camera_gizmo_btn", GizmoAxis::X));

    let _ = ctx.run(viewport_input(Vec::new(), 1.0 / 60.0), |c| {
        c.memory_mut(|m| m.request_focus(x_btn_id));
        egui::CentralPanel::default().show(c, |ui| {
            let r = ui.interact(rect, resp_id, egui::Sense::click_and_drag());
            cam.handle_interaction(&r, ui);
        });
    });
    for pressed in [true, false] {
        let ev = egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };
        let _ = ctx.run(viewport_input(vec![ev], 1.0 / 60.0), |c| {
            egui::CentralPanel::default().show(c, |ui| {
                let r = ui.interact(rect, resp_id, egui::Sense::click_and_drag());
                cam.handle_interaction(&r, ui);
            });
        });
    }
    assert_eq!(
        (cam.pitch, cam.yaw),
        (-std::f32::consts::FRAC_PI_2, -std::f32::consts::FRAC_PI_2)
    );

    for (key, expected) in [
        (egui::Key::Num2, (-std::f32::consts::FRAC_PI_2, 0.0_f32)),
        (egui::Key::Num3, (0.0_f32, 0.0_f32)),
        (egui::Key::Num4, (0.5_f32, 0.5_f32)),
    ] {
        for pressed in [true, false] {
            let ev = egui::Event::Key {
                key,
                physical_key: None,
                pressed,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            };
            let _ = ctx.run(viewport_input(vec![ev], 1.0 / 60.0), |c| {
                c.memory_mut(|m| m.request_focus(resp_id));
                egui::CentralPanel::default().show(c, |ui| {
                    let r = ui.interact(rect, resp_id, egui::Sense::click_and_drag());
                    cam.handle_interaction(&r, ui);
                });
            });
        }
        assert_eq!((cam.pitch, cam.yaw), expected);
    }
}

#[test]
fn test_camera_modifier_guards_prevent_ctrl_a_and_ctrl_s_drift() {
    let ctx = egui::Context::default();
    let mut cam = Camera3D::new(0.4, -0.3, 1.2);
    let baseline = cam;
    let resp_id = egui::Id::new("mod_guard_canvas");
    let rect = egui::Rect::from_min_size(egui::pos2(20.0, 20.0), egui::vec2(400.0, 300.0));

    for mods in [
        egui::Modifiers::CTRL,
        egui::Modifiers::COMMAND,
        egui::Modifiers::ALT,
    ] {
        for key in [
            egui::Key::A,
            egui::Key::S,
            egui::Key::W,
            egui::Key::D,
            egui::Key::Q,
            egui::Key::E,
            egui::Key::R,
        ] {
            let press = egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: mods,
            };
            let release = egui::Event::Key {
                key,
                physical_key: None,
                pressed: false,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            };
            for evs in [vec![press], Vec::new(), vec![release]] {
                let _ = ctx.run(viewport_input(evs, 1.0 / 60.0), |c| {
                    c.memory_mut(|m| m.request_focus(resp_id));
                    egui::CentralPanel::default().show(c, |ui| {
                        let r = ui.interact(rect, resp_id, egui::Sense::click_and_drag());
                        cam.handle_interaction(&r, ui);
                    });
                });
            }
            assert_eq!(cam, baseline, "{mods:?}+{key:?} drifted camera");
        }
    }
}

fn simulate_keyboard_hold_for_one_second(hz: usize) -> Camera3D {
    let sim_ctx = egui::Context::default();
    let mut c3d = Camera3D::new(0.5, 0.5, 1.0);
    let dt = 1.0_f32 / (hz as f32);
    let id = egui::Id::new(("hz_sim", hz));
    let rect = egui::Rect::from_min_size(egui::pos2(20.0, 20.0), egui::vec2(400.0, 300.0));
    for frame_idx in 0..hz {
        let evs = if frame_idx == 0 {
            vec![
                egui::Event::Key {
                    key: egui::Key::D,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::Key {
                    key: egui::Key::E,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ]
        } else {
            Vec::new()
        };
        let mut raw = viewport_input(evs, dt);
        raw.time = Some((frame_idx as f64) * f64::from(dt));
        let _ = sim_ctx.run(raw, |c| {
            c.memory_mut(|m| m.request_focus(id));
            egui::CentralPanel::default().show(c, |ui| {
                let r = ui.interact(rect, id, egui::Sense::click_and_drag());
                c3d.handle_interaction(&r, ui);
            });
        });
    }
    c3d
}

#[test]
fn test_camera_60hz_matches_144hz_keyboard_velocity_and_secondary_pan() {
    let cam_60 = simulate_keyboard_hold_for_one_second(60);
    let cam_144 = simulate_keyboard_hold_for_one_second(144);
    assert!((cam_60.yaw - cam_144.yaw).abs() < 1e-3);
    assert!((cam_60.zoom - cam_144.zoom).abs() < 1e-3);

    let ctx = egui::Context::default();
    let mut standalone = Camera3D::default();
    let id = egui::Id::new("standalone_sec_pan");
    let rect = egui::Rect::from_min_size(egui::pos2(20.0, 20.0), egui::vec2(300.0, 300.0));
    let p0 = egui::pos2(140.0, 140.0);
    let p1 = egui::pos2(190.0, 110.0);
    let seq = [
        Vec::new(),
        vec![
            egui::Event::PointerMoved(p0),
            egui::Event::PointerButton {
                pos: p0,
                button: egui::PointerButton::Secondary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        vec![egui::Event::PointerMoved(p1)],
        vec![egui::Event::PointerButton {
            pos: p1,
            button: egui::PointerButton::Secondary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    ];
    for evs in seq {
        let _ = ctx.run(viewport_input(evs, 1.0 / 60.0), |c| {
            egui::CentralPanel::default().show(c, |ui| {
                let r = ui.interact(rect, id, egui::Sense::click_and_drag());
                standalone.handle_interaction(&r, ui);
            });
        });
    }
    assert!(standalone.pan_x > 0.0 && standalone.pan_y > 0.0);
    assert_eq!((standalone.pitch, standalone.yaw), (0.5, 0.5));
}

#[cfg(all(feature = "pure_math", feature = "physics", feature = "ai"))]
#[test]
fn test_all_six_3d_tools_hide_2d_settings_and_preserve_exports() {
    let tool_specs = [
        ("SurfaceViewer", true),
        ("CurvatureHeatmap", true),
        ("CrystalViewer", true),
        ("AttractorPlotter", true),
        ("SpinVisualizer", true),
        ("Gaussian Splatting", false),
    ];

    for (id, expects_plot_export) in tool_specs {
        let meta = inventory::iter::<crate::framework::ToolMetadata>
            .into_iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("Missing 3D ToolMetadata for {id}"));
        let mut tool = (meta.build)();
        assert_eq!(tool.export_capabilities().dataset, expects_plot_export);
        assert_eq!(tool.export_capabilities().figure, expects_plot_export);

        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        egui_plot::clear_last_plot_datasets();
        let out = ctx.run(viewport_input(Vec::new(), 1.0 / 60.0), |c| tool.show(c));
        assert!(!egui_plot::get_last_plot_datasets().is_empty());
        let ak = out.platform_output.accesskit_update.expect("AccessKit");
        for (_, node) in &ak.nodes {
            let lbl = node.label().unwrap_or("");
            assert!(
                !lbl.contains("🔧 Settings")
                    && !lbl.contains("Plot grid and axis settings")
                    && !lbl.contains("Logarithmic X-Axis")
                    && !lbl.contains("Logarithmic Y-Axis")
                    && !lbl.contains("Fit Curve"),
                "3D tool '{id}' leaked 2D plot control '{lbl}'"
            );
        }
    }
}

#[test]
fn test_plot3d_viewport_clears_stale_log_memory_without_culling_negative_bounds() {
    let ctx = egui::Context::default();
    let mut cam = Camera3D::default();
    let mut pid = egui::Id::NULL;
    let _ = ctx.run(viewport_input(Vec::new(), 1.0 / 60.0), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            let _ = Plot3DViewport::new("log_clear_3d", 4.0).show(ui, &mut cam, |p| {
                pid = p.plot_id();
            });
        });
    });
    let mut stale = PlotMemory::load(&ctx, pid).expect("PlotMemory");
    stale.log_x = true;
    stale.log_y = true;
    stale.show_settings_modal = true;
    stale.store(&ctx, pid);

    let _ = ctx.run(viewport_input(Vec::new(), 1.0 / 60.0), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            let _ = Plot3DViewport::new("log_clear_3d", 4.0).show(ui, &mut cam, |p| {
                p.line(Line::new(
                    "neg_3d_seg",
                    PlotPoints::new(vec![[-3.0, -2.5], [2.5, 3.0]]),
                ));
            });
        });
    });
    let cleaned = PlotMemory::load(&ctx, pid).expect("PlotMemory after 3D frame");
    assert!(!cleaned.log_x && !cleaned.log_y && !cleaned.show_settings_modal);
    assert_eq!(cleaned.transform().log_axis(), egui::Vec2b::FALSE);
    assert!(cleaned.transform().bounds().min()[0] <= -4.0);
    assert!(cleaned.transform().bounds().min()[1] <= -4.0);
}
