use super::gizmo::GizmoAxis;
use super::*;
use eframe::egui;
use egui_plot::{commands::CommandRegistryData, Line, PlotPoints};

fn raw_in(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        predicted_dt: 1.0 / 60.0,
        events,
        ..Default::default()
    }
}

#[cfg(feature = "physics")]
fn pointer_click(pos: egui::Pos2) -> Vec<egui::Event> {
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

#[test]
fn test_all_16_camera_3d_command_palette_actions_execute_and_announce() {
    let ctx = egui::Context::default();
    let mut cam = Camera3D::new(0.2, -0.3, 1.0);
    let _ = ctx.run(raw_in(Vec::new()), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            let _ = Plot3DViewport::new("cmd_palette_3d", 4.0).show(ui, &mut cam, |_| {});
        });
    });

    let commands = [
        "Snap to X-Axis View",
        "Snap to Y-Axis View",
        "Snap to Z-Axis View",
        "Snap to Isometric View",
        "Toggle Projection",
        "Zoom In",
        "Zoom Out",
        "Pitch Up",
        "Pitch Down",
        "Yaw Left",
        "Yaw Right",
        "Pan Up",
        "Pan Down",
        "Pan Left",
        "Pan Right",
        "Reset View",
    ];
    assert_eq!(commands.len(), 16);

    for cmd in commands {
        crate::accessibility::clear_test_announcements();
        let before = cam;
        let _ = ctx.run(raw_in(Vec::new()), |c| {
            CommandRegistryData::dispatch(c, cmd, "Camera 3D");
            egui::CentralPanel::default().show(c, |ui| {
                let _ = Plot3DViewport::new("cmd_palette_3d", 4.0).show(ui, &mut cam, |_| {});
            });
        });
        assert_ne!(cam, before, "Command '{cmd}' did not update Camera3D");
        let anns = crate::accessibility::all_announcements();
        assert!(
            anns.iter().any(|(_, prio)| prio == "polite"),
            "Command '{cmd}' emitted no polite live announcement"
        );
    }
    assert_eq!((cam.pitch, cam.yaw), (0.2, -0.3));
}

#[cfg(feature = "physics")]
#[test]
fn test_attractor_plotter_command_palette_iso_snap_vs_home_reset() {
    use crate::framework::InteractiveTool;
    use crate::tabs::chaos::attractors::{AttractorPlotter, SolverExecutionState};

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = AttractorPlotter {
        execution_state: SolverExecutionState::Paused,
        ..Default::default()
    };
    assert_eq!((tool.camera.pitch, tool.camera.yaw), (0.0, 0.0));

    crate::accessibility::clear_test_announcements();
    let _ = ctx.run(raw_in(Vec::new()), |c| {
        CommandRegistryData::dispatch(c, "Snap to Isometric View", "Camera 3D");
        tool.show(c);
    });
    assert_eq!((tool.camera.pitch, tool.camera.yaw), (0.5, 0.5));
    assert!(crate::accessibility::all_announcements()
        .iter()
        .any(|(m, _)| m.contains("Isometric")));

    // Focus the 3D GraphicsDocument and press Key::Home to reset back to Attractor's (0.0, 0.0).
    let out = ctx.run(raw_in(Vec::new()), |c| tool.show(c));
    let ak = out.platform_output.accesskit_update.expect("AccessKit");
    let (doc_id, doc_node) = ak
        .nodes
        .iter()
        .find(|(_, n)| {
            n.role() == egui::accesskit::Role::GraphicsDocument
                && n.label().unwrap_or("").contains("Attractor")
        })
        .expect("Attractor GraphicsDocument");
    assert!(doc_node.value().unwrap_or("").contains("Pitch 28.6°"));

    let b = doc_node.bounds().expect("bounds");
    let center = egui::pos2(((b.x0 + b.x1) * 0.5) as f32, ((b.y0 + b.y1) * 0.5) as f32);
    let _ = ctx.run(raw_in(pointer_click(center)), |c| tool.show(c));

    crate::accessibility::clear_test_announcements();
    let home_ev = vec![egui::Event::Key {
        key: egui::Key::Home,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }];
    let out_reset = ctx.run(raw_in(home_ev), |c| tool.show(c));
    assert_eq!((tool.camera.pitch, tool.camera.yaw), (0.0, 0.0));
    let anns = crate::accessibility::all_announcements();
    assert!(anns.iter().any(|(m, _)| m.contains("Camera view reset")));
    assert!(!anns.iter().any(|(m, _)| m.contains("Plot bounds reset")));

    let ak_reset = out_reset
        .platform_output
        .accesskit_update
        .expect("AccessKit");
    let (_, reset_node) = ak_reset
        .nodes
        .iter()
        .find(|(id, _)| id == doc_id)
        .expect("Attractor GraphicsDocument after reset");
    assert!(reset_node.value().unwrap_or("").contains("Pitch 0.0°"));
}

#[test]
fn test_detached_iso_pill_unoccluded_across_all_snaps_and_scales() {
    let palette = crate::app::ThemeMode::Dark.palette();
    for size in [
        egui::vec2(160.0, 140.0),
        egui::vec2(320.0, 240.0),
        egui::vec2(640.0, 480.0),
    ] {
        let rect = egui::Rect::from_min_size(egui::pos2(20.0, 20.0), size);
        for snap in [
            Camera3D::snap_to_yz,
            Camera3D::snap_to_xz,
            Camera3D::snap_to_xy,
            Camera3D::snap_to_iso,
        ] {
            let mut cam = Camera3D::default();
            snap(&mut cam);
            let (center, axis_len, knob_r, handles) = cam.gizmo_geometry(rect, &palette);
            let iso = handles
                .iter()
                .find(|h| h.axis == GizmoAxis::Iso)
                .expect("Iso handle");
            assert_eq!(iso.button_rect.width(), 28.0);
            assert_eq!(iso.button_rect.height(), 24.0);
            let bg_radius = axis_len + knob_r + 4.0;
            assert!(
                iso.button_rect.min.y >= center.y + bg_radius,
                "ISO pill must sit completely below triad disk"
            );
            for other in handles.iter().filter(|h| h.axis != GizmoAxis::Iso) {
                assert!(
                    !iso.button_rect.intersects(other.button_rect),
                    "ISO pill overlapped {:?} button_rect",
                    other.axis
                );
            }
            for corner in [
                iso.button_rect.left_top() + egui::vec2(1.0, 1.0),
                iso.button_rect.center(),
                iso.button_rect.right_bottom() - egui::vec2(1.0, 1.0),
            ] {
                assert_eq!(
                    Camera3D::pick_frontmost_gizmo_axis(&handles, corner, knob_r),
                    Some(GizmoAxis::Iso)
                );
            }
        }
    }
}

fn drag_pointer_events(start: egui::Pos2, end: egui::Pos2) -> [Vec<egui::Event>; 3] {
    [
        vec![
            egui::Event::PointerMoved(start),
            egui::Event::PointerButton {
                pos: start,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        vec![egui::Event::PointerMoved(end)],
        vec![egui::Event::PointerButton {
            pos: end,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    ]
}

#[test]
fn test_continuous_drag_and_held_wasd_frames_suppress_live_announcements() {
    let ctx = egui::Context::default();
    let mut cam = Camera3D::default();
    let mut plot_center = egui::Pos2::ZERO;
    let mut plot_id = egui::Id::NULL;
    let _ = ctx.run(raw_in(Vec::new()), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            let pr = Plot3DViewport::new("silent_drag_vp", 4.0).show(ui, &mut cam, |p| {
                p.line(Line::new(
                    "seg",
                    PlotPoints::new(vec![[-1.0, -1.0], [1.0, 1.0]]),
                ));
            });
            plot_center = pr.response.rect.center();
            plot_id = pr.response.id;
        });
    });

    crate::accessibility::clear_test_announcements();
    let dest = plot_center + egui::vec2(30.0, -20.0);
    for evs in drag_pointer_events(plot_center, dest) {
        let _ = ctx.run(raw_in(evs), |c| {
            egui::CentralPanel::default().show(c, |ui| {
                let _ = Plot3DViewport::new("silent_drag_vp", 4.0).show(ui, &mut cam, |_| {});
            });
        });
    }
    assert_ne!((cam.pitch, cam.yaw), (0.5, 0.5));
    assert!(crate::accessibility::all_announcements().is_empty());

    for key in [egui::Key::W, egui::Key::A, egui::Key::Q, egui::Key::E] {
        let evs = vec![egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }];
        let _ = ctx.run(raw_in(evs), |c| {
            c.memory_mut(|m| m.request_focus(plot_id));
            egui::CentralPanel::default().show(c, |ui| {
                let _ = Plot3DViewport::new("silent_drag_vp", 4.0).show(ui, &mut cam, |_| {});
            });
        });
    }
    assert!(crate::accessibility::all_announcements().is_empty());
}
