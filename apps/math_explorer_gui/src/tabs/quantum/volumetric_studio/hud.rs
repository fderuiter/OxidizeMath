#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Viewport HUD overlays (`Orbit | Pan | Zoom | Recenter` pill, `BOUNDS` & `PROJECTION` badges,
//! `XYZ WORLD` 3D orientation gizmo, and quiet `Mesh / Step` telemetry pill) styled strictly via
//! [`crate::app::SemanticPalette`].

use super::state::FieldSourceMode;
use super::webgpu_runner::ViewportNavMode;
use super::VolumetricFieldStudioTool;
use crate::app::SemanticPalette;
use crate::camera::ProjectionMode;
use eframe::egui;

/// Renders all four Stitch-aligned HUD corner overlays inside `viewport_rect`.
/// Returns `true` if camera state was modified by a HUD button click (`Recenter` or `PROJECTION`).
pub(super) fn draw_viewport_hud_overlays(
    ui: &mut egui::Ui,
    viewport_rect: egui::Rect,
    tool: &mut VolumetricFieldStudioTool,
    palette: &SemanticPalette,
) -> bool {
    if viewport_rect.width() < 260.0 || viewport_rect.height() < 180.0 {
        return false;
    }
    let mut camera_changed = draw_top_left_nav_pill(ui, viewport_rect, tool, palette);
    camera_changed |= draw_top_right_bounds_and_projection(ui, viewport_rect, tool, palette);
    draw_bottom_left_xyz_gizmo(ui, viewport_rect, tool, palette);
    draw_bottom_right_telemetry_pill(ui, viewport_rect, tool, palette);
    camera_changed
}

fn hud_pill_frame(ui: &egui::Ui, palette: &SemanticPalette) -> egui::Frame {
    egui::Frame::new()
        .fill(palette.surface_fill.gamma_multiply(0.92))
        .stroke(egui::Stroke::new(1.0_f32, palette.graph_edge))
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::symmetric(8, 5))
        .shadow(ui.visuals().popup_shadow)
}

fn draw_top_left_nav_pill(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    tool: &mut VolumetricFieldStudioTool,
    palette: &SemanticPalette,
) -> bool {
    let mut reset_clicked = false;
    let pos = rect.left_top() + egui::vec2(12.0, 12.0);
    egui::Area::new(ui.id().with("vol_hud_top_left"))
        .order(egui::Order::Foreground)
        .fixed_pos(pos)
        .constrain_to(rect)
        .show(ui.ctx(), |ui| {
            hud_pill_frame(ui, palette).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    for (mode, label) in [
                        (ViewportNavMode::Orbit, "Orbit"),
                        (ViewportNavMode::Pan, "Pan"),
                        (ViewportNavMode::Zoom, "Zoom"),
                    ] {
                        let selected = tool.webgpu_runner.nav_mode == mode;
                        if ui.selectable_label(selected, label).clicked() {
                            tool.webgpu_runner.nav_mode = mode;
                        }
                    }
                    ui.separator();
                    if ui
                        .small_button("Recenter")
                        .on_hover_text("Reset 3D camera orbit, pan, and zoom (R)")
                        .clicked()
                    {
                        tool.config.camera.reset();
                        reset_clicked = true;
                    }
                });
            });
        });
    reset_clicked
}

fn draw_top_right_bounds_and_projection(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    tool: &mut VolumetricFieldStudioTool,
    palette: &SemanticPalette,
) -> bool {
    let mut proj_toggled = false;
    let bounds_text = if tool.config.source == FieldSourceMode::CustomImplicit {
        format!(
            "[{:.1}, {:+.1}]³",
            tool.config.x_bounds[0], tool.config.x_bounds[1]
        )
    } else {
        "[-2.5, +2.5]³".to_string()
    };
    let proj_label = match tool.config.camera.projection {
        ProjectionMode::Perspective => "45° PERSPECTIVE",
        ProjectionMode::Parallel => "PARALLEL ORTHO",
    };
    let anchor = rect.right_top() + egui::vec2(-12.0, 12.0);
    egui::Area::new(ui.id().with("vol_hud_top_right"))
        .order(egui::Order::Foreground)
        .pivot(egui::Align2::RIGHT_TOP)
        .fixed_pos(anchor)
        .constrain_to(rect)
        .show(ui.ctx(), |ui| {
            ui.horizontal(|ui| {
                hud_pill_frame(ui, palette).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("BOUNDS")
                                .size(9.5)
                                .strong()
                                .color(palette.muted_text),
                        );
                        ui.label(egui::RichText::new(&bounds_text).monospace().size(10.5));
                    });
                });
                hud_pill_frame(ui, palette).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("PROJECTION")
                                .size(9.5)
                                .strong()
                                .color(palette.muted_text),
                        );
                        if ui
                            .small_button(egui::RichText::new(proj_label).monospace().size(10.5))
                            .on_hover_text("Toggle 3D camera projection (Perspective / Parallel)")
                            .clicked()
                        {
                            tool.config.camera.projection = tool.config.camera.projection.toggle();
                            proj_toggled = true;
                        }
                    });
                });
            });
        });
    proj_toggled
}

fn draw_bottom_left_xyz_gizmo(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    tool: &VolumetricFieldStudioTool,
    palette: &SemanticPalette,
) {
    let anchor = rect.left_bottom() + egui::vec2(12.0, -12.0);
    egui::Area::new(ui.id().with("vol_hud_xyz_gizmo"))
        .order(egui::Order::Foreground)
        .pivot(egui::Align2::LEFT_BOTTOM)
        .fixed_pos(anchor)
        .constrain_to(rect)
        .interactable(false)
        .show(ui.ctx(), |ui| {
            hud_pill_frame(ui, palette).show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    let (gizmo_rect, _) =
                        ui.allocate_exact_size(egui::vec2(52.0, 44.0), egui::Sense::hover());
                    let painter = ui.painter_at(gizmo_rect);
                    let c = gizmo_rect.center() + egui::vec2(0.0, 2.0);
                    let len = 16.0_f32;
                    for (axis_vec, color, tag) in [
                        ([1.0, 0.0, 0.0], palette.axis_x, "X"),
                        ([0.0, 1.0, 0.0], palette.axis_y, "Y"),
                        ([0.0, 0.0, 1.0], palette.axis_z, "Z"),
                    ] {
                        let rot = tool.config.camera.rotate_to_camera_space(&axis_vec);
                        let tip = c + egui::vec2((rot[0] as f32) * len, -(rot[1] as f32) * len);
                        painter.line_segment([c, tip], egui::Stroke::new(1.8_f32, color));
                        painter.text(
                            tip + egui::vec2((rot[0] as f32) * 5.0, -(rot[1] as f32) * 5.0),
                            egui::Align2::CENTER_CENTER,
                            tag,
                            egui::FontId::monospace(9.0),
                            color,
                        );
                    }
                    painter.circle_filled(c, 2.2, palette.node_stroke);
                    ui.label(
                        egui::RichText::new("XYZ WORLD")
                            .monospace()
                            .size(8.5)
                            .color(palette.muted_text),
                    );
                });
            });
        });
}

fn draw_bottom_right_telemetry_pill(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    tool: &VolumetricFieldStudioTool,
    palette: &SemanticPalette,
) {
    let tris = (tool.engine.pos_mesh.indices.len() + tool.engine.neg_mesh.indices.len()) / 3;
    let dim = tool.active_voxel_dimension();
    let backend_tag = tool
        .webgpu_runner
        .adapter_status
        .hud_backend_label(tool.config.compute_backend);
    let step_ms = tool.webgpu_runner.last_step_ms;
    let status_dot = if tool.engine.diverged {
        palette.error
    } else {
        palette.success
    };
    let anchor = rect.right_bottom() + egui::vec2(-12.0, -12.0);
    egui::Area::new(ui.id().with("vol_hud_telemetry"))
        .order(egui::Order::Foreground)
        .pivot(egui::Align2::RIGHT_BOTTOM)
        .fixed_pos(anchor)
        .constrain_to(rect)
        .interactable(false)
        .show(ui.ctx(), |ui| {
            hud_pill_frame(ui, palette).show(ui, |ui| {
                ui.horizontal(|ui| {
                    let (dot_rect, _) =
                        ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                    ui.painter()
                        .circle_filled(dot_rect.center(), 3.5, status_dot);
                    ui.label(
                        egui::RichText::new(format!(
                            "Mesh: {tris} tris  |  Step: {step_ms:.1} ms ({backend_tag} {dim}³)"
                        ))
                        .monospace()
                        .size(10.5),
                    );
                });
            });
        });
}
