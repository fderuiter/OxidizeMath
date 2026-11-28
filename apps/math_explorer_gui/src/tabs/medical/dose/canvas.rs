//! Interactive `egui_plot` dose canvas, raster heatmap synthesis, sculpted contour overlays,
//! four draggable + keyboard-nudgeable coplanar gantry handles, and `SpatialBrushKit` sampling.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CanvasKeyTarget, DoseCalculationTool, DoseDisplayMode};
use crate::accessibility::{announce_live_status, register_canvas_accessibility};
use crate::app::SemanticPalette;
use crate::framework::{
    scoped_panel_id, BrushMode, BrushStroke, InteractionContext, InteractiveTool,
};
use crate::tabs::medical::imrt::contours::extract_mask_boundary_segments;
use crate::tabs::medical::imrt::{
    IMRT_BEAM_COUNT, IMRT_GRID_DIM, IMRT_OAR_CENTER, IMRT_OAR_RADIUS_CM, IMRT_PHANTOM_SIZE_CM,
    IMRT_TUMOR_CENTER, IMRT_TUMOR_RADIUS_CM, IMRT_VOXEL_COUNT,
};
use eframe::egui::{
    self, Color32, ColorImage, EventFilter, Key, Modifiers, Pos2, RichText, Stroke, TextureOptions,
};
use egui_plot::{Line, Plot, PlotImage, PlotPoint, PlotPoints, PlotUi, Points, Polygon, Text};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::medical::dose::kernel::{DoseKernel, ExponentialKernel};

/// Radius in phantom centimeters at which the 4 coplanar gantry handles orbit isocenter.
pub const GANTRY_HANDLE_RADIUS_CM: f64 = 8.6;
const GANTRY_RAY_INNER_CM: f64 = 5.4;
const GANTRY_HIT_RADIUS_CM: f64 = 1.15;

/// Synthesizes the point-source or active IMRT raster `ColorImage` for display and PNG export.
pub(crate) fn compute_dose_image_for_contrast(
    tool: &DoseCalculationTool,
    high_contrast: bool,
) -> ColorImage {
    if tool.imrt_mode {
        return compute_imrt_image(tool, high_contrast);
    }
    let (width, height) = (tool.width.max(1), tool.height.max(1));
    let mut pixels = vec![Color32::BLACK; width * height];
    let kernel = ExponentialKernel::new(tool.amplitude, tool.beta);
    let (dx, dy) = (
        IMRT_PHANTOM_SIZE_CM / width as f64,
        IMRT_PHANTOM_SIZE_CM / height as f64,
    );
    let max_dose = kernel.value_at(0.1).unwrap_or(10.0);
    for y in 0..height {
        let py = (y as f64 - height as f64 / 2.0) * dy;
        for x in 0..width {
            let px = (x as f64 - width as f64 / 2.0) * dx;
            let r = px.hypot(py);
            let dose = if r < 1e-3 {
                max_dose
            } else {
                kernel.value_at(r).unwrap_or(0.0)
            };
            let rgb = map_intensity((dose / max_dose).clamp(0.0, 1.0), high_contrast);
            pixels[flatten_2d_index(x, y, width)] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
        }
    }
    ColorImage::new([width, height], pixels)
}

pub(super) fn active_imrt_dose_slice(tool: &DoseCalculationTool) -> &[f64] {
    if !tool.gating_4d_enabled {
        return tool.imrt_workspace.dose_vec.as_slice();
    }
    match tool.display_mode {
        DoseDisplayMode::StaticPlan => tool.imrt_workspace.dose_vec.as_slice(),
        DoseDisplayMode::Gated4d if tool.imrt_workspace.motion.gated_steps > 0 => {
            tool.imrt_workspace.gated_dose_vec.as_slice()
        }
        DoseDisplayMode::Ungated4d if tool.imrt_workspace.motion.ungated_steps > 0 => {
            tool.imrt_workspace.ungated_dose_vec.as_slice()
        }
        DoseDisplayMode::Gated4d | DoseDisplayMode::Ungated4d => {
            tool.imrt_workspace.dose_vec.as_slice()
        }
    }
}

fn compute_imrt_image(tool: &DoseCalculationTool, high_contrast: bool) -> ColorImage {
    let mut pixels = vec![Color32::BLACK; IMRT_VOXEL_COUNT];
    let scale_gy = (tool.prescription_gy * 1.25).max(10.0);
    let dose_slice = active_imrt_dose_slice(tool);
    for y in 0..IMRT_GRID_DIM {
        let img_y = IMRT_GRID_DIM - 1 - y;
        for x in 0..IMRT_GRID_DIM {
            let dose = dose_slice[flatten_2d_index(x, y, IMRT_GRID_DIM)];
            let rgb = map_intensity((dose / scale_gy).clamp(0.0, 1.0), high_contrast);
            pixels[flatten_2d_index(x, img_y, IMRT_GRID_DIM)] =
                Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
        }
    }
    ColorImage::new([IMRT_GRID_DIM, IMRT_GRID_DIM], pixels)
}

fn map_intensity(intensity: f64, high_contrast: bool) -> [u8; 3] {
    if high_contrast {
        heatmap_color_high_contrast(intensity)
    } else {
        heatmap_color(intensity)
    }
}

/// Renders the interactive `egui_plot` dose heatmap, contour boundaries, gantry handles, and brush reticle.
pub fn render_dose_canvas(tool: &mut DoseCalculationTool, ui: &mut egui::Ui, ctx: &egui::Context) {
    if tool.texture.is_none() {
        let image = compute_dose_image_for_contrast(tool, tool.high_contrast);
        let filter = if tool.imrt_mode {
            TextureOptions::NEAREST
        } else {
            TextureOptions::LINEAR
        };
        tool.texture = Some(ctx.load_texture("dose_field", image, filter));
    }
    let Some(texture) = tool.texture.clone() else {
        return;
    };
    let palette = SemanticPalette::from_visuals(ui.visuals());
    let allow_drag =
        !tool.imrt_mode || (!tool.brush_kit.is_sculpting() && tool.dragging_beam.is_none());
    let mut pending_strokes: Vec<BrushStroke> = Vec::new();
    let mut gantry_changed = false;
    let mut plot_resp_out: Option<egui::Response> = None;

    Plot::new(scoped_panel_id(ui, "dose_heatmap"))
        .data_aspect(1.0)
        .view_aspect(1.0)
        .allow_drag(allow_drag)
        .show_grid(false)
        .show_axes([true, true])
        .show(ui, |plot_ui| {
            plot_ui.image(PlotImage::new(
                "dose_field",
                texture.id(),
                PlotPoint::new(0.0, 0.0),
                [IMRT_PHANTOM_SIZE_CM as f32, IMRT_PHANTOM_SIZE_CM as f32],
            ));
            if tool.imrt_mode {
                draw_contour_overlays(tool, &palette, plot_ui);
                gantry_changed |= sample_gantry_pointer(tool, plot_ui, ctx);
                gantry_changed |= sample_gantry_keyboard(tool, plot_ui, ctx);
                draw_gantry_handles(tool, &palette, plot_ui);
                sample_brush_if_active(tool, plot_ui, &mut pending_strokes);
            }
            tool.callout_manager.render_on_plot(plot_ui);
            plot_resp_out = Some(plot_ui.response().clone());
        });

    if let Some(resp) = plot_resp_out {
        update_focus_and_a11y(tool, ui, &resp);
        for stroke in pending_strokes {
            tool.on_brush(&InteractionContext::from_brush_stroke(&resp, stroke));
        }
    }
    if gantry_changed {
        tool.imrt_workspace.reset_4d_dose();
        tool.refresh_model();
    }
}

fn update_focus_and_a11y(tool: &mut DoseCalculationTool, ui: &mut egui::Ui, resp: &egui::Response) {
    if resp.clicked() || tool.dragging_beam.is_some() {
        resp.request_focus();
    }
    if resp.has_focus() {
        tool.focused_canvas = Some(resp.id);
    } else if tool.focused_canvas == Some(resp.id) {
        tool.focused_canvas = None;
    }
    let summary = if tool.imrt_mode {
        format!(
            "IMRT phantom ({}) — Beam {} at {:.0} deg, PTV mean {:.1} Gy, OAR max {:.1} Gy",
            tool.display_mode.label(),
            tool.selected_beam + 1,
            tool.beam_angles_deg[tool.selected_beam],
            tool.imrt_workspace.summary.tumor_mean_gy,
            tool.imrt_workspace.summary.oar_max_gy.0
        )
    } else {
        format!(
            "Point-source exponential dose kernel A={:.1}, beta={:.2}",
            tool.amplitude, tool.beta
        )
    };
    register_canvas_accessibility(
        ui,
        resp,
        egui::accesskit::Role::Canvas,
        "Radiotherapy dose and contouring canvas",
        Some(&summary),
    );
}

/// Computes `(x_cm, y_cm)` on the perimeter ring for coplanar gantry angle `angle_deg`.
#[must_use]
pub fn gantry_handle_pos_cm(angle_deg: f64) -> [f64; 2] {
    let rad = angle_deg.to_radians();
    [
        GANTRY_HANDLE_RADIUS_CM * rad.cos(),
        GANTRY_HANDLE_RADIUS_CM * rad.sin(),
    ]
}

fn draw_gantry_handles(
    tool: &DoseCalculationTool,
    palette: &SemanticPalette,
    plot_ui: &mut PlotUi<'_>,
) {
    for b in 0..IMRT_BEAM_COUNT {
        let angle = tool.beam_angles_deg[b];
        let outer = gantry_handle_pos_cm(angle);
        let rad = angle.to_radians();
        let inner = [
            GANTRY_RAY_INNER_CM * rad.cos(),
            GANTRY_RAY_INNER_CM * rad.sin(),
        ];
        let is_sel = tool.selected_beam == b || tool.dragging_beam == Some(b);
        let color = if is_sel { palette.accent } else { palette.info };
        plot_ui.line(
            Line::new(
                format!("Beam {} Axis", b + 1),
                PlotPoints::from(vec![outer, inner]),
            )
            .color(color)
            .width(if is_sel { 2.8_f32 } else { 1.8_f32 }),
        );
        plot_ui.points(
            Points::new(format!("Beam {} Handle", b + 1), vec![outer])
                .radius(if is_sel { 7.5_f32 } else { 5.8_f32 })
                .color(color),
        );
        let lbl = RichText::new(format!("B{} {:.0}°", b + 1, angle))
            .size(11.0)
            .color(color);
        let pt = PlotPoint::new(outer[0] * 1.11, outer[1] * 1.11);
        plot_ui.text(Text::new(format!("Beam {} Label", b + 1), pt, lbl).highlight(is_sel));
    }
}

fn sample_gantry_pointer(
    tool: &mut DoseCalculationTool,
    plot_ui: &PlotUi<'_>,
    ctx: &egui::Context,
) -> bool {
    let (pressed, down, released) = ctx.input(|i| {
        (
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            i.pointer.primary_released(),
        )
    });
    let resp = plot_ui.response();
    if pressed && (resp.hovered() || resp.dragged()) {
        if let Some(coord) = plot_ui.pointer_coordinate() {
            if let Some(hit) = hit_test_gantry_handle(&tool.beam_angles_deg, [coord.x, coord.y]) {
                tool.selected_beam = hit;
                tool.dragging_beam = Some(hit);
                tool.canvas_key_target = CanvasKeyTarget::Gantry;
            }
        }
    }
    let mut changed = false;
    if let Some(b) = tool.dragging_beam {
        if down {
            if let Some(coord) = plot_ui.pointer_coordinate() {
                if coord.x.hypot(coord.y) > 0.25 {
                    let deg = coord.y.atan2(coord.x).to_degrees().rem_euclid(360.0);
                    if (tool.beam_angles_deg[b] - deg).abs() > 1.0e-3 {
                        tool.beam_angles_deg[b] = (deg * 2.0).round() * 0.5;
                        changed = true;
                    }
                }
            }
        }
        if released || !down {
            let deg = tool.beam_angles_deg[b];
            tool.dragging_beam = None;
            announce_live_status(ctx, format!("Beam {} dragged to {:.1} degrees", b + 1, deg));
        }
    }
    changed
}

fn hit_test_gantry_handle(angles_deg: &[f64; IMRT_BEAM_COUNT], pt_cm: [f64; 2]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (b, &deg) in angles_deg.iter().enumerate() {
        let h = gantry_handle_pos_cm(deg);
        let dist = (pt_cm[0] - h[0]).hypot(pt_cm[1] - h[1]);
        if dist <= GANTRY_HIT_RADIUS_CM && best.is_none_or(|(_, d)| dist < d) {
            best = Some((b, dist));
        }
    }
    best.map(|(b, _)| b)
}

fn sample_gantry_keyboard(
    tool: &mut DoseCalculationTool,
    plot_ui: &PlotUi<'_>,
    ctx: &egui::Context,
) -> bool {
    let resp = plot_ui.response();
    if !resp.has_focus() {
        return false;
    }
    // Lock ONLY arrow keys; never trap Tab, Shift+Tab, or Escape.
    ctx.memory_mut(|m| {
        m.set_focus_lock_filter(
            resp.id,
            EventFilter {
                tab: false,
                horizontal_arrows: true,
                vertical_arrows: true,
                escape: false,
            },
        );
    });
    let (changed, announcement) = ctx.input_mut(|i| consume_gantry_key_inputs(tool, i));
    if let Some(msg) = announcement {
        announce_live_status(ctx, msg);
        ctx.request_repaint();
    }
    changed
}

fn consume_gantry_key_inputs(
    tool: &mut DoseCalculationTool,
    i: &mut egui::InputState,
) -> (bool, Option<String>) {
    if i.modifiers.ctrl || i.modifiers.command || i.modifiers.alt {
        return (false, None);
    }
    let mut changed = false;
    let mut msg = None;
    if i.consume_key(Modifiers::NONE, Key::G) {
        tool.canvas_key_target = CanvasKeyTarget::Gantry;
        msg = Some(format!(
            "Gantry angle keyboard control active on Beam {}",
            tool.selected_beam + 1
        ));
    } else if i.consume_key(Modifiers::NONE, Key::B) && tool.active_challenge.is_none() {
        tool.canvas_key_target = CanvasKeyTarget::Brush;
        msg = Some("Contour brush keyboard control active".to_string());
    }
    for (idx, key) in [Key::Num1, Key::Num2, Key::Num3, Key::Num4]
        .into_iter()
        .enumerate()
    {
        if i.consume_key(Modifiers::NONE, key) {
            tool.selected_beam = idx;
            tool.canvas_key_target = CanvasKeyTarget::Gantry;
            msg = Some(format!(
                "Selected Beam {} at {:.0} degrees",
                idx + 1,
                tool.beam_angles_deg[idx]
            ));
        }
    }
    if tool.canvas_key_target == CanvasKeyTarget::Gantry || tool.active_challenge.is_some() {
        if let Some(delta) = consume_gantry_arrow_delta(i) {
            let b = tool.selected_beam;
            let next = (tool.beam_angles_deg[b] + delta).rem_euclid(360.0);
            tool.beam_angles_deg[b] = next;
            changed = true;
            msg = Some(format!("Beam {} gantry angle {:.0} degrees", b + 1, next));
        }
    }
    (changed, msg)
}

fn consume_gantry_arrow_delta(i: &mut egui::InputState) -> Option<f64> {
    for (key, base) in [
        (Key::ArrowLeft, -1.0),
        (Key::ArrowDown, -1.0),
        (Key::ArrowRight, 1.0),
        (Key::ArrowUp, 1.0),
    ] {
        if i.consume_key(Modifiers::SHIFT, key) {
            return Some(5.0 * base);
        }
        if i.consume_key(Modifiers::NONE, key) {
            return Some(base);
        }
    }
    None
}

fn sample_brush_if_active(
    tool: &mut DoseCalculationTool,
    plot_ui: &mut PlotUi<'_>,
    out_strokes: &mut Vec<BrushStroke>,
) {
    if tool.active_challenge.is_some() || tool.dragging_beam.is_some() {
        return;
    }
    if tool.canvas_key_target == CanvasKeyTarget::Brush {
        let half = 0.5 * IMRT_PHANTOM_SIZE_CM;
        out_strokes.extend(tool.brush_kit.sample_plot(
            plot_ui,
            [-half, -half],
            [half, half],
            Some(BrushMode::Erase),
        ));
    }
}

fn draw_contour_overlays(
    tool: &DoseCalculationTool,
    palette: &SemanticPalette,
    plot_ui: &mut PlotUi<'_>,
) {
    if !tool.imrt_workspace.custom_contours {
        plot_ui.polygon(
            Polygon::new(
                "PTV Target (D_Rx)",
                circle_ring(IMRT_TUMOR_CENTER, IMRT_TUMOR_RADIUS_CM),
            )
            .fill_color(Color32::TRANSPARENT)
            .stroke(Stroke::new(2.0_f32, palette.success)),
        );
        plot_ui.polygon(
            Polygon::new(
                "OAR (D_lim)",
                circle_ring(IMRT_OAR_CENTER, IMRT_OAR_RADIUS_CM),
            )
            .fill_color(Color32::TRANSPARENT)
            .stroke(Stroke::new(2.0_f32, palette.error)),
        );
    } else {
        draw_custom_mask_edges(
            &tool.imrt_workspace.ptv_mask,
            0.0,
            Stroke::new(2.0_f32, palette.success),
            plot_ui,
        );
        draw_custom_mask_edges(
            &tool.imrt_workspace.oar_mask,
            0.0,
            Stroke::new(2.0_f32, palette.error),
            plot_ui,
        );
    }
    if tool.gating_4d_enabled && tool.imrt_workspace.motion.total_4d_steps > 0 {
        let shift = tool.imrt_workspace.motion.latest_shift_cm;
        if shift.abs() > 0.03 {
            plot_ui.polygon(
                Polygon::new(
                    "4D Respiratory OAR Excursion",
                    circle_ring(
                        [IMRT_OAR_CENTER[0] + shift, IMRT_OAR_CENTER[1]],
                        IMRT_OAR_RADIUS_CM,
                    ),
                )
                .fill_color(Color32::TRANSPARENT)
                .stroke(Stroke::new(1.3_f32, palette.warning)),
            );
        }
    }
}

fn draw_custom_mask_edges(
    mask: &[bool; IMRT_VOXEL_COUNT],
    shift_x_cm: f64,
    stroke: Stroke,
    plot_ui: &mut PlotUi<'_>,
) {
    let segs = extract_mask_boundary_segments(mask, shift_x_cm);
    let transform = plot_ui.transform();
    let clipped = plot_ui
        .ctx()
        .layer_painter(egui::LayerId::new(
            egui::Order::Middle,
            plot_ui.response().id.with("custom_contour_edges"),
        ))
        .with_clip_rect(*transform.frame());
    for [a, b] in segs {
        let pa: Pos2 = transform.position_from_point(&PlotPoint::new(a[0], a[1]));
        let pb: Pos2 = transform.position_from_point(&PlotPoint::new(b[0], b[1]));
        clipped.line_segment([pa, pb], stroke);
    }
}

fn circle_ring(center: [f64; 2], radius: f64) -> Vec<[f64; 2]> {
    (0..32)
        .map(|k| {
            let a = std::f64::consts::TAU * (k as f64) / 32.0;
            [center[0] + radius * a.cos(), center[1] + radius * a.sin()]
        })
        .collect()
}

// [cite:graph_parameters_rust]
