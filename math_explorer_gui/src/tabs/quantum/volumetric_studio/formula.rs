#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Custom 3D implicit formula (`f(x, y, z) = c`), TPMS shell thickening, parameter sliders,
//! and zero-allocation `CompiledExpr::eval_3d` voxel/splat sampling for [`super::VolumetricFieldStudioTool`].
// [cite:isosurface_extraction]
// [cite:gaussian_splatting]

use super::state::{
    VolumetricStudioConfig, DEFAULT_IMPLICIT_BOUNDS, DEFAULT_IMPLICIT_EXPR,
    IMPLICIT_DOMAIN_MAX_BOUND, IMPLICIT_DOMAIN_MIN_BOUND, IMPLICIT_DOMAIN_MIN_SPAN,
    MAX_IMPLICIT_WALL_THICKNESS, MIN_IMPLICIT_WALL_THICKNESS,
};
use super::VolumetricFieldStudioTool;
use crate::framework::{bounded_drag_value, enforce_ordered_drag_range, MathExpressionInputBar};
use crate::volumetric_scene::{rotation_from_normal, VolumeCutFilter, VolumetricSplatWorkspace};
use eframe::egui;
use math_commons::ast::{parse, CompiledExpr};
use math_explorer::ai::gaussian_splatting::Gaussian3D;
use math_explorer::applied::isosurface::{CentralDifferenceEstimator, VoxelGrid};
use math_explorer::math_kernel::types::flatten_3d_index;
use nalgebra::{Point3, Vector3};

/// `(Label, Expression, DefaultShellEnabled)` template definitions for 3D implicit surfaces and CSG solids.
pub const IMPLICIT_TEMPLATES: &[(&str, &str, bool)] = &[
    (
        "Gyroid TPMS",
        "sin(k * x) * cos(k * y) + sin(k * y) * cos(k * z) + sin(k * z) * cos(k * x)",
        true,
    ),
    (
        "Schwarz P",
        "cos(k * x) + cos(k * y) + cos(k * z)",
        true,
    ),
    (
        "Schwarz D",
        "sin(k * x) * sin(k * y) * sin(k * z) + sin(k * x) * cos(k * y) * cos(k * z) + cos(k * x) * sin(k * y) * cos(k * z) + cos(k * x) * cos(k * y) * sin(k * z)",
        true,
    ),
    (
        "Torus SDF",
        "sqrt((sqrt(x^2 + y^2) - R)^2 + z^2) - r",
        false,
    ),
    (
        "CSG Union & Bore",
        "max(min(sqrt(x^2 + y^2 + z^2) - R, max(max(abs(x), abs(y)), abs(z)) - b), r - sqrt(x^2 + y^2))",
        false,
    ),
    (
        "Twisted Torus",
        "sqrt((sqrt(x^2 + y^2) - R - a * cos(3 * atan2(y, x) + k * clamp(z, -2, 2)))^2 + z^2) - r",
        false,
    ),
];

/// Constructs the `MathExpressionInputBar` configured for `x, y, z` and up to 6 free parameter sliders.
#[must_use]
pub fn build_implicit_formula_bar(initial_expr: &str) -> MathExpressionInputBar {
    let presets = IMPLICIT_TEMPLATES
        .iter()
        .map(|&(name, expr, _)| (name, expr))
        .collect();
    MathExpressionInputBar::new("f(x,y,z)", initial_expr, presets)
        .with_axes(&["x", "y", "z"])
        .with_param_range(-4.0..=4.0, 1.0)
        .with_param_spec("k", "Wave k", 0.4..=4.5, 2.0)
        .with_param_spec("R", "Major R", 0.5..=2.2, 1.35)
        .with_param_spec("r", "Minor r", 0.2..=1.4, 0.55)
        .with_param_spec("a", "Modulation a", -1.5..=1.5, 0.25)
        .with_param_spec("b", "Half-Extent b", 0.4..=2.2, 1.25)
        .with_param_spec("c", "Offset c", -2.0..=2.0, 0.0)
}

/// Compiles the default Gyroid TPMS expression as an infallible fallback for [`super::engine::VolumetricEngine`].
#[must_use]
pub fn default_compiled_gyroid() -> CompiledExpr {
    let ast = parse(DEFAULT_IMPLICIT_EXPR).expect("default gyroid parses");
    CompiledExpr::compile_with_vars_and_params(&ast, &["x", "y", "z"], &["k"])
        .expect("default gyroid compiles")
}

/// Evaluates `compiled.eval_3d` across `grid` in-place with zero heap allocations.
///
/// Stores `-s(x, y, z)` so interior points satisfy `val >= 0` and exterior points satisfy `val < 0`,
/// where `s = |f(x,y,z) - c| - 0.5 * wall_thickness` when `cfg.shell_enabled` is true and
/// `s = f(x,y,z) - c` otherwise.
pub fn populate_implicit_voxel_grid(
    grid: &mut VoxelGrid,
    cfg: &VolumetricStudioConfig,
    compiled: &CompiledExpr,
    params: &[f64],
) {
    let (w, h, d) = (grid.width(), grid.height(), grid.depth());
    if w < 2 || h < 2 || d < 2 {
        return;
    }
    let inv_x = 1.0 / ((w - 1) as f64);
    let inv_y = 1.0 / ((h - 1) as f64);
    let inv_z = 1.0 / ((d - 1) as f64);
    let span_x = cfg.x_bounds[1] - cfg.x_bounds[0];
    let span_y = cfg.y_bounds[1] - cfg.y_bounds[0];
    let span_z = cfg.z_bounds[1] - cfg.z_bounds[0];
    let half_wall = 0.5 * cfg.wall_thickness;
    let level = cfg.implicit_level;
    let shell = cfg.shell_enabled;
    let buf = grid.data_mut();

    for z in 0..d {
        let fz = cfg.z_bounds[0] + (z as f64) * inv_z * span_z;
        for y in 0..h {
            let fy = cfg.y_bounds[0] + (y as f64) * inv_y * span_y;
            for x in 0..w {
                let fx = cfg.x_bounds[0] + (x as f64) * inv_x * span_x;
                let raw = compiled.eval_3d(fx, fy, fz, params);
                let signed_dist = if shell {
                    (raw - level).abs() - half_wall
                } else {
                    raw - level
                };
                let stored = if signed_dist.is_finite() {
                    (-signed_dist).clamp(-20.0, 20.0) as f32
                } else {
                    -10.0_f32
                };
                let idx = flatten_3d_index(x, y, z, w, h);
                buf[idx] = stored;
            }
        }
    }
}

/// Extracts gradient-aligned `Gaussian3D` splats strictly inside or along the implicit solid/shell (`val >= -skin`).
pub fn extract_implicit_splats_into(
    grid: &VoxelGrid,
    cut: Option<VolumeCutFilter>,
    ws: &mut VolumetricSplatWorkspace,
) {
    ws.scene.gaussians.clear();
    ws.grad_magnitudes.clear();
    let cap = ws
        .scene
        .gaussians
        .capacity()
        .min(ws.grad_magnitudes.capacity());
    let (w, h, d) = (grid.width(), grid.height(), grid.depth());
    let total = w * h * d;
    if cap == 0 || total == 0 {
        return;
    }
    let mut pos_peak = 1e-4_f32;
    let mut occupied = 0usize;
    let vs = grid.voxel_size();
    let skin = -0.05 * vs.x;
    for &v in grid.data() {
        if v >= skin {
            occupied += 1;
            pos_peak = pos_peak.max(v.abs());
        }
    }
    if occupied == 0 {
        return;
    }
    let step = (((occupied / cap.max(1)).max(1) as f64).cbrt().ceil() as usize).max(1);
    let orig = grid.origin();
    let base_r = ((vs.x.max(vs.y).max(vs.z)) as f64) * (step as f64) * 0.68;

    for z in (0..d).step_by(step) {
        let wz = (orig.z + (z as f32) * vs.z) as f64;
        for y in (0..h).step_by(step) {
            let wy = (orig.y + (y as f32) * vs.y) as f64;
            for x in (0..w).step_by(step) {
                if ws.scene.gaussians.len() >= cap {
                    return;
                }
                let val = grid.get(x, y, z);
                if val < skin {
                    continue;
                }
                let wx = (orig.x + (x as f32) * vs.x) as f64;
                if cut.is_some_and(|c| [wx, wy, wz][c.axis.min(2)] > c.max_coord) {
                    continue;
                }
                push_implicit_voxel_splat(grid, (x, y, z), [wx, wy, wz], val, pos_peak, base_r, ws);
            }
        }
    }
}

fn push_implicit_voxel_splat(
    grid: &VoxelGrid,
    ijk: (usize, usize, usize),
    world: [f64; 3],
    val: f32,
    pos_peak: f32,
    base_r: f64,
    ws: &mut VolumetricSplatWorkspace,
) {
    let (normal, grad_mag) =
        CentralDifferenceEstimator.world_normal_and_magnitude(grid, ijk.0, ijk.1, ijk.2);
    let rel = (val.max(0.0) / pos_peak).clamp(0.0, 1.0) as f64;
    let z_norm = ((world[2] + 2.5) * 0.2).clamp(0.0, 1.0);
    let color = Vector3::new(
        0.18 + 0.22 * z_norm,
        0.66 + 0.26 * rel,
        0.92 - 0.20 * z_norm,
    );
    ws.scene.gaussians.push(Gaussian3D {
        mean: Point3::new(world[0], world[1], world[2]),
        scale: Vector3::new(base_r, base_r, base_r * 0.45),
        rotation: rotation_from_normal(normal),
        opacity: (0.24 + 0.64 * (1.0 - 0.4 * rel)).clamp(0.08, 0.92),
        color,
    });
    ws.grad_magnitudes
        .push((grad_mag as f64) / (pos_peak as f64));
}

/// Renders template selectors, `MathExpressionInputBar`, shell thickening, and `[x, y, z]` bounds.
pub(super) fn draw_implicit_controls(
    tool: &mut VolumetricFieldStudioTool,
    ui: &mut egui::Ui,
) -> bool {
    let mut changed = draw_template_buttons(tool, ui);
    let prev_raw = tool.formula_bar.raw_text.clone();
    let bar_changed = tool.formula_bar.ui(ui);
    if bar_changed && tool.formula_bar.is_valid() {
        if tool.formula_bar.raw_text != prev_raw {
            for &(_, expr, default_shell) in IMPLICIT_TEMPLATES {
                if tool.formula_bar.raw_text == expr {
                    tool.config.shell_enabled = default_shell;
                }
            }
        }
        tool.sync_valid_formula_to_config_and_engine();
        changed = true;
    }
    changed |= ui
        .checkbox(
            &mut tool.config.shell_enabled,
            "Thicken Sheet Surface (|f − c| ≤ ½w)",
        )
        .changed();
    if tool.config.shell_enabled {
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut tool.config.wall_thickness,
                    MIN_IMPLICIT_WALL_THICKNESS..=MAX_IMPLICIT_WALL_THICKNESS,
                )
                .text("Wall Thickness w"),
            )
            .changed();
    }
    changed |= ui
        .add(egui::Slider::new(&mut tool.config.implicit_level, -2.0..=2.0).text("Level Offset c"))
        .changed();
    egui::CollapsingHeader::new("3D Domain Bounds [x, y, z]")
        .default_open(false)
        .show(ui, |ui| {
            changed |= draw_axis_bounds_row(ui, "x:", &mut tool.config.x_bounds);
            changed |= draw_axis_bounds_row(ui, "y:", &mut tool.config.y_bounds);
            changed |= draw_axis_bounds_row(ui, "z:", &mut tool.config.z_bounds);
            if ui.small_button("Reset [-2.5, 2.5]³").clicked() {
                tool.config.x_bounds = DEFAULT_IMPLICIT_BOUNDS;
                tool.config.y_bounds = DEFAULT_IMPLICIT_BOUNDS;
                tool.config.z_bounds = DEFAULT_IMPLICIT_BOUNDS;
                changed = true;
            }
        });
    changed
}

fn draw_template_buttons(tool: &mut VolumetricFieldStudioTool, ui: &mut egui::Ui) -> bool {
    let mut clicked = false;
    ui.label("Implicit & TPMS Templates:");
    ui.horizontal_wrapped(|ui| {
        for &(label, expr, shell) in IMPLICIT_TEMPLATES {
            if ui.small_button(label).clicked() {
                tool.formula_bar.set_expression(expr);
                tool.config.shell_enabled = shell;
                tool.config.implicit_level = 0.0;
                tool.sync_valid_formula_to_config_and_engine();
                clicked = true;
            }
        }
    });
    clicked
}

fn draw_axis_bounds_row(ui: &mut egui::Ui, label: &str, bounds: &mut [f64; 2]) -> bool {
    let prev = *bounds;
    let mut min_changed = false;
    let mut max_changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        min_changed = ui
            .add(bounded_drag_value(
                &mut bounds[0],
                IMPLICIT_DOMAIN_MIN_BOUND..=IMPLICIT_DOMAIN_MAX_BOUND,
                0.1,
            ))
            .changed();
        ui.label("to");
        max_changed = ui
            .add(bounded_drag_value(
                &mut bounds[1],
                IMPLICIT_DOMAIN_MIN_BOUND..=IMPLICIT_DOMAIN_MAX_BOUND,
                0.1,
            ))
            .changed();
    });
    let (mut lo, mut hi) = (bounds[0], bounds[1]);
    enforce_ordered_drag_range(
        &mut lo,
        &mut hi,
        IMPLICIT_DOMAIN_MIN_BOUND,
        IMPLICIT_DOMAIN_MAX_BOUND,
        IMPLICIT_DOMAIN_MIN_SPAN,
        min_changed,
        max_changed,
    );
    *bounds = [lo, hi];
    *bounds != prev
}
