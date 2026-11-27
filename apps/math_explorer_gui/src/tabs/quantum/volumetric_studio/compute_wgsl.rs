#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Standalone WebGPU `@compute @workgroup_size(4, 4, 4)` and `@fragment` WGSL code generation
//! plus collapsible `GPU Shader Preview` drawer for [`super::VolumetricFieldStudioTool`].

use super::implicit::default_implicit_param_value;
use super::state::{
    FieldSourceMode, ImplicitClosureMode, QuantumObservableChoice, VolumetricStudioConfig,
    STUDIO_VIEWPORT_EXTENT,
};
use crate::app::SemanticPalette;
use eframe::egui;
use math_commons::ast::{
    emit_wgsl_3d_compute_kernel, parse as parse_expr, CompiledExpr, ComputeKernelConfig,
};

/// Active WGSL stage displayed in the sidebar `GPU Shader Preview` drawer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShaderPreviewKind {
    /// `@compute @workgroup_size(4, 4, 4)` 3D voxel evaluation/PDE kernel.
    #[default]
    ComputeWgsl,
    /// `@fragment` raymarch / volumetric slice shader preview.
    FragmentWgsl,
}

/// Emits a standalone WebGPU `@compute @workgroup_size(4, 4, 4)` WGSL kernel calibrated to `cfg`.
pub fn build_volumetric_compute_wgsl(cfg: &VolumetricStudioConfig) -> Result<String, String> {
    match cfg.source {
        FieldSourceMode::TuringMorphogenesis => Ok(emit_turing_3d_compute_wgsl(cfg)),
        FieldSourceMode::QuantumSuperposition => Ok(emit_quantum_3d_compute_wgsl(cfg)),
        FieldSourceMode::CustomImplicit => emit_implicit_cad_compute_wgsl(cfg),
        FieldSourceMode::CustomImplicitAst => emit_custom_ast_compute_wgsl(cfg),
    }
}

/// Emits a standalone WebGPU `@fragment` volumetric preview WGSL shader calibrated to `cfg`.
pub fn build_volumetric_fragment_wgsl(cfg: &VolumetricStudioConfig) -> Result<String, String> {
    let dim = cfg.grid_resolution.max(16);
    let iso = cfg.iso_threshold.clamp(0.002, 0.8);
    let ext = STUDIO_VIEWPORT_EXTENT;
    let label = cfg.source.label();
    Ok(format!(
        "// OxidizeMath Volumetric Field Studio — Fragment Raymarch Preview ({label})\n\
struct ViewUniforms {{ resolution: vec2<f32>, iso_level: f32, grid_dim: u32 }};\n\
@group(0) @binding(0) var<uniform> u_view: ViewUniforms;\n\
@group(0) @binding(1) var<storage, read> u_voxels: array<f32>;\n\
const STUDIO_EXTENT: f32 = {ext:.4};\n\
const DEFAULT_DIM: u32 = {dim}u;\n\
const DEFAULT_ISO: f32 = {iso:.5};\n\
@fragment\n\
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {{\n\
    let res = max(u_view.resolution, vec2<f32>(1.0, 1.0));\n\
    let uv = (frag_coord.xy / res) * 2.0 - vec2<f32>(1.0, 1.0);\n\
    let d = select(DEFAULT_DIM, u_view.grid_dim, u_view.grid_dim > 1u);\n\
    let gx = u32(clamp((uv.x * 0.5 + 0.5) * f32(d - 1u), 0.0, f32(d - 1u)));\n\
    let gy = u32(clamp((uv.y * 0.5 + 0.5) * f32(d - 1u), 0.0, f32(d - 1u)));\n\
    let idx = ((d / 2u) * d + gy) * d + gx;\n\
    let val = u_voxels[idx];\n\
    let tau = select(DEFAULT_ISO, u_view.iso_level, u_view.iso_level > 0.0);\n\
    let mag = clamp(abs(val) / tau, 0.0, 1.0);\n\
    let col = select(vec3<f32>(0.97, 0.46, 0.24), vec3<f32>(0.23, 0.67, 0.98), val >= 0.0);\n\
    return vec4<f32>(col * mag, 1.0);\n\
}}\n"
    ))
}

fn emit_turing_3d_compute_wgsl(cfg: &VolumetricStudioConfig) -> String {
    let dim = cfg.grid_resolution.max(16);
    format!(
        "// OxidizeMath WebGPU 3D Schnakenberg Reaction-Diffusion Compute Kernel ({dim}^3)\n\
struct TuringParams {{ dims: vec3<u32>, dt: f32, du: f32, dv: f32, rate_a: f32, rate_b: f32 }};\n\
@group(0) @binding(0) var<uniform> params: TuringParams;\n\
@group(0) @binding(1) var<storage, read> state_in: array<vec2<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> state_out: array<vec2<f32>>;\n\
const DEFAULT_DIM: u32 = {dim}u;\n\
const DEFAULT_DT: f32 = {:.6};\n\
const DEFAULT_DU: f32 = {:.6};\n\
const DEFAULT_DV: f32 = {:.6};\n\
const DEFAULT_A: f32 = {:.6};\n\
const DEFAULT_B: f32 = {:.6};\n\
fn flat_idx(x: u32, y: u32, z: u32, d: vec3<u32>) -> u32 {{ return (z * d.y + y) * d.x + x; }}\n\
@compute @workgroup_size(4, 4, 4)\n\
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
    if (gid.x >= params.dims.x || gid.y >= params.dims.y || gid.z >= params.dims.z) {{ return; }}\n\
    let d = params.dims;\n\
    let idx = flat_idx(gid.x, gid.y, gid.z, d);\n\
    let xm = select(gid.x - 1u, d.x - 1u, gid.x == 0u);\n\
    let xp = select(gid.x + 1u, 0u, gid.x + 1u == d.x);\n\
    let ym = select(gid.y - 1u, d.y - 1u, gid.y == 0u);\n\
    let yp = select(gid.y + 1u, 0u, gid.y + 1u == d.y);\n\
    let zm = select(gid.z - 1u, d.z - 1u, gid.z == 0u);\n\
    let zp = select(gid.z + 1u, 0u, gid.z + 1u == d.z);\n\
    let c = state_in[idx];\n\
    let lap = state_in[flat_idx(xm, gid.y, gid.z, d)] + state_in[flat_idx(xp, gid.y, gid.z, d)]\n\
            + state_in[flat_idx(gid.x, ym, gid.z, d)] + state_in[flat_idx(gid.x, yp, gid.z, d)]\n\
            + state_in[flat_idx(gid.x, gid.y, zm, d)] + state_in[flat_idx(gid.x, gid.y, zp, d)] - 6.0 * c;\n\
    // 7-point 3D Laplacian + Schnakenberg kinetics: du/dt = D_u * lap_u + a - u + u^2 * v\n\
    let u2v = c.x * c.x * c.y;\n\
    let du = params.du * lap.x + (params.rate_a - c.x + u2v);\n\
    let dv = params.dv * lap.y + (params.rate_b - u2v);\n\
    state_out[idx] = clamp(c + params.dt * vec2<f32>(du, dv), vec2<f32>(0.0), vec2<f32>(12.0));\n\
}}\n",
        cfg.turing_dt, cfg.turing_du, cfg.turing_dv, cfg.turing_a, cfg.turing_b
    )
}

fn emit_quantum_3d_compute_wgsl(cfg: &VolumetricStudioConfig) -> String {
    let dim = cfg.grid_resolution.max(16);
    let ext = cfg.physical_quantum_extent();
    let prob_mode = u32::from(cfg.observable == QuantumObservableChoice::ProbabilityDensity);
    format!(
        "// OxidizeMath WebGPU 3D Hydrogenic Superposition Compute Kernel ({dim}^3)\n\
struct OrbitalParams {{ dims: vec3<u32>, time: f32, box_half_extent: f32, mix_c2: f32, rel_phase: f32, prob_density: u32 }};\n\
@group(0) @binding(0) var<uniform> params: OrbitalParams;\n\
@group(0) @binding(1) var<storage, read> psi_basis: array<vec2<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> field_out: array<f32>;\n\
const STATE_1: vec3<i32> = vec3<i32>({}, {}, {});\n\
const STATE_2: vec3<i32> = vec3<i32>({}, {}, {});\n\
const DEFAULT_EXTENT: f32 = {ext:.4};\n\
const DEFAULT_PROB_MODE: u32 = {prob_mode}u;\n\
@compute @workgroup_size(4, 4, 4)\n\
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
    if (gid.x >= params.dims.x || gid.y >= params.dims.y || gid.z >= params.dims.z) {{ return; }}\n\
    let idx = (gid.z * params.dims.y + gid.y) * params.dims.x + gid.x;\n\
    let basis = psi_basis[idx];\n\
    let w2 = clamp(params.mix_c2, 0.0, 1.0);\n\
    let c1 = sqrt(1.0 - w2);\n\
    let c2 = sqrt(w2);\n\
    let e1 = -0.5 / f32(max(STATE_1.x * STATE_1.x, 1));\n\
    let e2 = -0.5 / f32(max(STATE_2.x * STATE_2.x, 1));\n\
    let p1 = -e1 * params.time;\n\
    let p2 = -e2 * params.time + params.rel_phase;\n\
    let re = c1 * basis.x * cos(p1) + c2 * basis.y * cos(p2);\n\
    let im = c1 * basis.x * sin(p1) + c2 * basis.y * sin(p2);\n\
    // Hydrogenic superposition Re(Psi) or Born density |Psi|^2\n\
    field_out[idx] = select(re, re * re + im * im, params.prob_density != 0u);\n\
}}\n",
        cfg.n1, cfg.l1, cfg.m1, cfg.n2, cfg.l2, cfg.m2
    )
}

fn emit_implicit_cad_compute_wgsl(cfg: &VolumetricStudioConfig) -> Result<String, String> {
    let formula = strip_optional_rhs(&cfg.implicit_expr);
    let base_expr = parse_expr(formula).map_err(|e| format!("Implicit AST error: {e}"))?;
    let level_lit = format!("({formula}) - ({:.6})", cfg.implicit_level);
    let wrapped = match cfg.implicit_closure {
        ImplicitClosureMode::HalfSpaceSolid => format!("-({level_lit})"),
        ImplicitClosureMode::ThickenedWall => {
            let half_w = 0.5 * cfg.wall_thickness;
            format!("({half_w:.6}) - abs({level_lit})")
        }
    };
    let final_expr = parse_expr(&wrapped).unwrap_or(base_expr);
    let compiled = CompiledExpr::compile_with_vars(&final_expr, &["x", "y", "z", "t"])
        .map_err(|e| e.to_string())?;
    let dim = cfg.grid_resolution.max(2) as u32;
    let mut kernel_cfg = ComputeKernelConfig {
        grid_dims: [dim, dim, dim],
        workgroup_size: [4, 4, 4],
        x_bounds: cfg.x_bounds,
        y_bounds: cfg.y_bounds,
        z_bounds: cfg.z_bounds,
        params: Vec::with_capacity(compiled.param_names().len()),
    };
    for name in compiled.param_names() {
        let val = cfg
            .implicit_params
            .get(name)
            .copied()
            .unwrap_or_else(|| default_implicit_param_value(name));
        kernel_cfg.params.push((name.clone(), val));
    }
    emit_wgsl_3d_compute_kernel(&final_expr, &kernel_cfg).map_err(|e| e.to_string())
}

fn emit_custom_ast_compute_wgsl(cfg: &VolumetricStudioConfig) -> Result<String, String> {
    let expr = parse_expr(&cfg.custom_expr).map_err(|e| format!("Custom 3D AST error: {e}"))?;
    let compiled =
        CompiledExpr::compile_with_vars(&expr, &["x", "y", "z", "t"]).map_err(|e| e.to_string())?;
    let dim = cfg.grid_resolution.max(2) as u32;
    let mut kernel_cfg = ComputeKernelConfig::new_3d(dim, STUDIO_VIEWPORT_EXTENT);
    for name in compiled.param_names() {
        let val = cfg.custom_params.get(name).copied().unwrap_or(1.0);
        kernel_cfg.params.push((name.clone(), val));
    }
    emit_wgsl_3d_compute_kernel(&expr, &kernel_cfg).map_err(|e| e.to_string())
}

fn strip_optional_rhs(raw: &str) -> &str {
    match raw.split_once('=') {
        Some((lhs, _)) => lhs.trim(),
        None => raw.trim(),
    }
}

/// Renders the collapsible `GPU Shader Preview` drawer (`Compute WGSL` / `Fragment WGSL` + 6-row snippet).
pub fn draw_gpu_shader_preview_section(
    ui: &mut egui::Ui,
    cfg: &VolumetricStudioConfig,
    preview_kind: &mut ShaderPreviewKind,
) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    egui::CollapsingHeader::new("GPU Shader Preview")
        .default_open(true)
        .show(ui, |ui| {
            let wgsl_res = match *preview_kind {
                ShaderPreviewKind::ComputeWgsl => build_volumetric_compute_wgsl(cfg),
                ShaderPreviewKind::FragmentWgsl => build_volumetric_fragment_wgsl(cfg),
            };
            ui.horizontal_wrapped(|ui| {
                ui.radio_value(preview_kind, ShaderPreviewKind::ComputeWgsl, "Compute WGSL");
                ui.radio_value(
                    preview_kind,
                    ShaderPreviewKind::FragmentWgsl,
                    "Fragment WGSL",
                );
                if let Ok(ref full_wgsl) = wgsl_res {
                    if ui
                        .small_button("Copy WGSL")
                        .on_hover_text("Copy active WGSL shader source to clipboard")
                        .clicked()
                    {
                        ui.ctx().copy_text(full_wgsl.clone());
                    }
                }
            });
            let frame = egui::Frame::new()
                .fill(ui.visuals().extreme_bg_color)
                .stroke(egui::Stroke::new(1.0_f32, palette.graph_edge))
                .corner_radius(egui::CornerRadius::same(4))
                .inner_margin(egui::Margin::symmetric(8, 6));
            frame.show(ui, |ui| match wgsl_res {
                Ok(src) => render_six_line_snippet(ui, &src, &palette),
                Err(err) => {
                    ui.colored_label(palette.warning, err);
                }
            });
        });
}

fn render_six_line_snippet(ui: &mut egui::Ui, wgsl: &str, palette: &SemanticPalette) {
    let entry_idx = wgsl
        .lines()
        .position(|l| l.starts_with("@compute") || l.starts_with("@fragment"))
        .unwrap_or(0);
    for line in wgsl.lines().skip(entry_idx).take(6) {
        let display_line = if line.chars().count() > 38 {
            let prefix: String = line.chars().take(37).collect();
            format!("{prefix}…")
        } else {
            line.to_string()
        };
        let color = if line.starts_with("@compute") || line.starts_with("@fragment") {
            palette.info
        } else if line.trim_start().starts_with("//") {
            palette.muted_text
        } else {
            ui.visuals().text_color()
        };
        ui.label(
            egui::RichText::new(display_line)
                .monospace()
                .size(10.5)
                .color(color),
        );
    }
}
