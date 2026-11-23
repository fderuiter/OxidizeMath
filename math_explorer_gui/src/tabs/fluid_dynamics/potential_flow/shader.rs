#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ElementType, PotentialFlowTool, SerializedFlowElement, FLOW_VIEW_MAX, FLOW_VIEW_MIN};
use crate::framework::{InteractiveTool, ShaderDialect};
use math_commons::ast::{format_glsl_float, format_shader_f32, ShaderTarget};
use math_commons::math_kernel::colormap::colormap_prelude_for;
use math_commons::math_kernel::glsl_colormap::{
    glsl_ensure_dark_canvas_contrast_fn, glsl_heatmap_color_fn,
};
use std::fmt::Write;

/// Maximum number of potential-flow singularities baked into an exported GLSL shader.
pub const MAX_SHADER_SINGULARITIES: usize = 64;

impl PotentialFlowTool {
    pub(super) fn build_wgsl_shader(&self) -> Result<String, String> {
        if !self.has_finite_parameters() {
            return Err("Potential flow parameters are non-finite or out of bounds".to_string());
        }
        let view_range = format_shader_f32(self.view_range.clamp(FLOW_VIEW_MIN, FLOW_VIEW_MAX))
            .map_err(|e| e.to_string())?;
        let ref_speed =
            format_shader_f32(self.reference_flow_speed()).map_err(|e| e.to_string())?;
        let capped_count = self.elements.len().min(MAX_SHADER_SINGULARITIES);
        let hc_flag = u32::from(self.high_contrast);

        let mut out = String::with_capacity(4096);
        writeln!(
            out,
            "// OxidizeMath Analytical 2D Potential Flow Shader (WGSL) - {capped_count}/{} elements (cap {MAX_SHADER_SINGULARITIES})",
            self.elements.len()
        )
        .unwrap();
        writeln!(out, "const VIEW_RANGE: f32 = {view_range};").unwrap();
        writeln!(out, "const REF_SPEED: f32 = {ref_speed};").unwrap();
        writeln!(out, "const HIGH_CONTRAST: u32 = {hc_flag}u;\n").unwrap();
        out.push_str(colormap_prelude_for(ShaderTarget::Wgsl));
        out.push('\n');
        out.push_str(FLOW_ELEMENT_HELPERS_WGSL);
        write_wgsl_superposition_fn(&mut out, &self.elements[..capped_count])?;
        out.push_str(FLOW_RENDER_KERNEL_WGSL);
        Ok(out)
    }

    pub(super) fn build_glsl_shader(&self, dialect: ShaderDialect) -> Result<String, String> {
        if !self.has_finite_parameters() {
            return Err("Potential flow parameters are non-finite or out of bounds".to_string());
        }
        let view_range = format_glsl_float(self.view_range.clamp(FLOW_VIEW_MIN, FLOW_VIEW_MAX))
            .map_err(|e| e.to_string())?;
        let ref_speed =
            format_glsl_float(self.reference_flow_speed()).map_err(|e| e.to_string())?;
        let capped_count = self.elements.len().min(MAX_SHADER_SINGULARITIES);

        let mut out = String::with_capacity(4096);
        write_flow_header(&mut out, dialect, self.elements.len(), capped_count);
        writeln!(out, "const float VIEW_RANGE = {view_range};").unwrap();
        writeln!(out, "const float REF_SPEED = {ref_speed};").unwrap();
        writeln!(out, "const bool HIGH_CONTRAST = {};\n", self.high_contrast).unwrap();
        out.push_str(glsl_heatmap_color_fn());
        out.push('\n');
        out.push_str(glsl_ensure_dark_canvas_contrast_fn());
        out.push('\n');
        out.push_str(FLOW_ELEMENT_HELPERS_GLSL);
        write_superposition_fn(&mut out, &self.elements[..capped_count])?;
        out.push_str(FLOW_RENDER_KERNEL_GLSL);
        write_flow_entrypoint(&mut out, dialect);
        Ok(out)
    }

    fn reference_flow_speed(&self) -> f64 {
        let mut sum = 0.0;
        for el in self.elements.iter().take(MAX_SHADER_SINGULARITIES) {
            if el.element_type == ElementType::Uniform {
                sum += el.strength.abs();
            } else {
                sum += el.strength.abs() / (2.0 * std::f64::consts::PI * self.view_range.max(1.0));
            }
        }
        if sum > 1.0e-3 {
            sum * 1.75
        } else {
            10.0
        }
    }
}

fn write_flow_header(out: &mut String, dialect: ShaderDialect, total: usize, emitted: usize) {
    if dialect == ShaderDialect::WebGl2Es300 {
        out.push_str("#version 300 es\nprecision highp float;\n\n");
        out.push_str("uniform vec2 u_resolution;\nuniform float u_time;\nuniform vec4 u_mouse;\nout vec4 fragColor;\n\n");
    }
    writeln!(
        out,
        "// OxidizeMath Analytical 2D Potential Flow Shader ({}) - {emitted}/{total} elements (cap {MAX_SHADER_SINGULARITIES})",
        dialect.label()
    )
    .unwrap();
}

fn write_superposition_fn(
    out: &mut String,
    elements: &[SerializedFlowElement],
) -> Result<(), String> {
    out.push_str("\nvec4 eval_potential_flow(vec2 p) {\n    vec4 acc = vec4(0.0);\n");
    for el in elements {
        let s = format_glsl_float(el.strength).map_err(|e| e.to_string())?;
        let x = format_glsl_float(el.x).map_err(|e| e.to_string())?;
        let y = format_glsl_float(el.y).map_err(|e| e.to_string())?;
        match el.element_type {
            ElementType::Uniform => {
                let rad = format_glsl_float(el.angle.to_radians()).map_err(|e| e.to_string())?;
                writeln!(out, "    acc += flow_uniform(p, {s}, {rad});").unwrap();
            }
            ElementType::Source => {
                writeln!(out, "    acc += flow_source(p, {s}, vec2({x}, {y}));").unwrap();
            }
            ElementType::Sink => {
                let neg_s = format_glsl_float(-el.strength).map_err(|e| e.to_string())?;
                writeln!(out, "    acc += flow_source(p, {neg_s}, vec2({x}, {y}));").unwrap();
            }
            ElementType::Vortex => {
                writeln!(out, "    acc += flow_vortex(p, {s}, vec2({x}, {y}));").unwrap();
            }
            ElementType::Doublet => {
                writeln!(out, "    acc += flow_doublet(p, {s}, vec2({x}, {y}));").unwrap();
            }
        }
    }
    out.push_str("    return acc;\n}\n\n");
    Ok(())
}

fn write_flow_entrypoint(out: &mut String, dialect: ShaderDialect) {
    match dialect {
        ShaderDialect::WebGl2Es300 => {
            out.push_str(
                "\nvoid main() {\n    fragColor = render_potential_flow(gl_FragCoord.xy, max(u_resolution, vec2(1.0)));\n}\n",
            );
        }
        ShaderDialect::Shadertoy => {
            out.push_str(
                "\nvoid mainImage(out vec4 fragColor, in vec2 fragCoord) {\n    fragColor = render_potential_flow(fragCoord, max(iResolution.xy, vec2(1.0)));\n}\n",
            );
        }
    }
}

const FLOW_ELEMENT_HELPERS_GLSL: &str = r#"const float INV_TWO_PI = 0.15915494309189535;

float safe_atan2(float y, float x) {
    return (abs(x) < 1.0e-12 && abs(y) < 1.0e-12) ? 0.0 : atan(y, x);
}

vec4 flow_uniform(vec2 p, float vel, float angle_rad) {
    float c = cos(angle_rad);
    float s = sin(angle_rad);
    float psi = vel * (p.y * c - p.x * s);
    float phi = vel * (p.x * c + p.y * s);
    return vec4(vel * c, vel * s, psi, phi);
}

vec4 flow_source(vec2 p, float q, vec2 center) {
    vec2 d = p - center;
    float r2 = max(dot(d, d), 1.0e-6);
    vec2 uv = (q * INV_TWO_PI / r2) * d;
    float psi = (q * INV_TWO_PI) * safe_atan2(d.y, d.x);
    float phi = (q * INV_TWO_PI) * 0.5 * log(r2);
    return vec4(uv, psi, phi);
}

vec4 flow_vortex(vec2 p, float gamma, vec2 center) {
    vec2 d = p - center;
    float r2 = max(dot(d, d), 1.0e-6);
    vec2 uv = (gamma * INV_TWO_PI / r2) * vec2(-d.y, d.x);
    float psi = (gamma * INV_TWO_PI) * 0.5 * log(r2);
    float phi = -(gamma * INV_TWO_PI) * safe_atan2(d.y, d.x);
    return vec4(uv, psi, phi);
}

vec4 flow_doublet(vec2 p, float kappa, vec2 center) {
    vec2 d = p - center;
    float r2 = max(dot(d, d), 1.0e-4);
    float r4 = r2 * r2;
    float u = -kappa * (d.x * d.x - d.y * d.y) * INV_TWO_PI / r4;
    float v = -kappa * (2.0 * d.x * d.y) * INV_TWO_PI / r4;
    float psi = -(kappa * d.y) * INV_TWO_PI / r2;
    float phi = -(kappa * d.x) * INV_TWO_PI / r2;
    return vec4(u, v, psi, phi);
}
"#;

const FLOW_RENDER_KERNEL_GLSL: &str = r#"float iso_contour(float field_val, float spacing) {
    float s = field_val / max(spacing, 1.0e-4);
    float dist_line = abs(fract(s - 0.5) - 0.5);
    float fw = clamp(fwidth(s), 1.0e-4, 0.35);
    return 1.0 - smoothstep(0.0, fw * 1.4, dist_line);
}

vec4 render_potential_flow(vec2 frag_coord, vec2 res) {
    float aspect = res.x / max(res.y, 1.0);
    vec2 uv = (frag_coord / res) - 0.5;
    vec2 p = vec2(uv.x * 2.0 * VIEW_RANGE * aspect, uv.y * 2.0 * VIEW_RANGE);

    vec4 state = eval_potential_flow(p);
    float speed_norm = clamp(length(state.xy) / max(REF_SPEED, 1.0e-4), 0.0, 1.0);
    vec3 col = heatmap_color(speed_norm);

    float spacing = max(REF_SPEED * VIEW_RANGE * 0.08, 0.25);
    float stream_mask = iso_contour(state.z, spacing);
    float equipot_mask = iso_contour(state.w, spacing * 1.5);
    col = mix(col, vec3(0.95, 0.99, 1.0), 0.65 * stream_mask);
    col = mix(col, vec3(1.0, 0.88, 0.45), 0.28 * equipot_mask);

    if (HIGH_CONTRAST) {
        col = ensure_dark_canvas_contrast(col);
    }
    return vec4(clamp(col, 0.0, 1.0), 1.0);
}
"#;

fn write_wgsl_superposition_fn(
    out: &mut String,
    elements: &[SerializedFlowElement],
) -> Result<(), String> {
    out.push_str(
        "\nfn eval_potential_flow(p: vec2<f32>) -> vec4<f32> {\n    var acc = vec4<f32>(0.0);\n",
    );
    for el in elements {
        let s = format_shader_f32(el.strength).map_err(|e| e.to_string())?;
        let x = format_shader_f32(el.x).map_err(|e| e.to_string())?;
        let y = format_shader_f32(el.y).map_err(|e| e.to_string())?;
        match el.element_type {
            ElementType::Uniform => {
                let rad = format_shader_f32(el.angle.to_radians()).map_err(|e| e.to_string())?;
                writeln!(out, "    acc = acc + flow_uniform(p, {s}, {rad});").unwrap();
            }
            ElementType::Source => {
                writeln!(
                    out,
                    "    acc = acc + flow_source(p, {s}, vec2<f32>({x}, {y}));"
                )
                .unwrap();
            }
            ElementType::Sink => {
                let neg_s = format_shader_f32(-el.strength).map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "    acc = acc + flow_source(p, {neg_s}, vec2<f32>({x}, {y}));"
                )
                .unwrap();
            }
            ElementType::Vortex => {
                writeln!(
                    out,
                    "    acc = acc + flow_vortex(p, {s}, vec2<f32>({x}, {y}));"
                )
                .unwrap();
            }
            ElementType::Doublet => {
                writeln!(
                    out,
                    "    acc = acc + flow_doublet(p, {s}, vec2<f32>({x}, {y}));"
                )
                .unwrap();
            }
        }
    }
    out.push_str("    return acc;\n}\n\n");
    Ok(())
}

const FLOW_ELEMENT_HELPERS_WGSL: &str = r#"struct FlowUniforms {
    resolution: vec2<f32>,
    time: f32,
    pad0: f32,
};

@group(0) @binding(0)
var<uniform> u_flow: FlowUniforms;

const INV_TWO_PI: f32 = 0.15915494;

fn safe_atan2(y: f32, x: f32) -> f32 {
    if (abs(x) < 1.0e-12 && abs(y) < 1.0e-12) {
        return 0.0;
    }
    return atan2(y, x);
}

fn flow_uniform(p: vec2<f32>, vel: f32, angle_rad: f32) -> vec4<f32> {
    let c = cos(angle_rad);
    let s = sin(angle_rad);
    return vec4<f32>(vel * c, vel * s, vel * (p.y * c - p.x * s), vel * (p.x * c + p.y * s));
}

fn flow_source(p: vec2<f32>, q: f32, center: vec2<f32>) -> vec4<f32> {
    let d = p - center;
    let r2 = max(dot(d, d), 1.0e-6);
    let uv = (q * INV_TWO_PI / r2) * d;
    return vec4<f32>(uv, (q * INV_TWO_PI) * safe_atan2(d.y, d.x), (q * INV_TWO_PI) * 0.5 * log(r2));
}

fn flow_vortex(p: vec2<f32>, gamma: f32, center: vec2<f32>) -> vec4<f32> {
    let d = p - center;
    let r2 = max(dot(d, d), 1.0e-6);
    let uv = (gamma * INV_TWO_PI / r2) * vec2<f32>(-d.y, d.x);
    return vec4<f32>(uv, (gamma * INV_TWO_PI) * 0.5 * log(r2), -(gamma * INV_TWO_PI) * safe_atan2(d.y, d.x));
}

fn flow_doublet(p: vec2<f32>, kappa: f32, center: vec2<f32>) -> vec4<f32> {
    let d = p - center;
    let r2 = max(dot(d, d), 1.0e-4);
    let r4 = r2 * r2;
    let u = -kappa * (d.x * d.x - d.y * d.y) * INV_TWO_PI / r4;
    let v = -kappa * (2.0 * d.x * d.y) * INV_TWO_PI / r4;
    return vec4<f32>(u, v, -(kappa * d.y) * INV_TWO_PI / r2, -(kappa * d.x) * INV_TWO_PI / r2);
}
"#;

const FLOW_RENDER_KERNEL_WGSL: &str = r#"fn iso_contour(field_val: f32, spacing: f32) -> f32 {
    let s = field_val / max(spacing, 1.0e-4);
    let dist_line = abs(fract(s - 0.5) - 0.5);
    let fw = clamp(fwidth(s), 1.0e-4, 0.35);
    return 1.0 - smoothstep(0.0, fw * 1.4, dist_line);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let res = select(vec2<f32>(512.0, 512.0), u_flow.resolution, u_flow.resolution.x > 0.0 && u_flow.resolution.y > 0.0);
    let aspect = res.x / max(res.y, 1.0);
    let uv = vec2<f32>(frag_coord.x / res.x, 1.0 - frag_coord.y / res.y) - 0.5;
    let p = vec2<f32>(uv.x * 2.0 * VIEW_RANGE * aspect, uv.y * 2.0 * VIEW_RANGE);

    let state = eval_potential_flow(p);
    let speed_norm = clamp(length(state.xy) / max(REF_SPEED, 1.0e-4), 0.0, 1.0);
    var col = om_heatmap_color(speed_norm);

    let spacing = max(REF_SPEED * VIEW_RANGE * 0.08, 0.25);
    let stream_mask = iso_contour(state.z, spacing);
    let equipot_mask = iso_contour(state.w, spacing * 1.5);
    col = mix(col, vec3<f32>(0.95, 0.99, 1.0), 0.65 * stream_mask);
    col = mix(col, vec3<f32>(1.0, 0.88, 0.45), 0.28 * equipot_mask);
    return vec4<f32>(clamp(om_apply_high_contrast(col, HIGH_CONTRAST), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
"#;
