#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Standalone WebGL2 (`#version 300 es`) and Shadertoy (`mainImage`) `.frag` lensing shaders.

use std::fmt::Write;

use crate::framework::{InteractiveTool, ShaderDialect};
use math_commons::ast::format_glsl_float;
use math_commons::math_kernel::glsl_colormap::glsl_ensure_dark_canvas_contrast_fn;

use super::{GeodesicPlaygroundTool, LensingBackgroundKind, SpacetimePreset};

impl GeodesicPlaygroundTool {
    pub(super) fn build_lensing_glsl_shader(
        &self,
        dialect: ShaderDialect,
    ) -> Result<String, String> {
        if !self.has_finite_parameters() {
            return Err("Geodesic lensing parameters are non-finite".to_string());
        }
        let rs = format_glsl_float(self.state.scale_radius.clamp(0.5, 5.0))
            .map_err(|e| e.to_string())?;
        let chi =
            format_glsl_float(self.state.spin_chi.clamp(-0.95, 0.95)).map_err(|e| e.to_string())?;
        let fov = format_glsl_float(self.lensing_fov_b_max()).map_err(|e| e.to_string())?;
        let (b_neg_val, b_pos_val) = self
            .current_lensing_spec()
            .critical_impact_bounds()
            .unwrap_or_else(|_| self.deflection_table.critical_impact_bounds());
        let b_neg = format_glsl_float(b_neg_val.abs()).map_err(|e| e.to_string())?;
        let b_pos = format_glsl_float(b_pos_val.abs()).map_err(|e| e.to_string())?;
        let metric_id = match self.state.spacetime {
            SpacetimePreset::Schwarzschild | SpacetimePreset::CustomRadialLapse => 0,
            SpacetimePreset::KerrEquatorial => 1,
            SpacetimePreset::MorrisThorneWormhole | SpacetimePreset::CustomWormholeProfile => 2,
        };
        let bg_id = match self.state.lensing_background {
            LensingBackgroundKind::GalacticDisk => 0,
            LensingBackgroundKind::LeniaPortal => 1,
            LensingBackgroundKind::Starfield => 2,
        };

        let mut out = String::with_capacity(4096);
        write_lensing_header(&mut out, dialect, self.state.spacetime.label());
        writeln!(out, "const int METRIC_KIND = {metric_id};").unwrap();
        writeln!(out, "const int BG_KIND = {bg_id};").unwrap();
        writeln!(out, "const float SCALE_RS = {rs};").unwrap();
        writeln!(out, "const float SPIN_CHI = {chi};").unwrap();
        writeln!(out, "const float FOV_B_MAX = {fov};").unwrap();
        writeln!(out, "const float B_CRIT_NEG = {b_neg};").unwrap();
        writeln!(out, "const float B_CRIT_POS = {b_pos};").unwrap();
        writeln!(out, "const bool HIGH_CONTRAST = {};\n", self.high_contrast).unwrap();
        out.push_str(glsl_ensure_dark_canvas_contrast_fn());
        out.push('\n');
        out.push_str(LENSING_KERNEL_GLSL);
        write_lensing_entrypoint(&mut out, dialect);
        Ok(out)
    }
}

fn write_lensing_header(out: &mut String, dialect: ShaderDialect, metric_label: &str) {
    if dialect == ShaderDialect::WebGl2Es300 {
        out.push_str("#version 300 es\nprecision highp float;\n\n");
        out.push_str("uniform vec2 u_resolution;\nuniform float u_time;\nuniform vec4 u_mouse;\nout vec4 fragColor;\n\n");
    }
    writeln!(
        out,
        "// OxidizeMath Relativistic Gravitational Lensing Shader ({}) - {metric_label}",
        dialect.label()
    )
    .unwrap();
}

fn write_lensing_entrypoint(out: &mut String, dialect: ShaderDialect) {
    match dialect {
        ShaderDialect::WebGl2Es300 => {
            out.push_str(
                "\nvoid main() {\n    fragColor = render_lensing(gl_FragCoord.xy, max(u_resolution, vec2(1.0)), u_time);\n}\n",
            );
        }
        ShaderDialect::Shadertoy => {
            out.push_str(
                "\nvoid mainImage(out vec4 fragColor, in vec2 fragCoord) {\n    fragColor = render_lensing(fragCoord, max(iResolution.xy, vec2(1.0)), iTime);\n}\n",
            );
        }
    }
}

const LENSING_KERNEL_GLSL: &str = r#"float star_hash(vec2 cell) {
    return fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
}

vec3 sample_starfield(vec2 uv) {
    vec2 g = uv * 14.0;
    vec2 id = floor(g);
    vec2 f = fract(g) - 0.5;
    float h = star_hash(id);
    float star = (h > 0.84) ? exp(-42.0 * dot(f, f)) * h : 0.0;
    return vec3(0.08, 0.11, 0.18) + star * vec3(1.0, 0.96, 0.88);
}

vec3 sample_galaxy_source(vec2 uv, float t) {
    float r = length(uv) * 3.4;
    float phi = atan(uv.y, uv.x);
    float bulge = exp(-2.2 * r * r);
    float arm = 0.5 + 0.5 * cos(2.0 * (phi - 0.12 * t) - 3.4 * log(r + 0.15));
    float disk = exp(-r / 1.6) * (0.45 + 0.55 * arm);
    float gas = exp(-pow(r - 2.1, 2.0) / 1.8) * arm;
    return clamp(vec3(0.95 * bulge + 0.58 * disk, 0.72 * disk + 0.48 * gas, 0.35 * disk + 0.90 * gas) + sample_starfield(uv) * 0.4, 0.0, 1.0);
}

vec3 sample_lenia_portal(vec2 uv, float t) {
    vec2 c = vec2(0.18 * cos(0.4 * t), 0.18 * sin(0.4 * t));
    float r = length(uv - c);
    float orbium = exp(-22.0 * pow(r - 0.34, 2.0)) + 0.65 * exp(-36.0 * pow(r - 0.14, 2.0));
    return clamp(vec3(0.16 + 0.82 * orbium * orbium, 0.12 + 0.90 * orbium, 0.24 + 0.78 * sqrt(max(orbium, 0.0))), 0.0, 1.0);
}

vec4 render_lensing(vec2 frag_coord, vec2 res, float t) {
    vec2 ndc = (2.0 * frag_coord - res) / max(res.y, 1.0);
    vec2 b_vec = ndc * (0.5 * FOV_B_MAX);
    float b_rad = max(length(b_vec), 1e-5);
    float cos_psi = b_vec.x / b_rad;

    float b_crit = B_CRIT_POS;
    float twist = 0.0;
    if (METRIC_KIND == 1) {
        float w = 0.5 * (1.0 + cos_psi);
        float d_flat = 1.0 - 0.08 * abs(SPIN_CHI) * w;
        b_crit = (w * B_CRIT_POS + (1.0 - w) * B_CRIT_NEG) * d_flat;
        twist = (0.5 * SPIN_CHI * SCALE_RS * SCALE_RS) / (b_rad * b_rad + 1.5);
    }

    if (METRIC_KIND != 2 && b_rad < b_crit) {
        return vec4(HIGH_CONTRAST ? vec3(0.0) : vec3(0.02, 0.024, 0.045), 1.0);
    }

    bool lower_sheet = (METRIC_KIND == 2) && (b_rad < SCALE_RS);
    float defl = (2.0 * SCALE_RS) / max(b_rad - 0.65 * b_crit, 0.25 * SCALE_RS);
    vec2 dir = b_vec / b_rad;
    vec2 rot_dir = vec2(dir.x * cos(twist) - dir.y * sin(twist), dir.x * sin(twist) + dir.y * cos(twist));
    vec2 src_uv = lower_sheet
        ? rot_dir * (b_rad / max(SCALE_RS, 1e-4)) * 0.85
        : rot_dir * ((b_rad / max(0.5 * FOV_B_MAX, 1e-4)) - 0.18 * defl);

    vec3 col = (lower_sheet || BG_KIND == 1)
        ? sample_lenia_portal(src_uv, t)
        : ((BG_KIND == 0) ? sample_galaxy_source(src_uv, t) : sample_starfield(src_uv));

    float ring_edge = (METRIC_KIND == 2) ? SCALE_RS : b_crit;
    float ring_boost = exp(-18.0 * abs(b_rad - ring_edge) / max(ring_edge, 1e-4));
    col = clamp(col + ring_boost * vec3(0.92, 0.76, 0.38), 0.0, 1.0);
    if (HIGH_CONTRAST) {
        col = ensure_dark_canvas_contrast(col);
    }
    return vec4(col, 1.0);
}
"#;
