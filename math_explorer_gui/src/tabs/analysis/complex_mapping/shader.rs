#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{COMPLEX_COORD_MAX, COMPLEX_COORD_MIN, ComplexMappingTool, MappingFunction};
use crate::framework::{InteractiveTool, ShaderDialect};
use math_commons::ast::{GLSL_COMPLEX_HELPERS, format_glsl_float};
use math_commons::math_kernel::glsl_colormap::{
    glsl_cyclic_cosine_palette_fn, glsl_ensure_dark_canvas_contrast_fn,
};
use std::fmt::Write;

const SHADER_RESERVED_IDENTS: &[&str] = &[
    "GRID_DIVS",
    "GRID_MAX",
    "GRID_MIN",
    "HIGH_CONTRAST",
    "cabs",
    "cacos",
    "carg",
    "casin",
    "catan",
    "catan2",
    "cclamp",
    "ccosh",
    "ccos",
    "cdiv",
    "cexp",
    "cln",
    "clog10",
    "cmax",
    "cmin",
    "cmul",
    "cpow",
    "crect",
    "csinh",
    "csin",
    "csqrt",
    "cstep",
    "ctanh",
    "ctan",
    "cyclic_cosine_palette",
    "ensure_dark_canvas_contrast",
    "eval_mapping",
    "fragColor",
    "grid_line_mask",
    "iResolution",
    "linear_channel_to_srgb",
    "relative_luminance_srgb",
    "render_complex_domain",
    "srgb_channel_to_linear",
    "u_mouse",
    "u_resolution",
    "u_time",
];

impl ComplexMappingTool {
    pub(super) fn compile_mapping_glsl(&self) -> Result<(String, String), String> {
        match self.function {
            MappingFunction::ZSquared => Ok((String::new(), "cmul(z, z)".to_string())),
            MappingFunction::Exponential => Ok((String::new(), "cexp(z)".to_string())),
            MappingFunction::Sine => Ok((String::new(), "csin(z)".to_string())),
            MappingFunction::Inverse => Ok((String::new(), "cdiv(vec2(1.0, 0.0), z)".to_string())),
            MappingFunction::Custom => self.compile_custom_mapping_glsl(),
        }
    }

    fn compile_custom_mapping_glsl(&self) -> Result<(String, String), String> {
        let invalid_err = || {
            self.math_input
                .error_message()
                .unwrap_or_else(|| "Custom complex expression is invalid".to_string())
        };
        let expr = self.math_input.expr().ok_or_else(invalid_err)?;
        let compiled = self.math_input.compiled().ok_or_else(invalid_err)?;
        let param_names = compiled.param_names();
        let param_vals = self.math_input.param_slice();

        let mut allowed_vars = Vec::with_capacity(1 + param_names.len());
        allowed_vars.push("z");
        let mut param_decls = String::new();
        for (idx, name) in param_names.iter().enumerate() {
            if SHADER_RESERVED_IDENTS.contains(&name.as_str()) {
                return Err(format!(
                    "Parameter '{name}' collides with a reserved GLSL shader symbol"
                ));
            }
            let val = param_vals.get(idx).copied().unwrap_or(1.0);
            let lit = format_glsl_float(val).map_err(|e| e.to_string())?;
            writeln!(param_decls, "    const vec2 {name} = vec2({lit}, 0.0);").unwrap();
            allowed_vars.push(name.as_str());
        }
        let expr_glsl = expr
            .to_glsl_complex_with_vars(&allowed_vars)
            .map_err(|e| e.to_string())?;
        Ok((param_decls, expr_glsl))
    }

    pub(super) fn build_glsl_shader(&self, dialect: ShaderDialect) -> Result<String, String> {
        if !self.has_finite_parameters() {
            return Err("Complex mapping parameters are non-finite or out of bounds".to_string());
        }
        let (param_decls, expr_glsl) = self.compile_mapping_glsl()?;
        let g_min = format_glsl_float(self.grid_min.clamp(COMPLEX_COORD_MIN, COMPLEX_COORD_MAX))
            .map_err(|e| e.to_string())?;
        let g_max = format_glsl_float(self.grid_max.clamp(COMPLEX_COORD_MIN, COMPLEX_COORD_MAX))
            .map_err(|e| e.to_string())?;
        let density = self
            .grid_density
            .clamp(2, math_commons::registry::MAX_GRID_SIZE);

        let mut out = String::with_capacity(4096);
        write_mapping_header(&mut out, dialect, self.function.name());
        writeln!(out, "const float GRID_MIN = {g_min};").unwrap();
        writeln!(out, "const float GRID_MAX = {g_max};").unwrap();
        writeln!(out, "const float GRID_DIVS = {}.0;", density - 1).unwrap();
        writeln!(out, "const bool HIGH_CONTRAST = {};\n", self.high_contrast).unwrap();
        out.push_str(GLSL_COMPLEX_HELPERS);
        out.push('\n');
        out.push_str(glsl_cyclic_cosine_palette_fn());
        out.push('\n');
        out.push_str(glsl_ensure_dark_canvas_contrast_fn());
        writeln!(
            out,
            "\nvec2 eval_mapping(vec2 z) {{\n{param_decls}    return {expr_glsl};\n}}\n"
        )
        .unwrap();
        out.push_str(COMPLEX_DOMAIN_KERNEL_GLSL);
        write_mapping_entrypoint(&mut out, dialect);
        Ok(out)
    }
}

fn write_mapping_header(out: &mut String, dialect: ShaderDialect, fn_label: &str) {
    if dialect == ShaderDialect::WebGl2Es300 {
        out.push_str("#version 300 es\nprecision highp float;\n\n");
        out.push_str("uniform vec2 u_resolution;\nuniform float u_time;\nuniform vec4 u_mouse;\nout vec4 fragColor;\n\n");
    }
    writeln!(
        out,
        "// OxidizeMath Conformal Domain-Coloring Shader: {fn_label} ({})",
        dialect.label()
    )
    .unwrap();
}

fn write_mapping_entrypoint(out: &mut String, dialect: ShaderDialect) {
    match dialect {
        ShaderDialect::WebGl2Es300 => {
            out.push_str(
                "\nvoid main() {\n    fragColor = render_complex_domain(gl_FragCoord.xy, max(u_resolution, vec2(1.0)));\n}\n",
            );
        }
        ShaderDialect::Shadertoy => {
            out.push_str(
                "\nvoid mainImage(out vec4 fragColor, in vec2 fragCoord) {\n    fragColor = render_complex_domain(fragCoord, max(iResolution.xy, vec2(1.0)));\n}\n",
            );
        }
    }
}

const COMPLEX_DOMAIN_KERNEL_GLSL: &str = r#"float grid_line_mask(float coord, float step_size) {
    float scaled = coord / max(step_size, 1.0e-4);
    float dist_to_line = abs(fract(scaled - 0.5) - 0.5);
    float fw = max(fwidth(scaled), 1.0e-4);
    return 1.0 - smoothstep(0.0, fw * 1.35, dist_to_line);
}

vec4 render_complex_domain(vec2 frag_coord, vec2 res) {
    float span = max(GRID_MAX - GRID_MIN, 0.1);
    float center = 0.5 * (GRID_MIN + GRID_MAX);
    float aspect = res.x / max(res.y, 1.0);
    vec2 uv = (frag_coord / res) - 0.5;
    vec2 z = vec2(center + uv.x * span * aspect, center + uv.y * span);

    vec2 w = eval_mapping(z);
    float mag = length(w);
    float phase = carg(w);
    vec3 base_hue = cyclic_cosine_palette((phase + 3.141592653589793) / 0.15);

    float log_mod = log2(max(mag, 1.0e-6));
    float ring = 0.72 + 0.28 * fract(log_mod);
    float step_size = span / max(GRID_DIVS, 1.0);
    float w_grid = max(grid_line_mask(w.x, step_size), grid_line_mask(w.y, step_size));

    vec3 col = mix(base_hue * ring, vec3(0.96, 0.98, 1.0), 0.55 * w_grid);
    if (HIGH_CONTRAST) {
        col = ensure_dark_canvas_contrast(col);
    }
    return vec4(clamp(col, 0.0, 1.0), 1.0);
}
"#;
