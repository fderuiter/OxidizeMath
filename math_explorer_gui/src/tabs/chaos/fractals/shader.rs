#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{COORD_MAX, COORD_MIN, FractalMode, FractalViewer, ZOOM_MAX, ZOOM_MIN};
use crate::framework::{InteractiveTool, ShaderDialect};
use math_commons::ast::{GLSL_COMPLEX_HELPERS, format_glsl_float};
use math_commons::math_kernel::glsl_colormap::{
    glsl_cyclic_cosine_palette_fn, glsl_ensure_dark_canvas_contrast_fn,
};
use std::fmt::Write;

const FRACTAL_RESERVED_SYMBOLS: &[&str] = &[
    "CENTER",
    "CUSTOM_ESCAPE",
    "ESCAPE_R2",
    "HIGH_CONTRAST",
    "IS_JULIA",
    "JULIA_C",
    "MAX_ITER",
    "ZOOM",
    "cabs",
    "cacos",
    "carg",
    "casin",
    "catan",
    "catan2",
    "cclamp",
    "ccos",
    "ccosh",
    "cdiv",
    "cexp",
    "cln",
    "clog10",
    "cmax",
    "cmin",
    "cmul",
    "cpow",
    "crect",
    "csin",
    "csinh",
    "csqrt",
    "cstep",
    "ctan",
    "ctanh",
    "cyclic_cosine_palette",
    "ensure_dark_canvas_contrast",
    "eval_escape_step",
    "fragColor",
    "iResolution",
    "linear_channel_to_srgb",
    "relative_luminance_srgb",
    "render_fractal",
    "srgb_channel_to_linear",
    "u_mouse",
    "u_resolution",
    "u_time",
];

impl FractalViewer {
    fn compile_escape_step_fn(&self) -> Result<String, String> {
        if !self.mode.is_custom() {
            return Ok(
                "vec2 eval_escape_step(vec2 z, vec2 c) {\n    return vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;\n}\n"
                    .to_string(),
            );
        }
        let expr = self.math_input.expr().ok_or_else(|| {
            self.math_input
                .error_message()
                .unwrap_or_else(|| "Custom fractal expression is invalid".to_string())
        })?;
        let compiled = self.math_input.compiled().ok_or_else(|| {
            self.math_input
                .error_message()
                .unwrap_or_else(|| "Custom fractal expression failed to compile".to_string())
        })?;
        let values = self.math_input.param_slice();
        let names = compiled.param_names();
        let mut scope_vars = Vec::with_capacity(2 + names.len());
        scope_vars.extend(["z", "c"]);
        let mut body = String::from("vec2 eval_escape_step(vec2 z, vec2 c) {\n");
        for (i, param_name) in names.iter().enumerate() {
            if FRACTAL_RESERVED_SYMBOLS.contains(&param_name.as_str()) {
                return Err(format!(
                    "Parameter '{param_name}' conflicts with a reserved fractal shader identifier"
                ));
            }
            let raw_val = values.get(i).copied().unwrap_or(1.0);
            let glsl_val = format_glsl_float(raw_val).map_err(|e| e.to_string())?;
            writeln!(body, "    const vec2 {param_name} = vec2({glsl_val}, 0.0);").unwrap();
            scope_vars.push(param_name.as_str());
        }
        let step_expr = expr
            .to_glsl_complex_with_vars(&scope_vars)
            .map_err(|e| e.to_string())?;
        writeln!(body, "    return {step_expr};\n}}").unwrap();
        Ok(body)
    }

    pub(super) fn build_glsl_shader(&self, dialect: ShaderDialect) -> Result<String, String> {
        if !self.has_finite_parameters() {
            return Err("Fractal parameters are non-finite or out of bounds".to_string());
        }
        let step_fn = self.compile_escape_step_fn()?;
        let cx = format_glsl_float(self.center.re.clamp(COORD_MIN, COORD_MAX))
            .map_err(|e| e.to_string())?;
        let cy = format_glsl_float(self.center.im.clamp(COORD_MIN, COORD_MAX))
            .map_err(|e| e.to_string())?;
        let zoom = format_glsl_float(self.zoom.clamp(ZOOM_MIN, ZOOM_MAX))
            .map_err(|e| e.to_string())?;
        let jx = format_glsl_float(self.julia_c.re.clamp(COORD_MIN, COORD_MAX))
            .map_err(|e| e.to_string())?;
        let jy = format_glsl_float(self.julia_c.im.clamp(COORD_MIN, COORD_MAX))
            .map_err(|e| e.to_string())?;
        let max_iter = self
            .max_iter
            .clamp(10, math_commons::registry::MAX_ITERATIONS as u32);
        let is_julia = self.mode.uses_julia_c();
        let is_custom = self.mode.is_custom();
        let escape_r2 = if is_custom { "4.0" } else { "256.0" };

        let mut out = String::with_capacity(4096);
        write_dialect_header(&mut out, dialect, self.mode);
        writeln!(out, "const vec2 CENTER = vec2({cx}, {cy});").unwrap();
        writeln!(out, "const float ZOOM = {zoom};").unwrap();
        writeln!(out, "const int MAX_ITER = {max_iter};").unwrap();
        writeln!(out, "const bool IS_JULIA = {is_julia};").unwrap();
        writeln!(out, "const bool CUSTOM_ESCAPE = {is_custom};").unwrap();
        writeln!(out, "const float ESCAPE_R2 = {escape_r2};").unwrap();
        writeln!(out, "const vec2 JULIA_C = vec2({jx}, {jy});").unwrap();
        writeln!(out, "const bool HIGH_CONTRAST = {};\n", self.high_contrast).unwrap();
        if is_custom {
            out.push_str(GLSL_COMPLEX_HELPERS);
            out.push('\n');
        }
        out.push_str(glsl_cyclic_cosine_palette_fn());
        out.push('\n');
        out.push_str(glsl_ensure_dark_canvas_contrast_fn());
        out.push('\n');
        out.push_str(&step_fn);
        out.push('\n');
        out.push_str(FRACTAL_KERNEL_GLSL);
        write_dialect_entrypoint(&mut out, dialect);
        Ok(out)
    }
}

fn write_dialect_header(out: &mut String, dialect: ShaderDialect, mode: FractalMode) {
    if dialect == ShaderDialect::WebGl2Es300 {
        out.push_str("#version 300 es\nprecision highp float;\n\n");
        out.push_str("uniform vec2 u_resolution;\nuniform float u_time;\nuniform vec4 u_mouse;\nout vec4 fragColor;\n\n");
    }
    let family = match mode {
        FractalMode::Mandelbrot => "Analytical Mandelbrot",
        FractalMode::Julia => "Analytical Julia",
        FractalMode::CustomMandelbrot => "Custom Escape",
        FractalMode::CustomJulia => "Custom Julia Escape",
    };
    writeln!(
        out,
        "// OxidizeMath {family} Escape-Time Shader ({})",
        dialect.label()
    )
    .unwrap();
}

fn write_dialect_entrypoint(out: &mut String, dialect: ShaderDialect) {
    match dialect {
        ShaderDialect::WebGl2Es300 => {
            out.push_str(
                "\nvoid main() {\n    fragColor = render_fractal(gl_FragCoord.xy, max(u_resolution, vec2(1.0)));\n}\n",
            );
        }
        ShaderDialect::Shadertoy => {
            out.push_str(
                "\nvoid mainImage(out vec4 fragColor, in vec2 fragCoord) {\n    fragColor = render_fractal(fragCoord, max(iResolution.xy, vec2(1.0)));\n}\n",
            );
        }
    }
}

const FRACTAL_KERNEL_GLSL: &str = r#"vec4 render_fractal(vec2 frag_coord, vec2 res) {
    float scale_y = 3.0 / ZOOM;
    float aspect = res.x / max(res.y, 1.0);
    vec2 uv = (frag_coord / res) - 0.5;
    vec2 p = CENTER + vec2(uv.x * scale_y * aspect, uv.y * scale_y);

    vec2 z = IS_JULIA ? p : vec2(0.0, 0.0);
    vec2 c = IS_JULIA ? JULIA_C : p;
    bool escaped = false;
    int iter = 0;

    for (int i = 0; i < MAX_ITER; ++i) {
        float r2 = dot(z, z);
        if (r2 > ESCAPE_R2) {
            escaped = true;
            iter = i;
            break;
        }
        z = eval_escape_step(z, c);
    }

    if (!escaped) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    float band = float(iter);
    if (!CUSTOM_ESCAPE) {
        float r2 = max(dot(z, z), 4.0001);
        band = max(band - log2(max(log2(r2), 1.0e-6)) + 4.0, 0.0);
    }
    vec3 col = cyclic_cosine_palette(band);
    if (HIGH_CONTRAST) {
        col = ensure_dark_canvas_contrast(col);
    }
    return vec4(col, 1.0);
}
"#;
