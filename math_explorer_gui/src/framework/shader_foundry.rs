//! Standalone WebGPU/Bevy `.wgsl` and GLSL ES 3.00 / Shadertoy `.glsl` shader assembly and live preview UI.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use math_commons::ast::{
    complex_prelude_for, format_shader_f32, Expr, ShaderEmitError, ShaderTarget,
};
use math_commons::math_kernel::colormap::{colormap_prelude_for, ShaderPalette};

use super::ShaderExportFormat;

/// Default uniform values baked into standalone exported `.wgsl` and `.glsl` fragment shaders.
#[derive(Clone, Copy, Debug)]
pub struct ShaderUniformDefaults {
    /// Default viewport pixel resolution `[width, height]`.
    pub resolution: [u32; 2],
    /// Default complex viewport bounds `[re_min, im_min, re_max, im_max]`.
    pub viewport: [f64; 4],
    /// Default escape-time iteration ceiling.
    pub max_iter: u32,
    /// Default complex parameter $c = (\text{re}, \text{im})$.
    pub julia_c: [f64; 2],
    /// Selected analytical colormap family.
    pub palette: ShaderPalette,
    /// Whether WCAG 2.2 `3.0:1` linear-light contrast lifting is enabled by default.
    pub high_contrast: bool,
    /// Whether continuous Douady-Hubbard log-log escape renormalization is enabled.
    pub smooth_coloring: bool,
}

/// Cached live shader code preview state shared by shader-capable tools.
#[derive(Clone, Debug)]
pub struct ShaderPreviewState {
    /// Currently selected shader preview target (`WGSL` or `GLSL`).
    pub format: ShaderExportFormat,
    cached_key: Option<(ShaderExportFormat, u64)>,
    cached_source: Result<String, String>,
}

impl Default for ShaderPreviewState {
    fn default() -> Self {
        Self {
            format: ShaderExportFormat::Wgsl,
            cached_key: None,
            cached_source: Ok(String::new()),
        }
    }
}

impl ShaderPreviewState {
    /// Invalidates the cached shader preview so the next open preview frame regenerates it.
    pub fn invalidate(&mut self) {
        self.cached_key = None;
    }

    /// Returns the cached shader source for `revision`, recomputing via `build` only on cache miss.
    pub fn get_or_build<F>(&mut self, revision: u64, build: F) -> &Result<String, String>
    where
        F: FnOnce(ShaderExportFormat) -> Result<String, String>,
    {
        let key = (self.format, revision);
        if self.cached_key != Some(key) {
            self.cached_source = build(self.format);
            self.cached_key = Some(key);
        }
        &self.cached_source
    }
}

pub use super::shader_params::{bind_complex_shader_params, validate_complex_shader_vars};

struct FormattedDefaults {
    res_w: String,
    res_h: String,
    vp: [String; 4],
    julia: [String; 2],
    max_iter: u32,
    palette_id: u32,
    hc_flag: u32,
}

fn format_defaults(d: &ShaderUniformDefaults) -> Result<FormattedDefaults, String> {
    let map_err = |e: ShaderEmitError| e.to_string();
    Ok(FormattedDefaults {
        res_w: format_shader_f32(f64::from(d.resolution[0].max(1))).map_err(map_err)?,
        res_h: format_shader_f32(f64::from(d.resolution[1].max(1))).map_err(map_err)?,
        vp: [
            format_shader_f32(d.viewport[0]).map_err(map_err)?,
            format_shader_f32(d.viewport[1]).map_err(map_err)?,
            format_shader_f32(d.viewport[2]).map_err(map_err)?,
            format_shader_f32(d.viewport[3]).map_err(map_err)?,
        ],
        julia: [
            format_shader_f32(d.julia_c[0]).map_err(map_err)?,
            format_shader_f32(d.julia_c[1]).map_err(map_err)?,
        ],
        max_iter: d.max_iter.clamp(1, 5_000),
        palette_id: d.palette.palette_id(),
        hc_flag: u32::from(d.high_contrast),
    })
}

fn wgsl_uniform_header(fd: &FormattedDefaults) -> String {
    format!(
        r#"// OxidizeMath Symbolic Shader Foundry — WebGPU / Bevy Fragment Shader
struct ShaderUniforms {{
    resolution: vec2<f32>,
    time: f32,
    max_iter: u32,
    viewport: vec4<f32>,
    julia_c: vec2<f32>,
    palette_id: u32,
    high_contrast: u32,
}};

@group(0) @binding(0)
var<uniform> u_shader: ShaderUniforms;

const DEFAULT_RESOLUTION: vec2<f32> = vec2<f32>({}, {});
const DEFAULT_VIEWPORT: vec4<f32> = vec4<f32>({}, {}, {}, {});
const DEFAULT_JULIA_C: vec2<f32> = vec2<f32>({}, {});
const DEFAULT_MAX_ITER: u32 = {}u;
const DEFAULT_PALETTE_ID: u32 = {}u;
const DEFAULT_HIGH_CONTRAST: u32 = {}u;

fn om_resolve_res() -> vec2<f32> {{
    if (u_shader.resolution.x > 0.0 && u_shader.resolution.y > 0.0) {{
        return u_shader.resolution;
    }}
    return DEFAULT_RESOLUTION;
}}

fn om_resolve_viewport() -> vec4<f32> {{
    if (u_shader.viewport.z > u_shader.viewport.x && u_shader.viewport.w > u_shader.viewport.y) {{
        return u_shader.viewport;
    }}
    return DEFAULT_VIEWPORT;
}}

fn om_resolve_julia_c() -> vec2<f32> {{
    if (u_shader.max_iter > 0u) {{
        return u_shader.julia_c;
    }}
    return DEFAULT_JULIA_C;
}}
"#,
        fd.res_w,
        fd.res_h,
        fd.vp[0],
        fd.vp[1],
        fd.vp[2],
        fd.vp[3],
        fd.julia[0],
        fd.julia[1],
        fd.max_iter,
        fd.palette_id,
        fd.hc_flag
    )
}

fn glsl_uniform_header(fd: &FormattedDefaults) -> String {
    format!(
        r#"#version 300 es
precision highp float;
precision highp int;

// OxidizeMath Symbolic Shader Foundry — GLSL ES 3.00 / Shadertoy Fragment Shader
uniform vec3 iResolution;
uniform float iTime;
uniform vec4 u_viewport;
uniform vec2 u_julia_c;
uniform int u_use_custom_uniforms;
uniform int u_max_iter;
uniform int u_palette_id;
uniform int u_high_contrast;

const vec2 DEFAULT_RESOLUTION = vec2({}, {});
const vec4 DEFAULT_VIEWPORT = vec4({}, {}, {}, {});
const vec2 DEFAULT_JULIA_C = vec2({}, {});
const int DEFAULT_MAX_ITER = {};
const int DEFAULT_PALETTE_ID = {};
const int DEFAULT_HIGH_CONTRAST = {};

vec2 om_resolve_res() {{
    return (iResolution.x > 0.0 && iResolution.y > 0.0) ? iResolution.xy : DEFAULT_RESOLUTION;
}}

vec4 om_resolve_viewport() {{
    return (u_viewport.z > u_viewport.x && u_viewport.w > u_viewport.y) ? u_viewport : DEFAULT_VIEWPORT;
}}

vec2 om_resolve_julia_c() {{
    return (u_use_custom_uniforms != 0) ? u_julia_c : DEFAULT_JULIA_C;
}}
"#,
        fd.res_w,
        fd.res_h,
        fd.vp[0],
        fd.vp[1],
        fd.vp[2],
        fd.vp[3],
        fd.julia[0],
        fd.julia[1],
        fd.max_iter,
        fd.palette_id,
        fd.hc_flag
    )
}

fn emit_wgsl_fractal_main(expr_code: &str, use_julia_seed: bool, smooth: bool) -> String {
    let seed_init = if use_julia_seed {
        "var z = z_pixel;\n    let c = om_resolve_julia_c();"
    } else {
        "var z = vec2<f32>(0.0, 0.0);\n    let c = z_pixel;"
    };
    let (esc_r2, renorm_call) = if smooth {
        ("256.0", "om_renorm_escape(i, prev_norm_sqr, norm_sqr)")
    } else {
        ("4.0", "f32(i)")
    };
    format!(
        r#"fn om_renorm_escape(i: u32, prev_r2: f32, curr_r2: f32) -> f32 {{
    let lc = log(max(0.5 * log(curr_r2), 1.0e-12));
    if (i > 0u && prev_r2 > 1.05) {{
        let lp = log(max(0.5 * log(prev_r2), 1.0e-12));
        if (lc - lp > 1.0e-6) {{
            return max(f32(i) - 1.0 + clamp((1.0197814 - lp) / (lc - lp), 0.0, 1.0), 0.0);
        }}
    }}
    return max(f32(i) + clamp(1.0 - (lc - 1.0197814) / 0.6931472, 0.0, 1.0), 0.0);
}}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {{
    let res = om_resolve_res();
    let vp = om_resolve_viewport();
    let uv = vec2<f32>(frag_coord.x / res.x, 1.0 - frag_coord.y / res.y);
    let z_pixel = mix(vp.xy, vp.zw, uv);
    {seed_init}
    let t = vec2<f32>(u_shader.time, 0.0);
    let max_iter = select(DEFAULT_MAX_ITER, min(u_shader.max_iter, 5000u), u_shader.max_iter > 0u);
    let palette_id = select(DEFAULT_PALETTE_ID, u_shader.palette_id, u_shader.max_iter > 0u);
    let hc = select(DEFAULT_HIGH_CONTRAST, u_shader.high_contrast, u_shader.max_iter > 0u);
    var prev_norm_sqr = dot(z, z);
    for (var i: u32 = 0u; i < 5000u; i = i + 1u) {{
        if (i >= max_iter) {{ break; }}
        let norm_sqr = dot(z, z);
        if (norm_sqr > {esc_r2}) {{
            return vec4<f32>(om_eval_palette({renorm_call}, max_iter, palette_id, hc), 1.0);
        }}
        prev_norm_sqr = norm_sqr;
        z = {expr_code};
    }}
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}}
"#
    )
}

fn emit_glsl_fractal_main(expr_code: &str, use_julia_seed: bool, smooth: bool) -> String {
    let seed_init = if use_julia_seed {
        "vec2 z = z_pixel;\n    vec2 c = om_resolve_julia_c();"
    } else {
        "vec2 z = vec2(0.0, 0.0);\n    vec2 c = z_pixel;"
    };
    let (esc_r2, renorm_call) = if smooth {
        ("256.0", "om_renorm_escape(i, prev_norm_sqr, norm_sqr)")
    } else {
        ("4.0", "float(i)")
    };
    format!(
        r#"float om_renorm_escape(int i, float prev_r2, float curr_r2) {{
    float lc = log(max(0.5 * log(curr_r2), 1.0e-12));
    if (i > 0 && prev_r2 > 1.05) {{
        float lp = log(max(0.5 * log(prev_r2), 1.0e-12));
        if (lc - lp > 1.0e-6) {{
            return max(float(i) - 1.0 + clamp((1.0197814 - lp) / (lc - lp), 0.0, 1.0), 0.0);
        }}
    }}
    return max(float(i) + clamp(1.0 - (lc - 1.0197814) / 0.6931472, 0.0, 1.0), 0.0);
}}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {{
    vec2 res = om_resolve_res();
    vec4 vp = om_resolve_viewport();
    vec2 z_pixel = mix(vp.xy, vp.zw, fragCoord / res);
    {seed_init}
    vec2 t = vec2(iTime, 0.0);
    int max_iter = (u_max_iter > 0) ? min(u_max_iter, 5000) : DEFAULT_MAX_ITER;
    int palette_id = (u_use_custom_uniforms != 0) ? u_palette_id : DEFAULT_PALETTE_ID;
    int hc = (u_use_custom_uniforms != 0) ? u_high_contrast : DEFAULT_HIGH_CONTRAST;
    float prev_norm_sqr = dot(z, z);
    for (int i = 0; i < 5000; i++) {{
        if (i >= max_iter) {{ break; }}
        float norm_sqr = dot(z, z);
        if (norm_sqr > {esc_r2}) {{
            fragColor = vec4(om_eval_palette({renorm_call}, max_iter, palette_id, hc), 1.0);
            return;
        }}
        prev_norm_sqr = norm_sqr;
        z = {expr_code};
    }}
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
}}

#ifndef SHADERTOY
out vec4 om_fragColor;
void main() {{ mainImage(om_fragColor, gl_FragCoord.xy); }}
#endif
"#
    )
}

/// Assembles a standalone WebGPU/Bevy `.wgsl` or GLSL ES 3.00 / Shadertoy `.glsl` fractal fragment shader.
pub fn emit_fractal_shader(
    format: ShaderExportFormat,
    step_expr: &Expr,
    use_julia_seed: bool,
    defaults: &ShaderUniformDefaults,
) -> Result<String, String> {
    validate_complex_shader_vars(step_expr)?;
    let target = format.ast_target();
    let expr_code = step_expr
        .to_shader_complex(target)
        .map_err(|e| e.to_string())?;
    let fd = format_defaults(defaults)?;
    let header = match target {
        ShaderTarget::Wgsl => wgsl_uniform_header(&fd),
        ShaderTarget::Glsl => glsl_uniform_header(&fd),
    };
    let entry = match target {
        ShaderTarget::Wgsl => {
            emit_wgsl_fractal_main(&expr_code, use_julia_seed, defaults.smooth_coloring)
        }
        ShaderTarget::Glsl => {
            emit_glsl_fractal_main(&expr_code, use_julia_seed, defaults.smooth_coloring)
        }
    };
    Ok(format!(
        "{header}\n{}\n{}\n{entry}",
        complex_prelude_for(target),
        colormap_prelude_for(target)
    ))
}

/// Assembles a standalone `.wgsl` or `.glsl` complex domain-coloring fragment shader.
pub fn emit_domain_coloring_shader(
    format: ShaderExportFormat,
    map_expr: &Expr,
    defaults: &ShaderUniformDefaults,
) -> Result<String, String> {
    validate_complex_shader_vars(map_expr)?;
    let target = format.ast_target();
    let expr_code = map_expr
        .to_shader_complex(target)
        .map_err(|e| e.to_string())?;
    let fd = format_defaults(defaults)?;
    let entry = match target {
        ShaderTarget::Wgsl => format!(
            r#"@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {{
    let res = om_resolve_res();
    let vp = om_resolve_viewport();
    let uv = vec2<f32>(frag_coord.x / res.x, 1.0 - frag_coord.y / res.y);
    let z = mix(vp.xy, vp.zw, uv);
    let c = om_resolve_julia_c();
    let t = vec2<f32>(u_shader.time, 0.0);
    let palette_id = select(DEFAULT_PALETTE_ID, u_shader.palette_id, u_shader.max_iter > 0u);
    let hc = select(DEFAULT_HIGH_CONTRAST, u_shader.high_contrast, u_shader.max_iter > 0u);
    let w = {expr_code};
    return vec4<f32>(om_domain_color(w, palette_id, hc), 1.0);
}}
"#
        ),
        ShaderTarget::Glsl => format!(
            r#"void mainImage(out vec4 fragColor, in vec2 fragCoord) {{
    vec2 res = om_resolve_res();
    vec4 vp = om_resolve_viewport();
    vec2 uv = fragCoord / res;
    vec2 z = mix(vp.xy, vp.zw, uv);
    vec2 c = om_resolve_julia_c();
    vec2 t = vec2(iTime, 0.0);
    int palette_id = (u_use_custom_uniforms != 0) ? u_palette_id : DEFAULT_PALETTE_ID;
    int hc = (u_use_custom_uniforms != 0) ? u_high_contrast : DEFAULT_HIGH_CONTRAST;
    vec2 w = {expr_code};
    fragColor = vec4(om_domain_color(w, palette_id, hc), 1.0);
}}

#ifndef SHADERTOY
out vec4 om_fragColor;
void main() {{
    mainImage(om_fragColor, gl_FragCoord.xy);
}}
#endif
"#
        ),
    };
    let header = match target {
        ShaderTarget::Wgsl => wgsl_uniform_header(&fd),
        ShaderTarget::Glsl => glsl_uniform_header(&fd),
    };
    Ok(format!(
        "{header}\n{}\n{}\n{entry}",
        complex_prelude_for(target),
        colormap_prelude_for(target)
    ))
}

/// Renders a collapsible live WGSL / GLSL shader preview panel inside a tool's side panel.
pub fn render_shader_preview_section<F>(
    ui: &mut egui::Ui,
    id_salt: &str,
    preview: &mut ShaderPreviewState,
    revision: u64,
    build: F,
) where
    F: FnOnce(ShaderExportFormat) -> Result<String, String>,
{
    ui.separator();
    egui::CollapsingHeader::new("GPU Shader Preview")
        .id_salt(id_salt)
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Target:");
                ui.radio_value(&mut preview.format, ShaderExportFormat::Wgsl, "WGSL");
                ui.radio_value(
                    &mut preview.format,
                    ShaderExportFormat::Glsl,
                    "GLSL ES 3.00",
                );
            });
            match preview.get_or_build(revision, build) {
                Ok(code) => {
                    egui::ScrollArea::vertical()
                        .id_salt((id_salt, "scroll"))
                        .max_height(160.0)
                        .show(ui, |ui| {
                            let mut snippet = code.as_str();
                            ui.add(
                                egui::TextEdit::multiline(&mut snippet)
                                    .font(egui::TextStyle::Monospace)
                                    .desired_rows(6)
                                    .desired_width(f32::INFINITY),
                            );
                        });
                }
                Err(err) => {
                    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                    ui.colored_label(palette.error, format!("⚠️ {err}"));
                }
            }
        });
}
