#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Standalone WGSL (`.wgsl`), GLSL (`.glsl`), WebGL2 (`#version 300 es`), and Shadertoy (`mainImage`)
//! fragment shader generator for [`super::galactic_orbits_tool::GalacticOrbitSimulator`].

use crate::framework::{InteractiveTool, ShaderDialect};
use math_explorer::physics::astrophysics::{decompose_rotation_curve, OrbitParticleKind};

use super::galactic_orbits_tool::GalacticOrbitSimulator;
use super::HaloKindChoice;

impl GalacticOrbitSimulator {
    fn galactic_shader_uniforms(&self) -> Result<GalacticShaderUniforms, String> {
        if !self.has_finite_parameters() {
            return Err("Galactic orbit parameters are non-finite".to_string());
        }
        let halo_mode = match self.cfg.halo_kind {
            HaloKindChoice::BaryonicOnly => 0,
            HaloKindChoice::NfwCusp => 1,
            HaloKindChoice::BurkertCore => 2,
        };
        let outer_v = decompose_rotation_curve(
            4.0 * self.cfg.disk_scale_kpc,
            &self.system.baryons,
            &self.system.halo,
        )
        .v_circ_kms
        .max(10.0);

        let probe_xy = self
            .system
            .particles()
            .iter()
            .rev()
            .find(|p| !p.captured && p.kind == OrbitParticleKind::SlingshotProbe)
            .map(|p| [p.x_kpc, p.y_kpc])
            .unwrap_or([0.0, 0.0]);
        let has_probe = i32::from(self.system.slingshot_probe_count() > 0);
        Ok(GalacticShaderUniforms {
            halo_mode,
            r_disk: self.cfg.disk_scale_kpc,
            r_halo: self.cfg.halo_scale_kpc,
            gas_frac: self.cfg.gas_fraction,
            outer_v,
            elapsed_myr: self.system.elapsed_myr(),
            probe_xy,
            has_probe,
        })
    }

    pub(crate) fn build_galactic_wgsl_shader(&self) -> Result<String, String> {
        let u = self.galactic_shader_uniforms()?;
        Ok(galactic_wgsl_shader(&u))
    }

    pub(crate) fn build_galactic_glsl_shader(
        &self,
        dialect: ShaderDialect,
    ) -> Result<String, String> {
        let u = self.galactic_shader_uniforms()?;
        let core = galactic_shader_core(&u);
        match dialect {
            ShaderDialect::WebGl2Es300 => Ok(wrap_webgl2_galactic(&core)),
            ShaderDialect::Shadertoy => Ok(wrap_shadertoy_galactic(&core)),
        }
    }
}

struct GalacticShaderUniforms {
    halo_mode: i32,
    r_disk: f64,
    r_halo: f64,
    gas_frac: f64,
    outer_v: f64,
    elapsed_myr: f64,
    probe_xy: [f64; 2],
    has_probe: i32,
}

fn galactic_shader_core(u: &GalacticShaderUniforms) -> String {
    format!(
        r#"const int HALO_MODE = {halo_mode};
const float R_DISK = {r_disk:.6};
const float R_HALO = {r_halo:.6};
const float GAS_FRAC = {gas_frac:.6};
const float V_OUTER = {outer_v:.6};
const float ELAPSED_MYR = {elapsed_myr:.6};
const vec2 PROBE_XY = vec2({px:.6}, {py:.6});
const int HAS_PROBE = {has_probe};

float rotationCurveRatio(float r) {{
    float x = max(r, 0.05) / max(R_HALO, 0.2);
    if (HALO_MODE == 0) {{
        return clamp((r / R_DISK) * exp(-0.45 * r / R_DISK), 0.15, 1.2);
    }} else if (HALO_MODE == 1) {{
        return clamp(sqrt(log(1.0 + x) / max(x, 0.08)), 0.2, 1.35);
    }}
    return clamp(sqrt((x * x) / (1.0 + x * x)), 0.15, 1.25);
}}

vec3 renderGalacticOrbits(vec2 uv, float tSec) {{
    float fovKpc = 5.5 * R_DISK + R_HALO;
    vec2 p = uv * fovKpc;
    float r = length(p);
    float phi = atan(p.y, p.x);
    float vRatio = rotationCurveRatio(r);
    float omega = (0.045 * V_OUTER * vRatio) / max(r, 0.22 * R_DISK);
    float phase = phi - omega * (0.015 * ELAPSED_MYR + 0.35 * tSec);

    float disk = exp(-r / max(R_DISK, 0.15));
    float arm = 0.5 + 0.5 * cos(2.0 * (phase - 2.6 * log(max(r / R_DISK, 0.08) + 0.2)));
    float bulge = 0.9 * exp(-pow(r / (0.38 * R_DISK), 2.0));
    float lum = disk * (0.35 + (0.40 + 0.35 * GAS_FRAC) * pow(arm, 2.2)) + bulge;

    vec3 stellar = vec3(1.0, 0.80, 0.52) * bulge + vec3(0.38, 0.66, 1.0) * (disk * arm);
    float haloRing = exp(-160.0 * pow((r - R_HALO) / fovKpc, 2.0));
    vec3 col = stellar * (0.6 + 0.9 * lum) + vec3(0.18, 0.55, 0.72) * (0.35 * haloRing);

    if (HAS_PROBE == 1) {{
        float dProbe = length(p - PROBE_XY) / fovKpc;
        col += vec3(1.0, 0.78, 0.22) * exp(-900.0 * dProbe * dProbe);
    }}
    return clamp(col / (1.0 + 0.55 * col), 0.0, 1.0);
}}
"#,
        halo_mode = u.halo_mode,
        r_disk = u.r_disk,
        r_halo = u.r_halo,
        gas_frac = u.gas_frac,
        outer_v = u.outer_v,
        elapsed_myr = u.elapsed_myr,
        px = u.probe_xy[0],
        py = u.probe_xy[1],
        has_probe = u.has_probe,
    )
}

fn wrap_webgl2_galactic(core: &str) -> String {
    format!(
        r#"#version 300 es
// Standalone WebGL2 Galactic Disk, Dark-Matter Rotation Shear & Slingshot Shader
precision highp float;

uniform vec2 u_resolution;
uniform float u_time;
out vec4 fragColor;

{core}
void main() {{
    vec2 res = max(u_resolution, vec2(1.0));
    vec2 uv = (gl_FragCoord.xy - 0.5 * res) / (0.5 * res.y);
    fragColor = vec4(renderGalacticOrbits(uv, u_time), 1.0);
}}
"#
    )
}

fn wrap_shadertoy_galactic(core: &str) -> String {
    format!(
        r#"// Shadertoy Galactic Disk, Dark-Matter Rotation Shear & Slingshot Shader
{core}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {{
    vec2 res = max(iResolution.xy, vec2(1.0));
    vec2 uv = (fragCoord - 0.5 * res) / (0.5 * res.y);
    fragColor = vec4(renderGalacticOrbits(uv, iTime), 1.0);
}}
"#
    )
}

fn galactic_wgsl_shader(u: &GalacticShaderUniforms) -> String {
    let header = format!(
        "// Standalone WGSL Galactic Disk, Dark-Matter Rotation Shear & Slingshot Shader\n\
const HALO_MODE: i32 = {};\n\
const R_DISK: f32 = {:.6};\n\
const R_HALO: f32 = {:.6};\n\
const GAS_FRAC: f32 = {:.6};\n\
const V_OUTER: f32 = {:.6};\n\
const ELAPSED_MYR: f32 = {:.6};\n\
const PROBE_XY: vec2<f32> = vec2<f32>({:.6}, {:.6});\n\
const HAS_PROBE: i32 = {};\n\n",
        u.halo_mode,
        u.r_disk,
        u.r_halo,
        u.gas_frac,
        u.outer_v,
        u.elapsed_myr,
        u.probe_xy[0],
        u.probe_xy[1],
        u.has_probe,
    );
    format!("{header}{GALACTIC_KERNEL_WGSL}")
}

const GALACTIC_KERNEL_WGSL: &str = r#"struct GalacticUniforms {
    resolution: vec2<f32>,
    time: f32,
    pad0: f32,
};

@group(0) @binding(0)
var<uniform> u_galactic: GalacticUniforms;

fn rotation_curve_ratio(r: f32) -> f32 {
    let x = max(r, 0.05) / max(R_HALO, 0.2);
    if (HALO_MODE == 0) {
        return clamp((r / R_DISK) * exp(-0.45 * r / R_DISK), 0.15, 1.2);
    }
    if (HALO_MODE == 1) {
        return clamp(sqrt(log(1.0 + x) / max(x, 0.08)), 0.2, 1.35);
    }
    return clamp(sqrt((x * x) / (1.0 + x * x)), 0.15, 1.25);
}

fn render_galactic_orbits(uv: vec2<f32>, t_sec: f32) -> vec3<f32> {
    let fov_kpc = 5.5 * R_DISK + R_HALO;
    let p = uv * fov_kpc;
    let r = length(p);
    let phi = atan2(p.y, p.x);
    let v_ratio = rotation_curve_ratio(r);
    let omega = (0.045 * V_OUTER * v_ratio) / max(r, 0.22 * R_DISK);
    let phase = phi - omega * (0.015 * ELAPSED_MYR + 0.35 * t_sec);

    let disk = exp(-r / max(R_DISK, 0.15));
    let arm = 0.5 + 0.5 * cos(2.0 * (phase - 2.6 * log(max(r / R_DISK, 0.08) + 0.2)));
    let rb = r / (0.38 * R_DISK);
    let bulge = 0.9 * exp(-(rb * rb));
    let lum = disk * (0.35 + (0.40 + 0.35 * GAS_FRAC) * pow(arm, 2.2)) + bulge;

    let stellar = vec3<f32>(1.0, 0.80, 0.52) * bulge + vec3<f32>(0.38, 0.66, 1.0) * (disk * arm);
    let dr_halo = (r - R_HALO) / fov_kpc;
    let halo_ring = exp(-160.0 * dr_halo * dr_halo);
    var col = stellar * (0.6 + 0.9 * lum) + vec3<f32>(0.18, 0.55, 0.72) * (0.35 * halo_ring);

    if (HAS_PROBE == 1) {
        let d_probe = length(p - PROBE_XY) / fov_kpc;
        col = col + vec3<f32>(1.0, 0.78, 0.22) * exp(-900.0 * d_probe * d_probe);
    }
    return clamp(col / (vec3<f32>(1.0) + 0.55 * col), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let res = select(
        vec2<f32>(512.0, 512.0),
        u_galactic.resolution,
        u_galactic.resolution.x > 0.0 && u_galactic.resolution.y > 0.0,
    );
    let uv = (frag_coord.xy - 0.5 * res) / max(0.5 * res.y, 1.0);
    return vec4<f32>(render_galactic_orbits(uv, u_galactic.time), 1.0);
}
"#;

const PRESET_SPIRAL_WINDING: &str = concat!(
    r#"{"metadata":{"schema_version":"1.0","title":"Differential Spiral Winding","domain":"astrophysics","timestamp":"2026-11-21T00:00:00Z"},"#,
    r#""config":{"distance_mpc":16.0,"morphology":"TypeCode10","gas_fraction":0.60,"disk_scale_kpc":0.85,"halo_kind":"NfwCusp","#,
    r#""halo_log10_mass":10.3,"halo_scale_kpc":2.6,"halo_concentration":13.0,"dt_myr":1.8,"substeps_per_frame":4,"disk_tracers":24,"tidal_tracers":4,"tidal_tangential_ratio":0.70}}"#
);

const PRESET_TIDAL_PRECESSION: &str = concat!(
    r#"{"metadata":{"schema_version":"1.0","title":"Dwarf Tidal Stream Precession","domain":"astrophysics","timestamp":"2026-11-21T00:00:00Z"},"#,
    r#""config":{"distance_mpc":14.0,"morphology":"TypeCode95To99","gas_fraction":0.65,"disk_scale_kpc":0.75,"halo_kind":"BurkertCore","#,
    r#""halo_log10_mass":10.0,"halo_scale_kpc":2.0,"halo_concentration":11.0,"dt_myr":1.5,"substeps_per_frame":5,"disk_tracers":12,"tidal_tracers":12,"tidal_tangential_ratio":0.48}}"#
);

const PRESET_SLINGSHOT_CAPTURE: &str = concat!(
    r#"{"metadata":{"schema_version":"1.0","title":"Hypervelocity Slingshot Arena","domain":"astrophysics","timestamp":"2026-11-23T00:00:00Z"},"#,
    r#""config":{"distance_mpc":10.0,"morphology":"TypeCode10","gas_fraction":0.55,"disk_scale_kpc":0.90,"halo_kind":"NfwCusp","#,
    r#""halo_log10_mass":10.6,"halo_scale_kpc":3.0,"halo_concentration":15.0,"dt_myr":1.2,"substeps_per_frame":5,"disk_tracers":16,"tidal_tracers":6,"tidal_tangential_ratio":0.35}}"#
);

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "galactic_orbits_spiral_winding_starter",
        title: "Differential Spiral Winding",
        description: "Shear two-arm disk tracers into differential spiral arcs inside an NFW dark-matter halo.",
        category: "Galactic Dynamics",
        icon: "🌀",
        domain: "astrophysics",
        tool_name: Some("GalacticOrbitSimulator"),
        preset_generator: || PRESET_SPIRAL_WINDING,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "galactic_orbits_tidal_precession_starter",
        title: "Dwarf Tidal Stream Precession",
        description: "Integrate eccentric satellite stream rosetta orbits inside a cored Burkert (1995) dwarf potential.",
        category: "Galactic Dynamics",
        icon: "☄️",
        domain: "astrophysics",
        tool_name: Some("GalacticOrbitSimulator"),
        preset_generator: || PRESET_TIDAL_PRECESSION,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "galactic_orbits_slingshot_arena_starter",
        title: "Hypervelocity Slingshot Arena",
        description: "Aim and launch eccentric tracer stars across a massive NFW halo with live Doppler sonification.",
        category: "Galactic Dynamics",
        icon: "🚀",
        domain: "astrophysics",
        tool_name: Some("GalacticOrbitSimulator"),
        preset_generator: || PRESET_SLINGSHOT_CAPTURE,
    }
}
