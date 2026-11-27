//! Iterative 3D spatial AST validation and WebGPU `@compute` WGSL kernel generation.
// [cite:math_commons]

use super::eval::{MAX_EVAL_DEPTH, lookup_universal_real_const};
use super::expr::Expr;
use super::limits::{MAX_AST_NODES, MAX_GLSL_OUTPUT_BYTES};
use super::shader::{ShaderEmitError, format_shader_f32};
use super::shader_preludes::wgsl_real_prelude;
use std::fmt::Write as _;

/// Default 3D WebGPU compute workgroup dimensions (`4 * 4 * 4 = 64` invocations).
pub const DEFAULT_COMPUTE_WORKGROUP_3D: [u32; 3] = [4, 4, 4];

const COMPUTE_RESERVED_IDENTS: &[&str] = &[
    "cs_main",
    "gid",
    "u_vol",
    "field_in",
    "field_out",
    "VolumeComputeUniforms",
    "DEFAULT_DIMS",
    "DEFAULT_BOUNDS_MIN",
    "DEFAULT_BOUNDS_MAX",
    "dims",
    "bmin",
    "bmax",
    "pos",
    "idx",
    "raw_val",
];

/// Configuration for assembling a standalone 3D WebGPU `@compute` WGSL scalar-field kernel.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComputeKernelConfig {
    /// Cubic or rectilinear voxel grid dimensions `[nx, ny, nz]`.
    pub grid_dims: [u32; 3],
    /// Workgroup tile dimensions `[wx, wy, wz]` (`wx * wy * wz <= 256`).
    pub workgroup_size: [u32; 3],
    /// World-space `[x_min, x_max]` bounds.
    pub x_bounds: [f64; 2],
    /// World-space `[y_min, y_max]` bounds.
    pub y_bounds: [f64; 2],
    /// World-space `[z_min, z_max]` bounds.
    pub z_bounds: [f64; 2],
    /// Bound scalar parameter `(identifier, value)` pairs emitted as `const` `f32` bindings.
    pub params: Vec<(String, f64)>,
}

impl Default for ComputeKernelConfig {
    fn default() -> Self {
        Self::new_3d(64, 2.5)
    }
}

impl ComputeKernelConfig {
    /// Creates an isotropic `dim^3` compute configuration spanning `[-half_extent, +half_extent]^3`.
    #[must_use]
    pub fn new_3d(dim: u32, half_extent: f64) -> Self {
        let clamped_dim = dim.max(2);
        let ext = if half_extent.is_finite() && half_extent > 0.05 {
            half_extent
        } else {
            2.5
        };
        Self {
            grid_dims: [clamped_dim, clamped_dim, clamped_dim],
            workgroup_size: DEFAULT_COMPUTE_WORKGROUP_3D,
            x_bounds: [-ext, ext],
            y_bounds: [-ext, ext],
            z_bounds: [-ext, ext],
            params: Vec::new(),
        }
    }

    /// Appends a named scalar constant parameter `(name, value)` to this compute configuration.
    #[must_use]
    pub fn with_param(mut self, name: impl Into<String>, value: f64) -> Self {
        self.params.push((name.into(), value));
        self
    }
}

fn is_valid_compute_param_ident(ident: &str) -> bool {
    if ident.is_empty()
        || matches!(ident, "x" | "y" | "z" | "t")
        || lookup_universal_real_const(ident).is_some()
        || COMPUTE_RESERVED_IDENTS.contains(&ident)
        || ident.starts_with("om_")
        || ident.starts_with("c_")
        || ident.starts_with("gl_")
    {
        return false;
    }
    let mut chars = ident.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn validate_expr_allowed_vars(expr: &Expr, allowed_vars: &[&str]) -> Result<(), ShaderEmitError> {
    let mut visited = 0usize;
    let mut stack = Vec::with_capacity(16);
    stack.push((expr, 0usize));

    while let Some((node, depth)) = stack.pop() {
        if depth > MAX_EVAL_DEPTH {
            return Err(ShaderEmitError::RecursionLimitExceeded(MAX_EVAL_DEPTH));
        }
        visited += 1;
        if visited > MAX_AST_NODES {
            return Err(ShaderEmitError::NodeLimitExceeded(MAX_AST_NODES));
        }
        match node {
            Expr::Number(v) => {
                if !v.is_finite() {
                    return Err(ShaderEmitError::NonFiniteLiteral);
                }
            }
            Expr::Variable(name) => {
                if lookup_universal_real_const(name).is_none()
                    && (COMPUTE_RESERVED_IDENTS.contains(&name.as_str())
                        || !allowed_vars.contains(&name.as_str()))
                {
                    return Err(ShaderEmitError::InvalidIdentifier(name.clone()));
                }
            }
            Expr::Unary { expr: inner, .. } => stack.push((inner, depth + 1)),
            Expr::Binary { left, right, .. } => {
                stack.push((right, depth + 1));
                stack.push((left, depth + 1));
            }
            Expr::Function { func, args } => {
                let expected = func.arity();
                if args.len() != expected {
                    return Err(ShaderEmitError::WrongArgumentCount {
                        func: *func,
                        expected,
                        found: args.len(),
                    });
                }
                for arg in args.iter().rev() {
                    stack.push((arg, depth + 1));
                }
            }
        }
    }
    Ok(())
}

impl Expr {
    /// Lowers `self` iteratively to a real scalar (`f32`) WGSL expression after verifying every
    /// non-constant variable identifier appears in `allowed_vars` (e.g. `&["x", "y", "z", "t"]`).
    pub fn to_wgsl_3d_with_vars(&self, allowed_vars: &[&str]) -> Result<String, ShaderEmitError> {
        validate_expr_allowed_vars(self, allowed_vars)?;
        self.to_wgsl()
    }
}

fn validate_workgroup_and_dims(cfg: &ComputeKernelConfig) -> Result<(), ShaderEmitError> {
    let [wx, wy, wz] = cfg.workgroup_size;
    let total_threads = u64::from(wx) * u64::from(wy) * u64::from(wz);
    if wx == 0 || wy == 0 || wz == 0 || total_threads > 256 {
        return Err(ShaderEmitError::InvalidIdentifier(format!(
            "workgroup_size({wx},{wy},{wz})"
        )));
    }
    let [nx, ny, nz] = cfg.grid_dims;
    if nx < 2 || ny < 2 || nz < 2 {
        return Err(ShaderEmitError::InvalidIdentifier(format!(
            "grid_dims({nx},{ny},{nz})"
        )));
    }
    Ok(())
}

fn format_bounds_triple(
    xb: [f64; 2],
    yb: [f64; 2],
    zb: [f64; 2],
) -> Result<([String; 3], [String; 3]), ShaderEmitError> {
    if !(xb[0].is_finite()
        && xb[1].is_finite()
        && yb[0].is_finite()
        && yb[1].is_finite()
        && zb[0].is_finite()
        && zb[1].is_finite()
        && xb[1] > xb[0]
        && yb[1] > yb[0]
        && zb[1] > zb[0])
    {
        return Err(ShaderEmitError::NonFiniteLiteral);
    }
    Ok((
        [
            format_shader_f32(xb[0])?,
            format_shader_f32(yb[0])?,
            format_shader_f32(zb[0])?,
        ],
        [
            format_shader_f32(xb[1])?,
            format_shader_f32(yb[1])?,
            format_shader_f32(zb[1])?,
        ],
    ))
}

const COMPUTE_GRID_HELPERS_WGSL: &str = r#"fn om_resolve_dims() -> vec3<u32> {
    if (u_vol.dims.x >= 2u && u_vol.dims.y >= 2u && u_vol.dims.z >= 2u) {
        return u_vol.dims;
    }
    return DEFAULT_DIMS;
}

fn om_voxel_index(gid: vec3<u32>, dims: vec3<u32>) -> u32 {
    return (gid.z * dims.y + gid.y) * dims.x + gid.x;
}

fn om_voxel_world_pos(gid: vec3<u32>, dims: vec3<u32>) -> vec3<f32> {
    let use_u = u_vol.bounds_max.x > u_vol.bounds_min.x
        && u_vol.bounds_max.y > u_vol.bounds_min.y
        && u_vol.bounds_max.z > u_vol.bounds_min.z;
    let bmin = select(DEFAULT_BOUNDS_MIN, u_vol.bounds_min.xyz, use_u);
    let bmax = select(DEFAULT_BOUNDS_MAX, u_vol.bounds_max.xyz, use_u);
    let denom = vec3<f32>(
        f32(max(dims.x - 1u, 1u)),
        f32(max(dims.y - 1u, 1u)),
        f32(max(dims.z - 1u, 1u))
    );
    let uvw = vec3<f32>(gid) / denom;
    return mix(bmin, bmax, uvw);
}"#;

fn collect_compute_param_consts<'a>(
    params: &'a [(String, f64)],
    allowed_vars: &mut Vec<&'a str>,
) -> Result<String, ShaderEmitError> {
    let mut param_consts = String::new();
    for (name, val) in params {
        if !is_valid_compute_param_ident(name) || allowed_vars.contains(&name.as_str()) {
            return Err(ShaderEmitError::InvalidIdentifier(name.clone()));
        }
        let lit = format_shader_f32(*val)?;
        let _ = writeln!(param_consts, "const {name}: f32 = {lit};");
        allowed_vars.push(name.as_str());
    }
    Ok(param_consts)
}

/// Emits a complete standalone WebGPU `@compute` WGSL shader evaluating `expr` over a 3D voxel grid.
pub fn emit_wgsl_3d_compute_kernel(
    expr: &Expr,
    config: &ComputeKernelConfig,
) -> Result<String, ShaderEmitError> {
    validate_workgroup_and_dims(config)?;
    let (bmin, bmax) = format_bounds_triple(config.x_bounds, config.y_bounds, config.z_bounds)?;
    let mut allowed_vars = Vec::with_capacity(4 + config.params.len());
    allowed_vars.extend_from_slice(&["x", "y", "z", "t"]);
    let param_consts = collect_compute_param_consts(&config.params, &mut allowed_vars)?;
    let expr_wgsl = expr.to_wgsl_3d_with_vars(&allowed_vars)?;
    let [nx, ny, nz] = config.grid_dims;
    let [wx, wy, wz] = config.workgroup_size;
    let prelude = wgsl_real_prelude();
    let out = format!(
        "// OxidizeMath Symbolic Shader Foundry — WebGPU 3D Volumetric @compute Kernel\n\
struct VolumeComputeUniforms {{ dims: vec3<u32>, time: f32, bounds_min: vec4<f32>, bounds_max: vec4<f32> }};\n\
@group(0) @binding(0) var<uniform> u_vol: VolumeComputeUniforms;\n\
@group(0) @binding(1) var<storage, read> field_in: array<f32>;\n\
@group(0) @binding(2) var<storage, read_write> field_out: array<f32>;\n\
const DEFAULT_DIMS: vec3<u32> = vec3<u32>({nx}u, {ny}u, {nz}u);\n\
const DEFAULT_BOUNDS_MIN: vec3<f32> = vec3<f32>({}, {}, {});\n\
const DEFAULT_BOUNDS_MAX: vec3<f32> = vec3<f32>({}, {}, {});\n\
{param_consts}{prelude}\n{COMPUTE_GRID_HELPERS_WGSL}\n\n\
@compute @workgroup_size({wx}, {wy}, {wz})\n\
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
    let dims = om_resolve_dims();\n\
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {{ return; }}\n\
    let pos = om_voxel_world_pos(gid, dims);\n\
    let x: f32 = pos.x; let y: f32 = pos.y; let z: f32 = pos.z; let t: f32 = u_vol.time;\n\
    let idx = om_voxel_index(gid, dims);\n\
    let _prev: f32 = field_in[min(idx, max(arrayLength(&field_in), 1u) - 1u)];\n\
    let raw_val: f32 = {expr_wgsl};\n\
    let finite_ok = raw_val == raw_val && abs(raw_val) <= 3.4e38;\n\
    field_out[idx] = select(0.0, raw_val, finite_ok);\n\
}}\n",
        bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2],
    );
    if out.len() > MAX_GLSL_OUTPUT_BYTES {
        return Err(ShaderEmitError::OutputLimitExceeded(MAX_GLSL_OUTPUT_BYTES));
    }
    Ok(out)
}
