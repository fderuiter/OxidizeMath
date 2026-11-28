//! Bounded WebGPU `@compute @workgroup_size(...)` WGSL kernel emitters for PDE stencils,
//! quantum wavepackets, ODE RK4 rollouts, and Monte Carlo payoff ensembles.

use math_commons::ast::{
    format_shader_f32, wgsl_real_prelude, Expr, ShaderEmitError, MAX_GLSL_OUTPUT_BYTES,
};
use std::fmt::Write as _;

/// Kinetics law lowered into a standalone 2D reaction-diffusion `@compute` WGSL kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PdeComputeKinetics {
    /// Schnakenberg activator-inhibitor kinetics: $du/dt = D_u \nabla^2 u + a - u + u^2 v$.
    Schnakenberg,
    /// Gray-Scott autocatalytic kinetics: $du/dt = D_u \nabla^2 u - u v^2 + F(1 - u)$.
    GrayScott,
}

/// Calibrated parameters for [`emit_pde_stencil_compute_wgsl`].
#[derive(Clone, Copy, Debug)]
pub struct PdeStencilComputeSpec {
    /// Grid dimensions `[nx, ny]`.
    pub grid_dims: [u32; 2],
    /// Explicit integration time step $\Delta t$.
    pub dt: f64,
    /// Activator diffusion rate $D_u$.
    pub d_u: f64,
    /// Inhibitor diffusion rate $D_v$.
    pub d_v: f64,
    /// Schnakenberg $a$ or Gray-Scott feed rate $F$.
    pub rate_primary: f64,
    /// Schnakenberg $b$ or Gray-Scott kill rate $k$.
    pub rate_secondary: f64,
    /// Reaction kinetics variant.
    pub kinetics: PdeComputeKinetics,
}

/// Calibrated parameters for [`emit_quantum_wave_compute_wgsl`].
#[derive(Clone, Debug)]
pub struct QuantumWaveComputeSpec<'a> {
    /// Number of 1D spatial lattice points.
    pub grid_points: u32,
    /// Spatial domain bounds `[x_min, x_max]`.
    pub domain: [f64; 2],
    /// Integration time step $\Delta t$.
    pub dt: f64,
    /// Potential height $V_0$.
    pub v0: f64,
    /// Characteristic barrier/well width $w$.
    pub width: f64,
    /// Optional symbolic potential $V(x)$ lowered with `MAX_EVAL_DEPTH = 64` and `MAX_AST_NODES = 256`.
    pub potential_expr: Option<&'a Expr>,
    /// Bound scalar parameters `(name, value)` for `potential_expr`.
    pub params: Vec<(String, f64)>,
}

/// Calibrated parameters for [`emit_ode_rollout_compute_wgsl`].
#[derive(Clone, Debug)]
pub struct OdeRolloutComputeSpec<'a> {
    /// Primary RHS expression $dy/dt = f(t, y, v)$.
    pub dy_expr: &'a Expr,
    /// Optional coupled second-state RHS expression $dv/dt = g(t, y, v)$.
    pub dv_expr: Option<&'a Expr>,
    /// State variable names `[y_name, v_name]` exposed to the AST expressions.
    pub state_vars: [&'static str; 2],
    /// Explicit RK4 time step $\Delta t$.
    pub dt: f64,
    /// Number of RK4 steps per GPU invocation (`1..=4096`).
    pub steps: u32,
    /// Baseline initial state `[y0, v0]`.
    pub initial_state: [f64; 2],
    /// Bound scalar parameters `(name, value)`.
    pub params: Vec<(String, f64)>,
}

/// Payoff family lowered into a standalone WebGPU Monte Carlo `@compute` kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MonteCarloPayoffKind {
    /// 4D discounted geometric basket call option.
    BasketCall4D,
    /// 3D smooth separable exponential product $\prod_{j=1}^3 e^{u_j}$.
    SmoothExpProduct3D,
    /// 1D skewed lognormal wealth return $\exp(\sigma Z - \tfrac{1}{2}\sigma^2)$.
    SkewedLognormal1D,
}

/// Calibrated parameters for [`emit_monte_carlo_compute_wgsl`].
#[derive(Clone, Copy, Debug)]
pub struct MonteCarloComputeSpec {
    /// Number of Monte Carlo trajectories dispatched across workgroups.
    pub num_samples: u32,
    /// Exponential importance-sampling tilt $\theta$.
    pub importance_tilt: f64,
    /// Deterministic base PRNG seed.
    pub seed: u64,
    /// Benchmark payoff variant.
    pub payoff: MonteCarloPayoffKind,
}

fn check_wgsl_budget(src: String) -> Result<String, String> {
    if src.len() > MAX_GLSL_OUTPUT_BYTES {
        return Err(ShaderEmitError::OutputLimitExceeded(MAX_GLSL_OUTPUT_BYTES).to_string());
    }
    Ok(src)
}

fn fmt_f32(val: f64) -> Result<String, String> {
    format_shader_f32(val).map_err(|e| e.to_string())
}

fn emit_param_consts<'a>(
    params: &'a [(String, f64)],
    reserved: &[&str],
    allowed: &mut Vec<&'a str>,
) -> Result<String, String> {
    let mut out = String::new();
    for (name, val) in params {
        let key = name.as_str();
        let valid_ident = !key.is_empty()
            && !reserved.contains(&key)
            && !allowed.contains(&key)
            && !key.starts_with("om_")
            && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && !key.as_bytes()[0].is_ascii_digit();
        if !valid_ident {
            return Err(ShaderEmitError::InvalidIdentifier(name.clone()).to_string());
        }
        let lit = fmt_f32(*val)?;
        let _ = writeln!(out, "const {name}: f32 = {lit};");
        allowed.push(key);
    }
    Ok(out)
}

/// Emits a bounded 2D reaction-diffusion stencil WebGPU `@compute @workgroup_size(8, 8, 1)` WGSL kernel.
pub fn emit_pde_stencil_compute_wgsl(spec: &PdeStencilComputeSpec) -> Result<String, String> {
    let nx = spec.grid_dims[0].max(4);
    let ny = spec.grid_dims[1].max(4);
    let (dt, du, dv) = (fmt_f32(spec.dt)?, fmt_f32(spec.d_u)?, fmt_f32(spec.d_v)?);
    let (p0, p1) = (fmt_f32(spec.rate_primary)?, fmt_f32(spec.rate_secondary)?);
    let reaction = match spec.kinetics {
        PdeComputeKinetics::Schnakenberg => {
            "let u2v = c.x * c.x * c.y;\n    let ru = params.p0 - c.x + u2v;\n    let rv = params.p1 - u2v;"
        }
        PdeComputeKinetics::GrayScott => {
            "let uv2 = c.x * c.y * c.y;\n    let ru = -uv2 + params.p0 * (1.0 - c.x);\n    let rv = uv2 - (params.p0 + params.p1) * c.y;"
        }
    };
    check_wgsl_budget(format!(
        "// OxidizeMath Symbolic Shader Foundry — 2D Reaction-Diffusion @compute Kernel\n\
struct PdeUniforms {{ dims: vec2<u32>, dt: f32, du: f32, dv: f32, p0: f32, p1: f32, _pad: f32 }};\n\
@group(0) @binding(0) var<uniform> params: PdeUniforms;\n\
@group(0) @binding(1) var<storage, read> state_in: array<vec2<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> state_out: array<vec2<f32>>;\n\
const DEFAULT_DIMS: vec2<u32> = vec2<u32>({nx}u, {ny}u);\n\
const DEFAULT_DT: f32 = {dt};\n\
const DEFAULT_DU: f32 = {du};\n\
const DEFAULT_DV: f32 = {dv};\n\
const DEFAULT_P0: f32 = {p0};\n\
const DEFAULT_P1: f32 = {p1};\n\
fn cell_idx(ix: u32, iy: u32, d: vec2<u32>) -> u32 {{ return iy * d.x + ix; }}\n\
@compute @workgroup_size(8, 8, 1)\n\
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
    let d = select(DEFAULT_DIMS, params.dims, params.dims.x >= 4u && params.dims.y >= 4u);\n\
    if (gid.x >= d.x || gid.y >= d.y) {{ return; }}\n\
    let xm = select(gid.x - 1u, d.x - 1u, gid.x == 0u);\n\
    let xp = select(gid.x + 1u, 0u, gid.x + 1u == d.x);\n\
    let ym = select(gid.y - 1u, d.y - 1u, gid.y == 0u);\n\
    let yp = select(gid.y + 1u, 0u, gid.y + 1u == d.y);\n\
    let idx = cell_idx(gid.x, gid.y, d);\n\
    let c = state_in[idx];\n\
    let lap = state_in[cell_idx(xm, gid.y, d)] + state_in[cell_idx(xp, gid.y, d)]\n\
            + state_in[cell_idx(gid.x, ym, d)] + state_in[cell_idx(gid.x, yp, d)] - 4.0 * c;\n\
    {reaction}\n\
    let step_dt = select(DEFAULT_DT, params.dt, params.dt > 0.0);\n\
    let next_uv = c + step_dt * vec2<f32>(params.du * lap.x + ru, params.dv * lap.y + rv);\n\
    state_out[idx] = clamp(next_uv, vec2<f32>(0.0), vec2<f32>(12.0));\n\
}}\n"
    ))
}

/// Emits a bounded 1D Schrödinger wavepacket WebGPU `@compute @workgroup_size(64, 1, 1)` WGSL kernel.
pub fn emit_quantum_wave_compute_wgsl(spec: &QuantumWaveComputeSpec<'_>) -> Result<String, String> {
    let n = spec.grid_points.max(16);
    let (xmin, xmax) = (fmt_f32(spec.domain[0])?, fmt_f32(spec.domain[1])?);
    let (dt, v0, width) = (
        fmt_f32(spec.dt)?,
        fmt_f32(spec.v0)?,
        fmt_f32(spec.width.max(0.05))?,
    );
    let reserved = ["cs_main", "gid", "u_wave", "psi_in", "psi_out", "x", "idx"];
    let mut allowed = vec!["x", "v0", "width"];
    let param_consts = emit_param_consts(&spec.params, &reserved, &mut allowed)?;
    let pot_wgsl = match spec.potential_expr {
        Some(expr) => expr
            .to_wgsl_with_vars(&allowed)
            .map_err(|e| e.to_string())?,
        None => "select(0.0, v0, abs(x) <= 0.5 * width)".to_string(),
    };
    let prelude = wgsl_real_prelude();
    check_wgsl_budget(format!(
        "// OxidizeMath Symbolic Shader Foundry — 1D Quantum Wavepacket @compute Kernel\n\
struct WaveUniforms {{ n_pts: u32, dt: f32, x_min: f32, x_max: f32 }};\n\
@group(0) @binding(0) var<uniform> u_wave: WaveUniforms;\n\
@group(0) @binding(1) var<storage, read> psi_in: array<vec2<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> psi_out: array<vec2<f32>>;\n\
const DEFAULT_N: u32 = {n}u;\n\
const DEFAULT_DT: f32 = {dt};\n\
const DEFAULT_XMIN: f32 = {xmin};\n\
const DEFAULT_XMAX: f32 = {xmax};\n\
const v0: f32 = {v0};\n\
const width: f32 = {width};\n\
{param_consts}{prelude}\n\
@compute @workgroup_size(64, 1, 1)\n\
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
    let n_pts = select(DEFAULT_N, u_wave.n_pts, u_wave.n_pts >= 16u);\n\
    let idx = gid.x;\n\
    if (idx >= n_pts) {{ return; }}\n\
    if (idx == 0u || idx + 1u == n_pts) {{ psi_out[idx] = vec2<f32>(0.0, 0.0); return; }}\n\
    let dx = (DEFAULT_XMAX - DEFAULT_XMIN) / f32(n_pts - 1u);\n\
    let x: f32 = DEFAULT_XMIN + f32(idx) * dx;\n\
    let pot_v = clamp({pot_wgsl}, -500.0, 500.0);\n\
    let c = psi_in[idx];\n\
    let lap = (psi_in[idx - 1u] - 2.0 * c + psi_in[idx + 1u]) / max(dx * dx, 1e-6);\n\
    let h_psi = -0.5 * lap + pot_v * c;\n\
    let step_dt = select(DEFAULT_DT, u_wave.dt, u_wave.dt > 0.0);\n\
    psi_out[idx] = c + step_dt * vec2<f32>(h_psi.y, -h_psi.x);\n\
}}\n"
    ))
}

/// Emits a bounded RK4 initial-value rollout WebGPU `@compute @workgroup_size(64, 1, 1)` WGSL kernel.
pub fn emit_ode_rollout_compute_wgsl(spec: &OdeRolloutComputeSpec<'_>) -> Result<String, String> {
    let [v1, v2] = spec.state_vars;
    let reserved = ["cs_main", "gid", "u_ode", "ic_in", "traj_out", "om_rhs"];
    let mut allowed = vec!["t", v1, v2];
    let param_consts = emit_param_consts(&spec.params, &reserved, &mut allowed)?;
    let dy_wgsl = spec
        .dy_expr
        .to_wgsl_with_vars(&allowed)
        .map_err(|e| e.to_string())?;
    let dv_wgsl = match spec.dv_expr {
        Some(expr) => expr
            .to_wgsl_with_vars(&allowed)
            .map_err(|e| e.to_string())?,
        None => "0.0".to_string(),
    };
    let (dt, y0, v0) = (
        fmt_f32(spec.dt)?,
        fmt_f32(spec.initial_state[0])?,
        fmt_f32(spec.initial_state[1])?,
    );
    let steps = spec.steps.clamp(1, 4_096);
    let prelude = wgsl_real_prelude();
    check_wgsl_budget(format!(
        "// OxidizeMath Symbolic Shader Foundry — RK4 Ensemble Rollout @compute Kernel\n\
struct OdeUniforms {{ n_trajectories: u32, steps: u32, dt: f32, _pad: f32 }};\n\
@group(0) @binding(0) var<uniform> u_ode: OdeUniforms;\n\
@group(0) @binding(1) var<storage, read> ic_in: array<vec2<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> traj_out: array<vec2<f32>>;\n\
const DEFAULT_STEPS: u32 = {steps}u;\n\
const DEFAULT_DT: f32 = {dt};\n\
const DEFAULT_IC: vec2<f32> = vec2<f32>({y0}, {v0});\n\
{param_consts}{prelude}\n\
fn om_rhs(t: f32, st: vec2<f32>) -> vec2<f32> {{\n\
    let {v1}: f32 = st.x;\n\
    let {v2}: f32 = st.y;\n\
    return vec2<f32>({dy_wgsl}, {dv_wgsl});\n\
}}\n\
@compute @workgroup_size(64, 1, 1)\n\
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
    if (u_ode.n_trajectories > 0u && gid.x >= u_ode.n_trajectories) {{ return; }}\n\
    let dt_s = select(DEFAULT_DT, u_ode.dt, u_ode.dt > 0.0);\n\
    let n_steps = min(select(DEFAULT_STEPS, u_ode.steps, u_ode.steps > 0u), 4096u);\n\
    var st = select(DEFAULT_IC, ic_in[gid.x], u_ode.n_trajectories > 0u);\n\
    var t: f32 = 0.0;\n\
    for (var step: u32 = 0u; step < n_steps; step = step + 1u) {{\n\
        let k1 = om_rhs(t, st);\n\
        let k2 = om_rhs(t + 0.5 * dt_s, st + 0.5 * dt_s * k1);\n\
        let k3 = om_rhs(t + 0.5 * dt_s, st + 0.5 * dt_s * k2);\n\
        let k4 = om_rhs(t + dt_s, st + dt_s * k3);\n\
        st = st + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);\n\
        t = t + dt_s;\n\
    }}\n\
    traj_out[gid.x] = st;\n\
}}\n"
    ))
}

/// Emits a bounded parallel Monte Carlo payoff rollout WebGPU `@compute @workgroup_size(64, 1, 1)` WGSL kernel.
pub fn emit_monte_carlo_compute_wgsl(spec: &MonteCarloComputeSpec) -> Result<String, String> {
    let n = spec.num_samples.clamp(32, 1_048_576);
    let tilt = fmt_f32(spec.importance_tilt.clamp(0.0, 3.5))?;
    let seed_lo = (spec.seed & 0xFFFF_FFFF) as u32;
    let payoff_body = match spec.payoff {
        MonteCarloPayoffKind::BasketCall4D => {
            "let z_sum = z0 + z1 + z2 + z3 + 4.0 * tilt;\n\
    let geom = 100.0 * exp((0.05 - 0.045) + 0.30 * 0.25 * z_sum);\n\
    let raw = exp(-0.05) * max(geom - 100.0, 0.0);\n\
    let lr = exp(-tilt * (z0 + z1 + z2 + z3) - 2.0 * tilt * tilt);"
        }
        MonteCarloPayoffKind::SmoothExpProduct3D => {
            "let raw = exp(u0) * exp(u1) * exp(u2);\n    let lr = 1.0;"
        }
        MonteCarloPayoffKind::SkewedLognormal1D => {
            "let z_t = z0 + tilt;\n\
    let raw = exp(0.85 * z_t - 0.5 * 0.85 * 0.85);\n\
    let lr = exp(-tilt * z0 - 0.5 * tilt * tilt);"
        }
    };
    check_wgsl_budget(format!(
        "// OxidizeMath Symbolic Shader Foundry — Monte Carlo Variance-Reduction @compute Kernel\n\
struct McUniforms {{ n_samples: u32, seed: u32, tilt: f32, _pad: f32 }};\n\
@group(0) @binding(0) var<uniform> u_mc: McUniforms;\n\
@group(0) @binding(1) var<storage, read_write> payoffs_out: array<f32>;\n\
const DEFAULT_SAMPLES: u32 = {n}u;\n\
const DEFAULT_SEED: u32 = {seed_lo}u;\n\
const DEFAULT_TILT: f32 = {tilt};\n\
fn pcg_hash(v: u32) -> u32 {{\n\
    let st = v * 747796405u + 2891336453u;\n\
    let wd = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u;\n\
    return (wd >> 22u) ^ wd;\n\
}}\n\
fn rand_open01(rng: ptr<function, u32>) -> f32 {{\n\
    *rng = pcg_hash(*rng);\n\
    return (f32(*rng >> 8u) + 0.5) * (1.0 / 16777216.0);\n\
}}\n\
fn box_muller(u_a: f32, u_b: f32) -> vec2<f32> {{\n\
    let r = sqrt(-2.0 * log(max(u_a, 1e-7)));\n\
    let th = 6.2831853 * u_b;\n\
    return vec2<f32>(r * cos(th), r * sin(th));\n\
}}\n\
@compute @workgroup_size(64, 1, 1)\n\
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
    let total = select(DEFAULT_SAMPLES, u_mc.n_samples, u_mc.n_samples >= 32u);\n\
    if (gid.x >= total) {{ return; }}\n\
    var rng: u32 = pcg_hash(gid.x ^ select(DEFAULT_SEED, u_mc.seed, u_mc.seed != 0u));\n\
    let u0 = rand_open01(&rng); let u1 = rand_open01(&rng);\n\
    let u2 = rand_open01(&rng); let u3 = rand_open01(&rng);\n\
    let g01 = box_muller(u0, u1); let g23 = box_muller(u2, u3);\n\
    let z0 = g01.x; let z1 = g01.y; let z2 = g23.x; let z3 = g23.y;\n\
    let tilt = select(DEFAULT_TILT, u_mc.tilt, u_mc.tilt >= 0.0);\n\
    {payoff_body}\n\
    payoffs_out[gid.x] = raw * lr;\n\
}}\n"
    ))
}
