//! Parameter-calibrated LaTeX, Rust 2024, and Python (`scipy`) code export for [`super::OdeSolverTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::OdeSolverTool;
use crate::framework::{CodeArtifact, InteractiveTool};
use crate::presets::OdePreset;
use math_commons::ast::{
    format_latex_scalar, format_python_float, format_rust_f64, parse_equation, Expr,
};
use std::collections::HashMap;

struct ScalarOdeSpec<'a> {
    expr: &'a Expr,
    symbolic_rhs: &'a str,
    params: &'a HashMap<String, f64>,
    y0: f64,
    dt: f64,
    total_time: f64,
    steps: usize,
}

struct CoupledOdeExportSpec<'a> {
    dy_sym: String,
    dv_sym: String,
    dy_cal: String,
    dv_cal: String,
    rust_rhs_defs: String,
    py_dy: String,
    py_dv: String,
    init: [f64; 2],
    dt: f64,
    total_time: f64,
    steps: usize,
    comment: &'a str,
}

impl OdeSolverTool {
    pub(super) fn active_param_map(&self) -> HashMap<String, f64> {
        let mut params: HashMap<String, f64> = self.math_input.params_map().into_iter().collect();
        params.insert("k".to_string(), self.param_k);
        params.insert("r".to_string(), self.param_r);
        params.insert("cap_k".to_string(), self.param_cap_k);
        params.insert("K".to_string(), self.param_cap_k);
        params
    }

    pub(super) fn build_code_artifact(&self) -> Option<CodeArtifact> {
        if !self.has_finite_parameters() {
            return None;
        }
        let steps = crate::presets::checked_ode_step_count(self.dt, self.total_time)?;
        let params = self.active_param_map();

        match self.preset {
            OdePreset::Custom if self.custom_coupled => self.build_coupled_custom_artifact(steps),
            OdePreset::Custom => {
                let expr = self.math_input.expr()?;
                let sym = expr.to_latex();
                build_scalar_ode_artifact(&ScalarOdeSpec {
                    expr,
                    symbolic_rhs: &sym,
                    params: &params,
                    y0: self.ic_y0,
                    dt: self.dt,
                    total_time: self.total_time,
                    steps,
                })
            }
            OdePreset::Exponential => {
                let (_, expr) = parse_equation("k * y").ok()?;
                build_scalar_ode_artifact(&ScalarOdeSpec {
                    expr: &expr,
                    symbolic_rhs: r"k \cdot y",
                    params: &params,
                    y0: self.ic_y0,
                    dt: self.dt,
                    total_time: self.total_time,
                    steps,
                })
            }
            OdePreset::LogisticGrowth => {
                let (_, expr) = parse_equation("r * y * (1 - y / cap_k)").ok()?;
                build_scalar_ode_artifact(&ScalarOdeSpec {
                    expr: &expr,
                    symbolic_rhs: r"r \cdot y \cdot \left(1 - \frac{y}{K}\right)",
                    params: &params,
                    y0: self.ic_y0,
                    dt: self.dt,
                    total_time: self.total_time,
                    steps,
                })
            }
            OdePreset::HarmonicOscillator => build_harmonic_ode_artifact(
                self.param_k,
                [self.ic_y0, self.ic_v0],
                self.dt,
                self.total_time,
                steps,
            ),
        }
    }

    fn build_coupled_custom_artifact(&self, steps: usize) -> Option<CodeArtifact> {
        if !self.coupled_bank.is_valid() {
            return None;
        }
        let dy_expr = self.coupled_bank.bars().first()?.expr()?;
        let dv_expr = self.coupled_bank.bars().get(1)?.expr()?;
        let params: HashMap<String, f64> = self.coupled_bank.params_map().into_iter().collect();
        let vars = ["t", "y", "v"];
        let dy_fn = dy_expr.to_rust_fn("dy_rhs", &vars, &params).ok()?;
        let dv_fn = dv_expr.to_rust_fn("dv_rhs", &vars, &params).ok()?;
        let rust_rhs_defs = format!(
            "{dy_fn}\n\n{dv_fn}\n\n\
             pub fn rhs(t: f64, state: [f64; 2]) -> [f64; 2] {{\n\
             \x20   [dy_rhs(t, state[0], state[1]), dv_rhs(t, state[0], state[1])]\n\
             }}"
        );
        build_coupled_2d_artifact(&CoupledOdeExportSpec {
            dy_sym: dy_expr.to_latex(),
            dv_sym: dv_expr.to_latex(),
            dy_cal: dy_expr.to_latex_with_params(&params).ok()?,
            dv_cal: dv_expr.to_latex_with_params(&params).ok()?,
            rust_rhs_defs,
            py_dy: dy_expr.to_python_expr_with_vars(&vars, &params).ok()?,
            py_dv: dv_expr.to_python_expr_with_vars(&vars, &params).ok()?,
            init: [self.ic_y0, self.ic_v0],
            dt: self.dt,
            total_time: self.total_time,
            steps,
            comment: "Coupled 2-state",
        })
    }
}

fn build_scalar_ode_artifact(spec: &ScalarOdeSpec<'_>) -> Option<CodeArtifact> {
    let cal_rhs_tex = spec.expr.to_latex_with_params(spec.params).ok()?;
    let y0_tex = format_latex_scalar(spec.y0).ok()?;
    let dt_tex = format_latex_scalar(spec.dt).ok()?;
    let t_tex = format_latex_scalar(spec.total_time).ok()?;

    let symbolic_latex = format!(
        "\\begin{{aligned}}\n\\frac{{dy}}{{dt}} &= {} \\\\\ny(0) &= y_0\n\\end{{aligned}}",
        spec.symbolic_rhs
    );
    let calibrated_latex = format!(
        "\\begin{{aligned}}\n\\frac{{dy}}{{dt}} &= {cal_rhs_tex} \\\\\ny(0) &= {y0_tex}, \\quad \\Delta t = {dt_tex}, \\quad t \\in [0, {t_tex}]\n\\end{{aligned}}"
    );
    let rust_source = build_scalar_ode_rust(spec)?;
    let python_source = build_scalar_ode_python(spec)?;

    Some(CodeArtifact {
        symbolic_latex,
        calibrated_latex,
        rust_source,
        python_source,
    })
}

fn build_scalar_ode_rust(spec: &ScalarOdeSpec<'_>) -> Option<String> {
    let rhs_fn = spec.expr.to_rust_fn("rhs", &["t", "y"], spec.params).ok()?;
    let y0_rs = format_rust_f64(spec.y0).ok()?;
    let dt_rs = format_rust_f64(spec.dt).ok()?;
    let steps = spec.steps;
    Some(format!(
        "// Standalone Rust 2024 ODE solver generated by OxidizeMath (zero external crates)\n\
         {rhs_fn}\n\n\
         pub fn rk4_step(t: f64, y: f64, dt: f64) -> f64 {{\n\
         \x20   let k1 = rhs(t, y);\n\
         \x20   let k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1);\n\
         \x20   let k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2);\n\
         \x20   let k4 = rhs(t + dt, y + dt * k3);\n\
         \x20   y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)\n\
         }}\n\n\
         fn main() {{\n\
         \x20   let dt: f64 = {dt_rs};\n\
         \x20   let steps: usize = {steps};\n\
         \x20   let mut y: f64 = {y0_rs};\n\
         \x20   println!(\"t,y\");\n\
         \x20   for step in 0..=steps {{\n\
         \x20       let t = (step as f64) * dt;\n\
         \x20       println!(\"{{t:.6}},{{y:.9}}\");\n\
         \x20       if step < steps {{\n\
         \x20           y = rk4_step(t, y, dt);\n\
         \x20       }}\n\
         \x20   }}\n\
         }}\n"
    ))
}

fn build_scalar_ode_python(spec: &ScalarOdeSpec<'_>) -> Option<String> {
    let py_rhs = spec.expr.to_python_ode_rhs(spec.params).ok()?;
    let y0_py = format_python_float(spec.y0).ok()?;
    let t_py = format_python_float(spec.total_time).ok()?;
    let num_pts = spec.steps.saturating_add(1);
    Some(format!(
        "\"\"\"Standalone Python ODE solver generated by OxidizeMath.\"\"\"\n\
         import numpy as np\n\
         from scipy.integrate import solve_ivp\n\n\
         def rhs(t: float, y: np.ndarray) -> list[float]:\n\
         \x20   return [{py_rhs}]\n\n\
         t_eval = np.linspace(0.0, {t_py}, {num_pts})\n\
         sol = solve_ivp(rhs, (0.0, {t_py}), [{y0_py}], t_eval=t_eval, method=\"RK45\", rtol=1e-9, atol=1e-12)\n\n\
         if __name__ == \"__main__\":\n\
         \x20   print(f\"Final state y({{sol.t[-1]:.4f}}) = {{sol.y[0, -1]:.9f}}\")\n"
    ))
}

fn build_harmonic_ode_artifact(
    k: f64,
    init: [f64; 2],
    dt: f64,
    total_time: f64,
    steps: usize,
) -> Option<CodeArtifact> {
    let k_tex = format_latex_scalar(k).ok()?;
    let k_rs = format_rust_f64(k).ok()?;
    let k_py = format_python_float(k).ok()?;
    let rust_rhs_defs = format!(
        "pub fn rhs(_t: f64, state: [f64; 2]) -> [f64; 2] {{\n\
         \x20   [state[1], -({k_rs}) * state[0]]\n\
         }}"
    );
    build_coupled_2d_artifact(&CoupledOdeExportSpec {
        dy_sym: "v".to_string(),
        dv_sym: r"-k \cdot y".to_string(),
        dy_cal: "v".to_string(),
        dv_cal: format!(r"-{k_tex} \cdot y"),
        rust_rhs_defs,
        py_dy: "v".to_string(),
        py_dv: format!("-({k_py}) * y"),
        init,
        dt,
        total_time,
        steps,
        comment: "Harmonic Oscillator",
    })
}

fn build_coupled_2d_artifact(spec: &CoupledOdeExportSpec<'_>) -> Option<CodeArtifact> {
    let [y0, v0] = spec.init;
    let y0_tex = format_latex_scalar(y0).ok()?;
    let v0_tex = format_latex_scalar(v0).ok()?;
    let dt_tex = format_latex_scalar(spec.dt).ok()?;
    let t_tex = format_latex_scalar(spec.total_time).ok()?;

    let symbolic_latex = format!(
        "\\begin{{aligned}}\n\\frac{{dy}}{{dt}} &= {} \\\\\n\\frac{{dv}}{{dt}} &= {} \\\\\ny(0) &= y_0, \\quad v(0) = v_0\n\\end{{aligned}}",
        spec.dy_sym, spec.dv_sym
    );
    let calibrated_latex = format!(
        "\\begin{{aligned}}\n\\frac{{dy}}{{dt}} &= {} \\\\\n\\frac{{dv}}{{dt}} &= {} \\\\\ny(0) &= {y0_tex}, \\quad v(0) = {v0_tex}, \\quad \\Delta t = {dt_tex}, \\quad t \\in [0, {t_tex}]\n\\end{{aligned}}",
        spec.dy_cal, spec.dv_cal
    );
    let rust_source = build_coupled_2d_rust(spec)?;
    let python_source = build_coupled_2d_python(spec)?;

    Some(CodeArtifact {
        symbolic_latex,
        calibrated_latex,
        rust_source,
        python_source,
    })
}

fn build_coupled_2d_rust(spec: &CoupledOdeExportSpec<'_>) -> Option<String> {
    let [y0, v0] = spec.init;
    let y0_rs = format_rust_f64(y0).ok()?;
    let v0_rs = format_rust_f64(v0).ok()?;
    let dt_rs = format_rust_f64(spec.dt).ok()?;
    let steps = spec.steps;
    let comment = spec.comment;
    let rhs_defs = &spec.rust_rhs_defs;
    Some(format!(
        "// Standalone Rust 2024 {comment} RK4 solver generated by OxidizeMath\n\
         {rhs_defs}\n\n\
         pub fn rk4_step(t: f64, state: [f64; 2], dt: f64) -> [f64; 2] {{\n\
         \x20   let k1 = rhs(t, state);\n\
         \x20   let k2 = rhs(t + 0.5 * dt, [state[0] + 0.5 * dt * k1[0], state[1] + 0.5 * dt * k1[1]]);\n\
         \x20   let k3 = rhs(t + 0.5 * dt, [state[0] + 0.5 * dt * k2[0], state[1] + 0.5 * dt * k2[1]]);\n\
         \x20   let k4 = rhs(t + dt, [state[0] + dt * k3[0], state[1] + dt * k3[1]]);\n\
         \x20   [\n\
         \x20       state[0] + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),\n\
         \x20       state[1] + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),\n\
         \x20   ]\n\
         }}\n\n\
         fn main() {{\n\
         \x20   let dt: f64 = {dt_rs};\n\
         \x20   let steps: usize = {steps};\n\
         \x20   let mut state: [f64; 2] = [{y0_rs}, {v0_rs}];\n\
         \x20   for step in 0..=steps {{\n\
         \x20       let t = (step as f64) * dt;\n\
         \x20       println!(\"{{t:.6}},{{:.9}},{{:.9}}\", state[0], state[1]);\n\
         \x20       if step < steps {{\n\
         \x20           state = rk4_step(t, state, dt);\n\
         \x20       }}\n\
         \x20   }}\n\
         }}\n"
    ))
}

fn build_coupled_2d_python(spec: &CoupledOdeExportSpec<'_>) -> Option<String> {
    let [y0, v0] = spec.init;
    let y0_py = format_python_float(y0).ok()?;
    let v0_py = format_python_float(v0).ok()?;
    let t_py = format_python_float(spec.total_time).ok()?;
    let num_pts = spec.steps.saturating_add(1);
    let comment = spec.comment;
    let py_dy = &spec.py_dy;
    let py_dv = &spec.py_dv;
    Some(format!(
        "\"\"\"Standalone Python {comment} solver generated by OxidizeMath.\"\"\"\n\
         import numpy as np\n\
         from scipy.integrate import solve_ivp\n\n\
         def rhs(t: float, state: np.ndarray) -> list[float]:\n\
         \x20   y, v = float(state[0]), float(state[1])\n\
         \x20   return [float({py_dy}), float({py_dv})]\n\n\
         t_eval = np.linspace(0.0, {t_py}, {num_pts})\n\
         sol = solve_ivp(rhs, (0.0, {t_py}), [{y0_py}, {v0_py}], t_eval=t_eval, method=\"RK45\", rtol=1e-9, atol=1e-12)\n\n\
         if __name__ == \"__main__\":\n\
         \x20   print(f\"Final state y({{sol.t[-1]:.4f}}) = {{sol.y[0, -1]:.9f}}\")\n"
    ))
}
