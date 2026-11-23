//! Schwarz-Christoffel conformal polygon transformation (`SchwarzChristoffel`,
//! `conformal_scale_factor`, `area_magnification`), Schwarz reflection (`schwarz_reflect`),
//! and Cauchy / residue contour integration (`path_integrate`, `cauchy_integral_value`,
//! `cauchy_derivative`, `residue_simple_limit`, `residue_via_integral`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MeromorphicPreset, PolygonPreset, RiemannSurfaceStudio};
use math_commons::ast::{parse, CompiledExpr};
use math_explorer::pure_math::analysis::complex::continuation::schwarz_reflect;
use math_explorer::pure_math::analysis::complex::integration::{
    cauchy_derivative, cauchy_integral_value, path_integrate, residue_simple_limit,
    residue_via_integral,
};
use math_explorer::pure_math::analysis::complex::mapping::{
    area_magnification, conformal_scale_factor, SchwarzChristoffel,
};
use num_complex::Complex64;
use std::f64::consts::TAU;

fn build_sc_map(preset: PolygonPreset, scale: f64) -> SchwarzChristoffel {
    let a = Complex64::new(scale.clamp(0.3, 2.5), 0.0);
    let b = Complex64::new(0.0, 0.0);
    match preset {
        PolygonPreset::EquilateralTriangle => {
            SchwarzChristoffel::new(vec![-1.0, 1.0], vec![-2.0 / 3.0, -2.0 / 3.0], a, b)
        }
        PolygonPreset::SteppedChannel => {
            SchwarzChristoffel::new(vec![-1.0, 0.0, 1.0], vec![-0.5, 0.5, -0.5], a, b)
        }
        PolygonPreset::SymmetricQuad => SchwarzChristoffel::new(
            vec![-1.5, -0.5, 0.5, 1.5],
            vec![-0.45, -0.55, -0.55, -0.45],
            a,
            b,
        ),
    }
}

fn compile_custom_real_numerator(expr_str: &str) -> Option<CompiledExpr> {
    let ast = parse(expr_str).ok()?;
    CompiledExpr::compile_with_vars(&ast, &["x"]).ok()
}

fn eval_real_analytic_upper(compiled: Option<&CompiledExpr>, z: Complex64) -> Complex64 {
    // Evaluate real-coefficient analytic series approximation around Re(z) using real derivatives
    // on Im(z) >= 0 so f(x) is strictly real on Im(z) = 0 (satisfying Schwarz reflection hypothesis).
    let Some(expr) = compiled else {
        return z.cos() + 0.5 * z * z;
    };
    let x = z.re;
    let y = z.im;
    let h = 1e-4_f64;
    let f0 = expr.eval_slice(&[x]);
    let fp = expr.eval_slice(&[x + h]);
    let fm = expr.eval_slice(&[x - h]);
    let fpp = expr.eval_slice(&[x + 2.0 * h]);
    let fmm = expr.eval_slice(&[x - 2.0 * h]);
    if ![f0, fp, fm, fpp, fmm].iter().all(|v| v.is_finite()) {
        return Complex64::new(0.0, 0.0);
    }
    let d1 = (fp - fm) / (2.0 * h);
    let d2 = (fp - 2.0 * f0 + fm) / (h * h);
    let d3 = (fpp - 2.0 * fp + 2.0 * fm - fmm) / (2.0 * h * h * h);
    // Taylor continuation in i*y: f(x + i y) = f(x) + i y f'(x) - (y^2 / 2) f''(x) - i (y^3 / 6) f'''(x)
    let re = f0 - 0.5 * y * y * d2;
    let im = y * d1 - (y * y * y / 6.0) * d3;
    Complex64::new(re.clamp(-1e4, 1e4), im.clamp(-1e4, 1e4))
}

fn eval_meromorphic(
    preset: MeromorphicPreset,
    compiled_custom: Option<&CompiledExpr>,
    pole: Complex64,
    z: Complex64,
) -> Complex64 {
    match preset {
        MeromorphicPreset::SimplePoleRational => {
            let denom = z - pole;
            if denom.norm_sqr() < 1e-20 {
                Complex64::new(0.0, 0.0)
            } else {
                (z + 1.0) / denom
            }
        }
        MeromorphicPreset::ExponentialOverPole => {
            let denom = z - pole;
            if denom.norm_sqr() < 1e-20 {
                Complex64::new(0.0, 0.0)
            } else {
                let safe_z = Complex64::new(z.re.clamp(-20.0, 20.0), z.im.clamp(-20.0, 20.0));
                (0.5 * safe_z).exp() / denom
            }
        }
        MeromorphicPreset::EntireCubic => z * z * z - 0.5 * z + 0.25,
        MeromorphicPreset::CustomReflectedPole => {
            let num = schwarz_reflect(|w| eval_real_analytic_upper(compiled_custom, w), z);
            let denom = z - pole;
            if denom.norm_sqr() < 1e-20 {
                Complex64::new(0.0, 0.0)
            } else {
                num / denom
            }
        }
    }
}

impl RiemannSurfaceStudio {
    pub(crate) fn parameters_are_valid(&self) -> bool {
        self.camera.is_valid()
            && (2..=6).contains(&self.root_n)
            && (1..=3).contains(&self.log_sheet_span)
            && self.radial_max.is_finite()
            && (0.6..=2.5).contains(&self.radial_max)
            && (8..=crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION).contains(&self.radial_steps)
            && (16..=crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION)
                .contains(&self.angular_steps)
            && (-5..=5).contains(&self.start_sheet)
            && (-4..=4).contains(&self.winding_turns)
            && self.loop_radius.is_finite()
            && (0.2..=2.0).contains(&self.loop_radius)
            && self
                .loop_center
                .iter()
                .all(|v| v.is_finite() && v.abs() <= 3.0)
            && self.sc_scale.is_finite()
            && (0.3..=2.5).contains(&self.sc_scale)
            && self.probe_z.iter().all(|v| v.is_finite() && v.abs() <= 3.0)
            && self.pole_z.iter().all(|v| v.is_finite() && v.abs() <= 3.0)
            && (32..=400).contains(&self.contour_steps)
            && (1..=3).contains(&self.derivative_order)
    }

    pub(crate) fn clamp_all_bounds(&mut self) {
        if !self.camera.is_valid() {
            self.camera.reset();
        }
        self.root_n = self.root_n.clamp(2, 6);
        self.log_sheet_span = self.log_sheet_span.clamp(1, 3);
        self.radial_max = if self.radial_max.is_finite() {
            self.radial_max.clamp(0.6, 2.5)
        } else {
            1.8
        };
        let max_res = crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION;
        self.radial_steps = self.radial_steps.clamp(8, max_res);
        self.angular_steps = self.angular_steps.clamp(16, max_res);
        self.start_sheet = self.start_sheet.clamp(-5, 5);
        self.winding_turns = self.winding_turns.clamp(-4, 4);
        self.loop_radius = if self.loop_radius.is_finite() {
            self.loop_radius.clamp(0.2, 2.0)
        } else {
            1.15
        };
        for c in &mut self.loop_center {
            *c = if c.is_finite() {
                c.clamp(-3.0, 3.0)
            } else {
                0.0
            };
        }
        self.sc_scale = if self.sc_scale.is_finite() {
            self.sc_scale.clamp(0.3, 2.5)
        } else {
            1.0
        };
        for c in &mut self.probe_z {
            *c = if c.is_finite() {
                c.clamp(-3.0, 3.0)
            } else {
                0.35
            };
        }
        for c in &mut self.pole_z {
            *c = if c.is_finite() {
                c.clamp(-3.0, 3.0)
            } else {
                0.2
            };
        }
        self.contour_steps = self.contour_steps.clamp(32, 400);
        self.derivative_order = self.derivative_order.clamp(1, 3);
        self.dirty = true;
    }

    pub(crate) fn recompute_conformal_and_cauchy(&mut self) {
        let sc = build_sc_map(self.polygon_preset, self.sc_scale);
        let z_base = Complex64::new(0.0, 1.0);
        let sc_probe = Complex64::new(self.probe_z[0], self.probe_z[1].abs().max(0.18));

        self.diagnostics.conformal_scale = conformal_scale_factor(|z| sc.derivative(z), sc_probe);
        self.diagnostics.area_mag = area_magnification(|z| sc.derivative(z), sc_probe);

        let w_probe = sc.transform(sc_probe, z_base, 40);
        self.sc_probe_image = [w_probe.re, w_probe.im];
        let w_refl = schwarz_reflect(|w| sc.transform(w, z_base, 40), sc_probe.conj());
        self.diagnostics.reflected_value = w_refl;
        self.sc_reflected_probe_image = [w_refl.re, w_refl.im];

        self.rebuild_sc_grid_curves(&sc, z_base);
        self.recompute_cauchy_and_residues();
    }

    fn rebuild_sc_grid_curves(&mut self, sc: &SchwarzChristoffel, z_base: Complex64) {
        self.sc_upper_curves.clear();
        self.sc_reflected_curves.clear();
        let x_samples = 22_usize;
        for &y in &[0.12_f64, 0.35, 0.7, 1.2] {
            let mut upper = Vec::with_capacity(x_samples + 1);
            let mut lower = Vec::with_capacity(x_samples + 1);
            for ix in 0..=x_samples {
                let x = -1.8 + 3.6 * (ix as f64 / x_samples as f64);
                let z_up = Complex64::new(x, y);
                let w_up = sc.transform(z_up, z_base, 32);
                if w_up.re.is_finite() && w_up.im.is_finite() {
                    upper.push([w_up.re, w_up.im]);
                }
                if self.enable_schwarz_reflection {
                    let w_lo = schwarz_reflect(|w| sc.transform(w, z_base, 32), z_up.conj());
                    if w_lo.re.is_finite() && w_lo.im.is_finite() {
                        lower.push([w_lo.re, w_lo.im]);
                    }
                }
            }
            self.sc_upper_curves.push(upper);
            if self.enable_schwarz_reflection {
                self.sc_reflected_curves.push(lower);
            }
        }
    }

    fn recompute_cauchy_and_residues(&mut self) {
        let compiled_custom =
            compile_custom_real_numerator(self.custom_numerator_bar.raw_text.as_str());
        let preset = self.meromorphic_preset;
        let pole = Complex64::new(self.pole_z[0], self.pole_z[1]);
        let probe = Complex64::new(self.probe_z[0], self.probe_z[1]);
        let center = Complex64::new(self.loop_center[0], self.loop_center[1]);
        let radius = self.loop_radius.max(0.2);
        let turns = self.winding_turns;
        let steps = self.contour_steps.clamp(32, 400);

        let f = |z: Complex64| eval_meromorphic(preset, compiled_custom.as_ref(), pole, z);
        let gamma = |t: f64| center + Complex64::from_polar(radius, turns as f64 * t);
        let dgamma = |t: f64| {
            let w = turns as f64;
            Complex64::new(0.0, w) * Complex64::from_polar(radius, w * t)
        };

        self.diagnostics.contour_integral = path_integrate(f, gamma, dgamma, 0.0, TAU, steps);
        self.diagnostics.cauchy_value = cauchy_integral_value(f, probe, center, radius, steps);
        self.diagnostics.exact_probe_value = f(probe);
        self.diagnostics.cauchy_nth_deriv = cauchy_derivative(
            f,
            probe,
            self.derivative_order.clamp(1, 3),
            center,
            radius,
            steps,
        );
        self.diagnostics.residue_integral = residue_via_integral(f, pole, 0.08, steps);
        self.diagnostics.residue_limit = residue_simple_limit(f, pole, 1e-6);
    }

    pub(crate) fn recompute_if_dirty(&mut self) {
        if !self.dirty {
            return;
        }
        self.rebuild_surface_and_monodromy();
        self.recompute_conformal_and_cauchy();
        self.dirty = false;
    }
}
