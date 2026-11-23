#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// [cite:modular_polynomials_review]
// [cite:hurwitz_kronecker]
//! Finite scalar observables and locked-prime Hasse-Weil challenge evaluation for elliptic tools.

use math_explorer::pure_math::elliptic_curves::{
    FpSingularCubic, PrimeWeierstrassCurve, RealSingularCubic, WeierstrassCurveFp,
    j_invariant_mod_p,
};

/// Canonical `ChallengeMetadata` ID for the locked-prime $p = 23$ Hasse-Weil boundary challenge.
pub const CHALLENGE_HASSE_WEIL_BOUNDARY_P23: &str = "hasse_weil_boundary_p23";

fn sanitize_finite(val: f64) -> f64 {
    if val.is_finite() { val } else { 0.0 }
}

fn compute_fp_observable_tuple(
    safe_p: i64,
    mod_a: i64,
    mod_b: i64,
) -> (bool, u64, i64, f64, f64, f64) {
    let hasse_floor = (2.0 * (safe_p as f64).sqrt()).floor();
    match PrimeWeierstrassCurve::new(mod_a, mod_b, safe_p) {
        Ok(c) => {
            let ord = c.point_count().unwrap_or(1);
            let tr = safe_p + 1 - (ord as i64);
            let gap = (hasse_floor - (tr.abs() as f64)).max(0.0);
            (
                false,
                ord,
                tr,
                gap,
                c.discriminant_mod_p() as f64,
                j_invariant_mod_p(&c) as f64,
            )
        }
        Err(_) => {
            let ord = FpSingularCubic::classify(mod_a, mod_b, safe_p)
                .ok()
                .flatten()
                .map_or(safe_p as usize, |s| s.projective_point_count());
            (
                true,
                ord as u64,
                safe_p + 1 - (ord as i64),
                hasse_floor,
                0.0,
                0.0,
            )
        }
    }
}

/// Builds the 9 finite scalar observables exposed to `math_commons::ast` and `StarterGallery`.
#[must_use]
pub fn build_real_and_fp_observables(
    real_mode: bool,
    real_a: f64,
    real_b: f64,
    p: i64,
    mod_a: i64,
    mod_b: i64,
    point_order: usize,
) -> Vec<(&'static str, f64)> {
    let safe_p = p.clamp(5, 997);
    let (fp_singular, group_order, frob_trace, hasse_gap, fp_disc, fp_j) =
        compute_fp_observable_tuple(safe_p, mod_a, mod_b);

    let real_core = 4.0 * real_a.powi(3) + 27.0 * real_b.powi(2);
    let real_sing = RealSingularCubic::classify(real_a, real_b).is_some();
    let real_disc = if real_sing { 0.0 } else { -16.0 * real_core };
    let real_comps = if !real_sing && real_disc > 0.0 {
        2.0
    } else {
        1.0
    };
    let real_j = if real_sing || real_core.abs() <= 1e-12 {
        0.0
    } else {
        (1728.0 * (4.0 * real_a.powi(3)) / real_core).clamp(-1.0e9, 1.0e9)
    };

    let ord_p = point_order.max(1) as f64;
    let cofactor = ((group_order.max(1) as f64) / ord_p).max(1.0);
    let (disc, j_inv, is_sing) = if real_mode {
        (real_disc, real_j, if real_sing { 1.0 } else { 0.0 })
    } else {
        (fp_disc, fp_j, if fp_singular { 1.0 } else { 0.0 })
    };

    vec![
        ("discriminant", sanitize_finite(disc)),
        ("real_components", sanitize_finite(real_comps)),
        ("j_invariant", sanitize_finite(j_inv)),
        ("group_order", sanitize_finite(group_order as f64)),
        ("frobenius_trace", sanitize_finite(frob_trace as f64)),
        ("hasse_gap", sanitize_finite(hasse_gap)),
        ("point_order", sanitize_finite(ord_p)),
        ("cofactor", sanitize_finite(cofactor)),
        ("is_singular", sanitize_finite(is_sing)),
    ]
}

/// Evaluates the locked-prime ($p = 23$) Hasse-Weil boundary challenge (`hasse_gap == 0`).
#[must_use]
pub fn evaluate_hasse_weil_p23(a: i64, b: i64, p: i64) -> crate::framework::ChallengeEvaluation {
    let target_p = 23_i64;
    let bound = (2.0 * (target_p as f64).sqrt()).floor();
    if p != target_p {
        return crate::framework::ChallengeEvaluation::new(
            0.0,
            false,
            "hasse_gap",
            bound,
            format!("Prime modulus must stay locked at p = 23 (got p = {p})."),
        );
    }
    match WeierstrassCurveFp::new(target_p, a, b) {
        Ok(curve) => {
            let order = curve.cardinality();
            let trace = target_p + 1 - (order as i64);
            let gap = (bound - (trace.abs() as f64)).max(0.0);
            let score = ((bound - gap) / bound * 100.0).clamp(0.0, 100.0);
            let passed = gap == 0.0;
            crate::framework::ChallengeEvaluation::new(
                score,
                passed,
                "hasse_gap",
                gap,
                format!(
                    "F_23: #E = {order}, a_23 = {trace}, |a_23| / ⌊2√23⌋ = {} / 9 (hasse_gap = {gap:.0})",
                    trace.abs()
                ),
            )
        }
        Err(_) => crate::framework::ChallengeEvaluation::new(
            0.0,
            false,
            "hasse_gap",
            bound,
            "Curve is singular over F_23 (Δ ≡ 0 mod 23); choose a smooth Hasse-Weil boundary cell.",
        ),
    }
}

impl crate::tabs::number_theory::elliptic_curve_playground::EllipticCurvePlayground {
    pub(crate) fn populate_singular_curve_hot_loop(&mut self) {
        self.is_singular = true;
        self.orbit_len = 0;
        self.frobenius_trace = 0;
        self.p_order = 1;
        self.scalar_kp = math_explorer::pure_math::elliptic_curves::AffinePointFp::Infinity;
        if let Ok(Some(sing)) = FpSingularCubic::classify(self.a, self.b, self.p) {
            self.points_len = sing
                .enumerate_projective_points(&mut self.points_buf)
                .unwrap_or(0);
            self.group_order = sing.projective_point_count();
            if !sing.contains_affine_fp(self.point_p) {
                self.point_p = math_explorer::pure_math::elliptic_curves::AffinePointFp::Finite {
                    x: sing.singular_x,
                    y: 0,
                };
            }
            if !sing.contains_affine_fp(self.point_q) {
                self.point_q = self.points_buf.get(1).copied().unwrap_or_default();
            }
            self.chord = sing.chord_step_fp(self.point_p, self.point_q);
            self.rebuild_ray_segments();
        } else {
            self.points_len = 0;
            self.group_order = 0;
            self.chord = None;
            self.ray_segments.clear();
        }
    }

    pub(crate) fn nt_observables(&self) -> Vec<(&'static str, f64)> {
        build_real_and_fp_observables(
            false,
            self.a as f64,
            self.b as f64,
            self.p,
            self.a,
            self.b,
            self.p_order,
        )
    }

    pub(crate) fn apply_nt_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                self.recompute_all();
                Ok(())
            }
            Some(CHALLENGE_HASSE_WEIL_BOUNDARY_P23) => {
                self.active_challenge = Some(CHALLENGE_HASSE_WEIL_BOUNDARY_P23);
                self.p = 23;
                let _ = self.atlas.set_axis_keys("mod_a", "mod_b");
                self.recompute_all();
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for EllipticCurvePlayground"
            )),
        }
    }

    pub(crate) fn render_nt_atlas_and_handoff_controls(&mut self, ui: &mut eframe::egui::Ui) {
        if self.atlas.show_toggle_button(ui) && self.atlas.is_visible() {
            let _ = self.atlas.set_axis_keys("mod_a", "mod_b");
        }
        super::handoff::render_handoff_buttons(
            ui,
            super::handoff::EllipticHandoffTarget::NumberTheoryTorus,
            super::handoff::EllipticTorusHandoff {
                target: super::handoff::EllipticHandoffTarget::NumberTheoryTorus,
                p: self.p,
                a: self.a,
                b: self.b,
                point_p: self.point_p,
                point_q: self.point_q,
                scalar_k: self.scalar_k,
            },
        );
    }

    pub(crate) fn render_nt_atlas_split(&mut self, ui: &mut eframe::egui::Ui) {
        if !self.atlas.is_visible() {
            return;
        }
        let (xk, _) = self.atlas.axis_keys();
        let p = self.p;
        let sig = if xk == "mod_a" { p as u64 } else { 0_u64 };
        let (ax, ay) = (self.a as f64, self.b as f64);
        let atlas_h = (ui.available_height() * 0.40_f32).clamp(175.0_f32, 280.0_f32);
        let mut teleport = None;
        ui.allocate_ui(
            eframe::egui::vec2(ui.available_width(), atlas_h),
            |atlas_ui| {
                teleport =
                    self.atlas
                        .show_panel(atlas_ui, "nt_ec_torus", ax, ay, sig, move |keys| {
                            std::sync::Arc::new(super::atlas::WeierstrassRegimeCartographer::new(
                                keys, p, 1.0,
                            ))
                        });
            },
        );
        ui.separator();
        if let Some(ev) = teleport {
            self.a = (ev.x_val.round() as i64).clamp(-997, 997);
            self.b = (ev.y_val.round() as i64).clamp(-997, 997);
            self.ray_progress = 1.0_f32;
            self.recompute_all();
        }
    }

    pub(crate) fn populate_nonsingular_curve(
        &mut self,
        curve: math_explorer::pure_math::elliptic_curves::WeierstrassCurveFp,
    ) {
        self.is_singular = false;
        self.points_len = curve.enumerate_points(&mut self.points_buf).unwrap_or(0);
        if !curve.contains_point(self.point_p) {
            self.point_p = self.points_buf.get(1).copied().unwrap_or_default();
        }
        if !curve.contains_point(self.point_q) {
            let idx = if self.points_len > 2 {
                2
            } else {
                1.min(self.points_len.saturating_sub(1))
            };
            self.point_q = self.points_buf[idx];
        }
        self.group_order = self.points_len;
        self.frobenius_trace = self.p + 1 - (self.group_order as i64);
        self.p_order = curve.point_order(self.point_p).unwrap_or(1);
        self.scalar_k = self.scalar_k.clamp(1, 512);
        self.scalar_kp = curve
            .scalar_mul(self.scalar_k, self.point_p)
            .unwrap_or_default();
        self.orbit_len = curve
            .torsion_orbit(self.point_p, &mut self.orbit_buf)
            .unwrap_or(0);
        self.chord = curve.chord_step(self.point_p, self.point_q).ok();
        self.rebuild_ray_segments();
    }

    pub(crate) fn step_orbit_hot_loop(&mut self) {
        let Ok(curve) = math_explorer::pure_math::elliptic_curves::WeierstrassCurveFp::new(
            self.p, self.a, self.b,
        ) else {
            return;
        };
        self.scalar_k = (self.scalar_k % (self.p_order.max(1) as i64)) + 1;
        self.scalar_kp = curve
            .scalar_mul(self.scalar_k, self.point_p)
            .unwrap_or_default();
        self.ray_progress = 1.0_f32;
        self.rebuild_ray_segments();
    }

    pub(crate) fn snap_nearest_prime(&self, candidate: i64) -> i64 {
        let c = candidate.clamp(5, 997);
        let pos = self.primes_table.partition_point(|&q| q < c);
        let hi = self.primes_table.get(pos).copied();
        let lo = pos
            .checked_sub(1)
            .and_then(|i| self.primes_table.get(i).copied());
        match (lo, hi) {
            (Some(l), Some(h)) if (c - l).abs() <= (h - c).abs() => l,
            (_, Some(h)) => h,
            (Some(l), None) => l,
            (None, None) => 17,
        }
    }

    pub(crate) fn step_prime(&mut self, delta: isize) {
        let pos = self.primes_table.partition_point(|&q| q < self.p);
        let max_idx = self.primes_table.len().saturating_sub(1) as isize;
        let next = ((pos as isize) + delta).clamp(0, max_idx) as usize;
        if let Some(&next_p) = self.primes_table.get(next) {
            self.p = next_p;
            self.ray_progress = 1.0_f32;
            self.recompute_all();
        }
    }
}
