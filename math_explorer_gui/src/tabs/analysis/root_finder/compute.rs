#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    RootDimension, RootFinderTool, ScalarRootMethod, ScalarStepRecord, System2DStepRecord,
    DEFAULT_ROOT_TOL, MAX_ROOT_TOL, MIN_ROOT_TOL,
};
use math_explorer::pure_math::analysis::roots::{
    Bisection, Brent, MultivariateNewton, NewtonRaphson, RootFinder, Secant,
};

const CONTOUR_GRID_N: usize = 60;

impl RootFinderTool {
    pub(crate) fn eval_scalar_raw(&self, x: f64) -> f64 {
        self.scalar_input
            .active_compiled()
            .map_or(f64::NAN, |c| c.eval_1d(x, self.scalar_input.param_slice()))
    }

    pub(crate) fn eval_2d_pair(&self, x: f64, y: f64) -> [f64; 2] {
        let compiled = self.coupled_bank.active_compiled();
        let params = self.coupled_bank.param_slice();
        let f1 = compiled
            .first()
            .map_or(f64::NAN, |c| c.eval_2d(x, y, params));
        let f2 = compiled
            .get(1)
            .map_or(f64::NAN, |c| c.eval_2d(x, y, params));
        [f1, f2]
    }

    pub(crate) fn recalculate(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
        self.converged_root_1d = None;
        self.converged_root_2d = None;
        self.final_residual = 0.0;
        self.status_error = None;
        self.scalar_steps.clear();
        self.system_steps.clear();
        self.cached_1d_curve.clear();
        self.contours_f1.clear();
        self.contours_f2.clear();

        match self.dimension {
            RootDimension::Scalar1D => self.recalculate_1d(),
            RootDimension::System2D => self.recalculate_2d(),
        }
        self.selected_step = self.selected_step.min(self.max_step_index());
    }

    fn effective_tol(&self) -> f64 {
        if self.tolerance.is_finite() {
            self.tolerance.clamp(MIN_ROOT_TOL, MAX_ROOT_TOL)
        } else {
            DEFAULT_ROOT_TOL
        }
    }

    fn recalculate_1d(&mut self) {
        if !self.bracket_min.is_finite()
            || !self.bracket_max.is_finite()
            || self.bracket_min >= self.bracket_max
        {
            self.status_error = Some("Interval requires finite a < b.".into());
            return;
        }
        let span = self.bracket_max - self.bracket_min;
        let plot_min = self.bracket_min - 0.15 * span;
        let plot_max = self.bracket_max + 0.15 * span;
        self.cached_1d_curve = (0..=400)
            .filter_map(|i| {
                let x = plot_min + (plot_max - plot_min) * (i as f64 / 400.0);
                let y = self.eval_scalar_raw(x);
                (x.is_finite() && y.is_finite()).then_some([x, y.clamp(-1e6, 1e6)])
            })
            .collect();

        let tol = self.effective_tol();
        let max_it = self.max_iterations.clamp(4, 200);
        let f = |x: f64| self.eval_scalar_raw(x);
        let res = match self.scalar_method {
            ScalarRootMethod::Bisection => {
                Bisection::new(max_it, tol).find_root(f, self.bracket_min, self.bracket_max)
            }
            ScalarRootMethod::Brent => {
                Brent::new(max_it, tol).find_root(f, self.bracket_min, self.bracket_max)
            }
            ScalarRootMethod::NewtonRaphson => {
                NewtonRaphson::new(max_it, tol).find_root(f, self.bracket_min, self.bracket_max)
            }
            ScalarRootMethod::Secant => {
                Secant::new(max_it, tol).find_root(f, self.bracket_min, self.bracket_max)
            }
        };

        self.trace_scalar_steps(tol, max_it);
        match res {
            Ok(root) => {
                self.converged_root_1d = Some(root);
                let fv = self.eval_scalar_raw(root).abs();
                self.final_residual = if fv.is_finite() { fv } else { 0.0 };
            }
            Err(err) => {
                self.status_error = Some(err.to_string());
                self.final_residual = self.scalar_steps.last().map_or(0.0, |s| {
                    if s.fx.is_finite() {
                        s.fx.abs()
                    } else {
                        0.0
                    }
                });
            }
        }
    }

    fn trace_scalar_steps(&mut self, tol: f64, max_it: usize) {
        match self.scalar_method {
            ScalarRootMethod::Bisection => self.trace_bisection_steps(tol, max_it),
            ScalarRootMethod::Brent => self.trace_brent_steps(tol, max_it),
            ScalarRootMethod::NewtonRaphson => self.trace_newton_1d_steps(tol, max_it),
            ScalarRootMethod::Secant => self.trace_secant_steps(tol, max_it),
        }
    }

    fn trace_bisection_steps(&mut self, tol: f64, max_it: usize) {
        let (mut a, mut b) = (self.bracket_min, self.bracket_max);
        let (mut fa, fb) = (self.eval_scalar_raw(a), self.eval_scalar_raw(b));
        if !fa.is_finite() || !fb.is_finite() {
            return;
        }
        for step in 0..max_it {
            let mid = a + 0.5 * (b - a);
            let fm = self.eval_scalar_raw(mid);
            if !fm.is_finite() {
                break;
            }
            self.scalar_steps.push(ScalarStepRecord {
                step_index: step,
                x: mid,
                fx: fm,
                bracket: Some([a.min(b), a.max(b)]),
                construction_segment: Some([[a, fa], [b, self.eval_scalar_raw(b)]]),
            });
            if fm == 0.0 || fm.abs() <= tol || (b - a).abs() <= tol {
                break;
            }
            if (fa < 0.0 && fm > 0.0) || (fa > 0.0 && fm < 0.0) {
                b = mid;
            } else if (fb < 0.0 && fm > 0.0) || (fb > 0.0 && fm < 0.0) {
                a = mid;
                fa = fm;
            } else {
                break;
            }
        }
    }

    fn trace_brent_steps(&mut self, tol: f64, max_it: usize) {
        let (mut a, mut b) = (self.bracket_min, self.bracket_max);
        let (mut fa, mut fb) = (self.eval_scalar_raw(a), self.eval_scalar_raw(b));
        if !fa.is_finite() || !fb.is_finite() || !((fa < 0.0 && fb > 0.0) || (fa > 0.0 && fb < 0.0))
        {
            return;
        }
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
        let (mut c, mut fc, mut d, mut mflag) = (a, fa, b - a, true);
        for step in 0..max_it {
            let tol1 = 2.0 * f64::EPSILON * b.abs() + 0.5 * tol;
            let xm = 0.5 * (a - b);
            let mut s = brent_candidate_1d(a, b, c, fa, fb, fc);
            let prev = if mflag { (b - c).abs() } else { (c - d).abs() };
            let q = 0.25 * (3.0 * a + b);
            if !s.is_finite()
                || s <= q.min(b)
                || s >= q.max(b)
                || (s - b).abs() >= 0.5 * prev
                || prev < tol1
            {
                s = b + xm;
                mflag = true;
            } else {
                mflag = false;
            }
            let fs = self.eval_scalar_raw(s);
            if !fs.is_finite() {
                break;
            }
            self.scalar_steps.push(ScalarStepRecord {
                step_index: step,
                x: s,
                fx: fs,
                bracket: Some([a.min(b), a.max(b)]),
                construction_segment: Some([[a, fa], [b, fb]]),
            });
            if fs == 0.0 || fs.abs() <= tol || xm.abs() <= tol1 {
                break;
            }
            (d, c, fc) = (c, b, fb);
            if (fa < 0.0 && fs > 0.0) || (fa > 0.0 && fs < 0.0) {
                (b, fb) = (s, fs);
            } else {
                (a, fa) = (s, fs);
            }
            if fa.abs() < fb.abs() {
                std::mem::swap(&mut a, &mut b);
                std::mem::swap(&mut fa, &mut fb);
            }
        }
    }

    fn trace_newton_1d_steps(&mut self, tol: f64, max_it: usize) {
        let mut x = 0.5 * (self.bracket_min + self.bracket_max);
        let h = 1e-7;
        for step in 0..max_it {
            let fx = self.eval_scalar_raw(x);
            if !fx.is_finite() {
                break;
            }
            let dfx = (self.eval_scalar_raw(x + h) - self.eval_scalar_raw(x - h)) / (2.0 * h);
            let x_next = if dfx.is_finite() && dfx.abs() > 1e-14 {
                x - fx / dfx
            } else {
                x
            };
            self.scalar_steps.push(ScalarStepRecord {
                step_index: step,
                x,
                fx,
                bracket: None,
                construction_segment: x_next.is_finite().then_some([[x, fx], [x_next, 0.0]]),
            });
            if fx.abs() <= tol
                || (x_next - x).abs() <= tol
                || !dfx.is_finite()
                || dfx.abs() <= 1e-14
            {
                break;
            }
            x = x_next;
        }
    }

    fn trace_secant_steps(&mut self, tol: f64, max_it: usize) {
        let (mut x0, mut x1) = (self.bracket_min, self.bracket_max);
        let (mut f0, mut f1) = (self.eval_scalar_raw(x0), self.eval_scalar_raw(x1));
        if !f0.is_finite() || !f1.is_finite() {
            return;
        }
        for step in 0..max_it {
            let denom = f1 - f0;
            let x2 = if denom.abs() > 1e-15 {
                x1 - f1 * (x1 - x0) / denom
            } else {
                x1
            };
            self.scalar_steps.push(ScalarStepRecord {
                step_index: step,
                x: x1,
                fx: f1,
                bracket: None,
                construction_segment: x2.is_finite().then_some([[x0, f0], [x2, 0.0]]),
            });
            if f1.abs() <= tol || (x2 - x1).abs() <= tol || denom.abs() <= 1e-15 {
                break;
            }
            let f2 = self.eval_scalar_raw(x2);
            if !f2.is_finite() {
                break;
            }
            x0 = x1;
            f0 = f1;
            x1 = x2;
            f1 = f2;
        }
    }

    fn recalculate_2d(&mut self) {
        let tol = self.effective_tol();
        let max_it = self.max_iterations.clamp(4, 200);
        let solver = MultivariateNewton::new(max_it, tol);
        let res = solver.solve_fixed(|pt| self.eval_2d_pair(pt[0], pt[1]), self.init_2d);
        self.trace_newton_2d_steps(tol, max_it);
        match res {
            Ok(root) => {
                self.converged_root_2d = Some(root);
                let fv = self.eval_2d_pair(root[0], root[1]);
                let inf = fv[0].abs().max(fv[1].abs());
                self.final_residual = if inf.is_finite() { inf } else { 0.0 };
            }
            Err(err) => {
                self.status_error = Some(err.to_string());
                self.final_residual = self.system_steps.last().map_or(0.0, |s| {
                    if s.inf_norm.is_finite() {
                        s.inf_norm
                    } else {
                        0.0
                    }
                });
            }
        }
        let center = self.converged_root_2d.unwrap_or(self.init_2d);
        let half = self.domain_span_2d.clamp(0.5, 25.0);
        let bounds = [
            center[0] - half,
            center[0] + half,
            center[1] - half,
            center[1] + half,
        ];
        self.contours_f1 = extract_zero_contours(|x, y| self.eval_2d_pair(x, y)[0], bounds);
        self.contours_f2 = extract_zero_contours(|x, y| self.eval_2d_pair(x, y)[1], bounds);
    }

    fn trace_newton_2d_steps(&mut self, tol: f64, max_it: usize) {
        let mut cur = self.init_2d;
        for step in 0..=max_it {
            let r = self.eval_2d_pair(cur[0], cur[1]);
            if !r[0].is_finite() || !r[1].is_finite() {
                break;
            }
            let inf_norm = r[0].abs().max(r[1].abs());
            self.system_steps.push(System2DStepRecord {
                step_index: step,
                point: cur,
                residual: r,
                inf_norm,
            });
            if inf_norm <= tol || step == max_it {
                break;
            }
            let Some(next) = damped_2d_step(|x, y| self.eval_2d_pair(x, y), cur, r, inf_norm)
            else {
                break;
            };
            let step_inf = (next[0] - cur[0]).abs().max((next[1] - cur[1]).abs());
            cur = next;
            if step_inf <= tol {
                let rf = self.eval_2d_pair(cur[0], cur[1]);
                if rf[0].is_finite() && rf[1].is_finite() {
                    self.system_steps.push(System2DStepRecord {
                        step_index: step + 1,
                        point: cur,
                        residual: rf,
                        inf_norm: rf[0].abs().max(rf[1].abs()),
                    });
                }
                break;
            }
        }
    }
}

fn brent_candidate_1d(a: f64, b: f64, c: f64, fa: f64, fb: f64, fc: f64) -> f64 {
    if (fa - fc).abs() > f64::EPSILON && (fb - fc).abs() > f64::EPSILON {
        a * fb * fc / ((fa - fb) * (fa - fc))
            + b * fa * fc / ((fb - fa) * (fb - fc))
            + c * fa * fb / ((fc - fa) * (fc - fb))
    } else {
        b - fb * (b - a) / (fb - fa)
    }
}

fn damped_2d_step<F>(eval: F, cur: [f64; 2], r: [f64; 2], cur_norm: f64) -> Option<[f64; 2]>
where
    F: Fn(f64, f64) -> [f64; 2],
{
    let hx = 1e-7 * cur[0].abs().max(1.0);
    let hy = 1e-7 * cur[1].abs().max(1.0);
    let fx_p = eval(cur[0] + hx, cur[1]);
    let fx_m = eval(cur[0] - hx, cur[1]);
    let fy_p = eval(cur[0], cur[1] + hy);
    let fy_m = eval(cur[0], cur[1] - hy);
    let j00 = (fx_p[0] - fx_m[0]) / (2.0 * hx);
    let j10 = (fx_p[1] - fx_m[1]) / (2.0 * hx);
    let j01 = (fy_p[0] - fy_m[0]) / (2.0 * hy);
    let j11 = (fy_p[1] - fy_m[1]) / (2.0 * hy);
    let det = j00 * j11 - j01 * j10;
    if !det.is_finite() || det.abs() <= 1e-14 {
        return None;
    }
    let dx = (-r[0] * j11 + r[1] * j01) / det;
    let dy = (-j00 * r[1] + j10 * r[0]) / det;
    let mut alpha = 1.0;
    for attempt in 0..8 {
        let cand = [cur[0] + alpha * dx, cur[1] + alpha * dy];
        let rc = eval(cand[0], cand[1]);
        if rc[0].is_finite() && rc[1].is_finite() {
            let n = rc[0].abs().max(rc[1].abs());
            if n < cur_norm || attempt == 7 {
                return Some(cand);
            }
        }
        alpha *= 0.5;
    }
    None
}

fn extract_zero_contours<F>(f: F, bounds: [f64; 4]) -> Vec<Vec<[f64; 2]>>
where
    F: Fn(f64, f64) -> f64,
{
    use math_commons::math_kernel::types::flatten_2d_index;
    let [x_min, x_max, y_min, y_max] = bounds;
    let dx = (x_max - x_min) / (CONTOUR_GRID_N as f64);
    let dy = (y_max - y_min) / (CONTOUR_GRID_N as f64);
    let stride = CONTOUR_GRID_N + 1;
    let mut vals = vec![0.0; stride * stride];
    for iy in 0..=CONTOUR_GRID_N {
        let y = y_min + (iy as f64) * dy;
        for ix in 0..=CONTOUR_GRID_N {
            let x = x_min + (ix as f64) * dx;
            vals[flatten_2d_index(ix, iy, stride)] = f(x, y);
        }
    }

    let mut segments = Vec::new();
    for iy in 0..CONTOUR_GRID_N {
        let (y0, y1) = (y_min + (iy as f64) * dy, y_min + ((iy + 1) as f64) * dy);
        for ix in 0..CONTOUR_GRID_N {
            let (x0, x1) = (x_min + (ix as f64) * dx, x_min + ((ix + 1) as f64) * dx);
            let c = [
                vals[flatten_2d_index(ix, iy, stride)],
                vals[flatten_2d_index(ix + 1, iy, stride)],
                vals[flatten_2d_index(ix + 1, iy + 1, stride)],
                vals[flatten_2d_index(ix, iy + 1, stride)],
            ];
            collect_cell_zero_segments([x0, x1, y0, y1], c, &mut segments);
        }
    }
    segments
}

fn collect_cell_zero_segments(rect: [f64; 4], c: [f64; 4], segments: &mut Vec<Vec<[f64; 2]>>) {
    if !c.iter().all(|v| v.is_finite()) {
        return;
    }
    let [x0, x1, y0, y1] = rect;
    let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    let mut hits = [[0.0; 2]; 4];
    let mut count = 0;
    for edge in 0..4 {
        let next = (edge + 1) % 4;
        let (va, vb) = (c[edge], c[next]);
        if (va <= 0.0 && vb > 0.0) || (va >= 0.0 && vb < 0.0) {
            let denom = va - vb;
            let t = if denom.abs() > 1e-15 {
                (va / denom).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let pa = corners[edge];
            let pb = corners[next];
            hits[count] = [pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])];
            count += 1;
        }
    }
    if count >= 2 {
        segments.push(vec![hits[0], hits[1]]);
    }
    if count == 4 {
        segments.push(vec![hits[2], hits[3]]);
    }
}
