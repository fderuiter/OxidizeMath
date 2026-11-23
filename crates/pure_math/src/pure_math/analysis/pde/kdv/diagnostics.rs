//! Zero-allocation invariant integration and sub-cell parabolic soliton peak tracking
//! for [`crate::pure_math::analysis::pde::kdv::KdvSolver1D`].

// [cite:advanced_linear_algebra]

use num_complex::Complex;

use super::KdvSolver1D;
use super::config::{
    DetectedPulsePeak, KdvEquationMode, KdvParams, SolitonInvariants, wrap_periodic_coord,
};

impl KdvSolver1D {
    /// Computes the conserved hierarchy $(M, P, H)$ for the active wave mode with zero heap allocations.
    #[verified_engine::verified]
    pub fn compute_invariants(&mut self, params: &KdvParams) -> SolitonInvariants {
        match params.mode {
            KdvEquationMode::KdvSoliton => self.compute_kdv_invariants(params.nonlinear_enabled),
            KdvEquationMode::CubicNls1D => {
                let sigma = if params.nonlinear_enabled {
                    params.nls_coupling
                } else {
                    0.0
                };
                self.compute_nls_invariants(sigma)
            }
        }
    }

    /// Computes conserved KdV mass $M = \int u\,dx$, momentum $P = \frac{1}{2}\int u^2\,dx$,
    /// and Hamiltonian $H = \int (\frac{1}{2} u_x^2 - \alpha u^3)\,dx$ using spectral differentiation.
    #[verified_engine::verified]
    pub fn compute_kdv_invariants(&mut self, nonlinear_enabled: bool) -> SolitonInvariants {
        let n = self.num_points;
        let inv_n = 1.0 / (n as f64);
        for i in 0..n {
            self.work_spec[i] = Complex::new(self.u[i], 0.0);
        }
        self.fft_fwd
            .process_with_scratch(&mut self.work_spec, &mut self.fft_scratch);
        for i in 0..n {
            let ik = Complex::new(0.0, self.k_deriv[i]);
            self.work_spec[i] *= ik;
        }
        self.fft_inv
            .process_with_scratch(&mut self.work_spec, &mut self.fft_scratch);

        let alpha = if nonlinear_enabled { 1.0 } else { 0.0 };
        let mut sum_m = 0.0_f64;
        let mut sum_p = 0.0_f64;
        let mut sum_h = 0.0_f64;
        for i in 0..n {
            let ui = self.u[i];
            let uxi = self.work_spec[i].re * inv_n;
            sum_m += ui;
            sum_p += 0.5 * ui * ui;
            sum_h += 0.5 * uxi * uxi - alpha * ui * ui * ui;
        }
        SolitonInvariants {
            mass: sum_m * self.dx,
            momentum: sum_p * self.dx,
            hamiltonian: sum_h * self.dx,
        }
    }

    /// Computes conserved 1D cubic NLS norm $M = \int |\psi|^2\,dx$, momentum
    /// $P = \operatorname{Im}\int \psi^* \psi_x\,dx$, and Hamiltonian $H = \int (|\psi_x|^2 - \sigma|\psi|^4)\,dx$.
    #[verified_engine::verified]
    pub fn compute_nls_invariants(&mut self, coupling_sigma: f64) -> SolitonInvariants {
        let n = self.num_points;
        let inv_n = 1.0 / (n as f64);
        self.work_spec.copy_from_slice(&self.psi);
        self.fft_fwd
            .process_with_scratch(&mut self.work_spec, &mut self.fft_scratch);

        let mut spectral_p = 0.0_f64;
        let mut spectral_kin = 0.0_f64;
        for i in 0..n {
            let mag_sq = self.work_spec[i].norm_sqr();
            spectral_p += self.k_deriv[i] * mag_sq;
            spectral_kin += self.k_sq[i] * mag_sq;
        }

        let mut sum_norm = 0.0_f64;
        let mut sum_quartic = 0.0_f64;
        for z in &self.psi {
            let rho = z.norm_sqr();
            sum_norm += rho;
            sum_quartic += rho * rho;
        }
        let parseval_scale = self.dx * inv_n;
        SolitonInvariants {
            mass: sum_norm * self.dx,
            momentum: spectral_p * parseval_scale,
            hamiltonian: spectral_kin * parseval_scale - coupling_sigma * sum_quartic * self.dx,
        }
    }

    /// Locates the two tallest well-separated local maxima of the active scalar profile
    /// ($u(x)$ in KdV mode or $|\psi(x)|$ in NLS mode) with 3-point parabolic sub-grid refinement.
    #[must_use]
    #[verified_engine::verified]
    pub fn locate_top_two_peaks(
        &self,
        mode: KdvEquationMode,
    ) -> (DetectedPulsePeak, DetectedPulsePeak) {
        let n = self.num_points;
        let sample = |idx: usize| -> f64 {
            match mode {
                KdvEquationMode::KdvSoliton => self.u[idx],
                KdvEquationMode::CubicNls1D => self.psi[idx].norm(),
            }
        };

        let mut best_idx: Option<usize> = None;
        let mut best_val = f64::NEG_INFINITY;
        for i in 0..n {
            let prev = sample((i + n - 1) % n);
            let curr = sample(i);
            let next = sample((i + 1) % n);
            if curr >= prev && curr >= next && curr > best_val {
                best_val = curr;
                best_idx = Some(i);
            }
        }
        let Some(idx1) = best_idx else {
            return (DetectedPulsePeak::default(), DetectedPulsePeak::default());
        };
        let peak1 = self.refine_parabolic_peak(idx1, &sample);

        let min_sep_cells = (n / 24).max(3);
        let mut second_idx: Option<usize> = None;
        let mut second_val = f64::NEG_INFINITY;
        for i in 0..n {
            let cyc_dist = (i + n - idx1)
                .rem_euclid(n)
                .min((idx1 + n - i).rem_euclid(n));
            if cyc_dist < min_sep_cells {
                continue;
            }
            let prev = sample((i + n - 1) % n);
            let curr = sample(i);
            let next = sample((i + 1) % n);
            if curr >= prev && curr >= next && curr > second_val {
                second_val = curr;
                second_idx = Some(i);
            }
        }
        let peak2 = second_idx
            .map(|idx2| self.refine_parabolic_peak(idx2, &sample))
            .unwrap_or_default();
        (peak1, peak2)
    }

    /// Measures empirical 2-soliton spatial phase shifts $(\Delta_{\text{fast}}, \Delta_{\text{slow}})$
    /// relative to unperturbed ballistic trajectories $x_{i,0} + c_i t$ on periodic $[-L/2, L/2)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn measure_two_soliton_phase_shifts(
        &self,
        fast_c: f64,
        fast_x0: f64,
        slow_c: f64,
        slow_x0: f64,
    ) -> (f64, f64) {
        let (peak_fast, peak_slow) = self.locate_top_two_peaks(KdvEquationMode::KdvSoliton);
        let expected_fast = fast_x0 + fast_c * self.time;
        let expected_slow = slow_x0 + slow_c * self.time;
        let d_fast = wrap_periodic_coord(peak_fast.x - expected_fast, self.length);
        let d_slow = wrap_periodic_coord(peak_slow.x - expected_slow, self.length);
        (d_fast, d_slow)
    }

    fn refine_parabolic_peak<F>(&self, idx: usize, sample: &F) -> DetectedPulsePeak
    where
        F: Fn(usize) -> f64,
    {
        let n = self.num_points;
        let y_m = sample((idx + n - 1) % n);
        let y_0 = sample(idx);
        let y_p = sample((idx + 1) % n);
        let denom = y_m - 2.0 * y_0 + y_p;
        let offset = if denom.abs() > 1.0e-14 {
            (0.5 * (y_m - y_p) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        let raw_x = self.x_coords[idx] + offset * self.dx;
        let refined_amp = y_0 - 0.25 * (y_m - y_p) * offset;
        DetectedPulsePeak {
            x: wrap_periodic_coord(raw_x, self.length),
            amplitude: refined_amp,
        }
    }
}
