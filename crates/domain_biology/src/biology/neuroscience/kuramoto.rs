//! Zero-allocation Kuramoto coupled phase-oscillator network solver.
//!
//! Models collective phase synchronization on arbitrary weighted graphs:
//! $$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j} A_{ji} \sin(\theta_j - \theta_i)$$
//! with global coherence tracked via the complex Kuramoto order parameter
//! $r(t) e^{i\psi(t)} = \frac{1}{N} \sum_{j=0}^{N-1} e^{i\theta_j}$.

use std::f64::consts::TAU;

/// Coupled Kuramoto phase-oscillator network with pre-allocated step buffers.
///
/// # Examples
/// ```rust
/// # use domain_biology::__doctest_facade as math_explorer;
/// use math_explorer::biology::neuroscience::KuramotoNetwork;
///
/// let mut net = KuramotoNetwork::new(6, 4.0);
/// for _ in 0..200 {
///     net.step(0.05);
/// }
/// let (r, _psi) = net.order_parameter();
/// assert!((0.0..=1.0).contains(&r));
/// ```
#[derive(Debug, Clone)]
pub struct KuramotoNetwork {
    phases: Vec<f64>,
    natural_frequencies: Vec<f64>,
    scratch_state: Vec<f64>,
    scratch_deriv: Vec<f64>,
    scratch_accum: Vec<f64>,
    couplings: Vec<(usize, usize, f64)>,
    coupling: f64,
    base_frequency: f64,
    frequency_spread: f64,
    time: f64,
}

impl Default for KuramotoNetwork {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new(5, 2.5)
    }
}

impl KuramotoNetwork {
    /// Creates an `N`-oscillator Kuramoto network initialized on a bidirectional ring.
    #[verified_engine::verified]
    pub fn new(num_nodes: usize, coupling: f64) -> Self {
        let n = num_nodes.max(1);
        let mut ring_edges = Vec::with_capacity(n);
        if n > 1 {
            for i in 0..n {
                ring_edges.push((i, (i + 1) % n, 1.0));
            }
        }
        let mut net = Self {
            phases: Vec::with_capacity(n),
            natural_frequencies: Vec::with_capacity(n),
            scratch_state: Vec::with_capacity(n),
            scratch_deriv: Vec::with_capacity(n),
            scratch_accum: Vec::with_capacity(n),
            couplings: Vec::with_capacity(n * 2),
            coupling: if coupling.is_finite() {
                coupling.max(0.0)
            } else {
                0.0
            },
            base_frequency: 1.5,
            frequency_spread: 0.6,
            time: 0.0,
        };
        net.reconfigure_topology(n, &ring_edges, false, 0.6);
        net.reset_phases();
        net
    }

    /// Reconfigures node buffers and coupling arcs when graph topology changes.
    ///
    /// Existing oscillator phases for indices `< num_nodes` are preserved so live
    /// edge edits smoothly perturb synchronization without resetting phase state.
    #[verified_engine::verified]
    pub fn reconfigure_topology(
        &mut self,
        num_nodes: usize,
        edges: &[(usize, usize, f64)],
        directed: bool,
        frequency_spread: f64,
    ) {
        let n = num_nodes.max(1);
        let old_len = self.phases.len();
        if n != old_len {
            self.phases.resize(n, 0.0);
            for i in old_len..n {
                self.phases[i] = Self::seeded_phase(i, n);
            }
            self.natural_frequencies.resize(n, self.base_frequency);
            self.scratch_state.resize(n, 0.0);
            self.scratch_deriv.resize(n, 0.0);
            self.scratch_accum.resize(n, 0.0);
        }
        self.set_frequency_spread(frequency_spread);
        self.rebuild_couplings(n, edges, directed);
    }

    #[verified_engine::verified]
    fn rebuild_couplings(
        &mut self,
        num_nodes: usize,
        edges: &[(usize, usize, f64)],
        directed: bool,
    ) {
        self.couplings.clear();
        let needed = if directed {
            edges.len()
        } else {
            edges.len().saturating_mul(2)
        };
        if self.couplings.capacity() < needed {
            self.couplings.reserve(needed - self.couplings.capacity());
        }
        for &(u, v, weight) in edges {
            if u < num_nodes && v < num_nodes && u != v && weight.is_finite() {
                let w = weight.max(0.0);
                self.couplings.push((u, v, w));
                if !directed {
                    self.couplings.push((v, u, w));
                }
            }
        }
    }

    #[verified_engine::verified]
    fn seeded_phase(index: usize, total: usize) -> f64 {
        let denom = total.max(1) as f64;
        let golden_offset = ((index as f64) * 0.618_033_988_749_895).fract();
        (TAU * ((index as f64) / denom * 0.65 + 0.35 * golden_offset)).rem_euclid(TAU)
    }

    /// Resets oscillator phases to deterministic spread-out initial angles and zeroes time.
    #[verified_engine::verified]
    pub fn reset_phases(&mut self) {
        let n = self.phases.len();
        for (i, phase) in self.phases.iter_mut().enumerate() {
            *phase = Self::seeded_phase(i, n);
        }
        self.time = 0.0;
    }

    /// Updates natural frequencies deterministically around the base frequency.
    #[verified_engine::verified]
    pub fn set_frequency_spread(&mut self, spread: f64) {
        let clamped = if spread.is_finite() {
            spread.clamp(0.0, 10.0)
        } else {
            0.0
        };
        self.frequency_spread = clamped;
        let n = self.natural_frequencies.len();
        if n <= 1 {
            if let Some(w0) = self.natural_frequencies.first_mut() {
                *w0 = self.base_frequency;
            }
            return;
        }
        let span = (n - 1) as f64;
        for (i, omega) in self.natural_frequencies.iter_mut().enumerate() {
            let centered = ((i as f64) / span) * 2.0 - 1.0;
            *omega = self.base_frequency + clamped * centered;
        }
    }

    /// Sets the global coupling strength $K \ge 0$.
    #[verified_engine::verified]
    pub fn set_coupling(&mut self, coupling: f64) {
        if coupling.is_finite() {
            self.coupling = coupling.clamp(0.0, 100.0);
        }
    }

    /// Returns the global coupling strength $K$.
    #[verified_engine::verified]
    pub fn coupling(&self) -> f64 {
        self.coupling
    }

    /// Returns the natural frequency spread $\Delta\omega$.
    #[verified_engine::verified]
    pub fn frequency_spread(&self) -> f64 {
        self.frequency_spread
    }

    /// Returns the number of coupled oscillators $N$.
    #[verified_engine::verified]
    pub fn num_nodes(&self) -> usize {
        self.phases.len()
    }

    /// Returns elapsed integration time $t$.
    #[verified_engine::verified]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Returns current oscillator phases $\theta_i \in [0, 2\pi)$.
    #[verified_engine::verified]
    pub fn phases(&self) -> &[f64] {
        &self.phases
    }

    /// Returns oscillator natural frequencies $\omega_i$.
    #[verified_engine::verified]
    pub fn natural_frequencies(&self) -> &[f64] {
        &self.natural_frequencies
    }

    /// Sets the phase $\theta_{\text{idx}}$ of a single oscillator (wrapped to $[0, 2\pi)$).
    #[verified_engine::verified]
    pub fn set_phase(&mut self, idx: usize, phase: f64) {
        if idx < self.phases.len() && phase.is_finite() {
            self.phases[idx] = phase.rem_euclid(TAU);
        }
    }

    #[verified_engine::verified]
    fn eval_derivatives(
        natural_frequencies: &[f64],
        couplings: &[(usize, usize, f64)],
        coupling: f64,
        theta: &[f64],
        out: &mut [f64],
    ) {
        out.copy_from_slice(natural_frequencies);
        let scale = coupling / (theta.len() as f64);
        for &(src, dst, weight) in couplings {
            let delta = theta[src] - theta[dst];
            out[dst] += scale * weight * delta.sin();
        }
    }

    /// Advances all oscillator phases by `dt` via 4th-order Runge-Kutta (RK4) with zero heap allocations.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        let n = self.phases.len();
        if n == 0 || !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let half_dt = 0.5 * dt;
        Self::eval_derivatives(
            &self.natural_frequencies,
            &self.couplings,
            self.coupling,
            &self.phases,
            &mut self.scratch_deriv,
        );
        for i in 0..n {
            self.scratch_accum[i] = self.scratch_deriv[i];
            self.scratch_state[i] = self.phases[i] + half_dt * self.scratch_deriv[i];
        }
        Self::eval_derivatives(
            &self.natural_frequencies,
            &self.couplings,
            self.coupling,
            &self.scratch_state,
            &mut self.scratch_deriv,
        );
        for i in 0..n {
            self.scratch_accum[i] += 2.0 * self.scratch_deriv[i];
            self.scratch_state[i] = self.phases[i] + half_dt * self.scratch_deriv[i];
        }
        Self::eval_derivatives(
            &self.natural_frequencies,
            &self.couplings,
            self.coupling,
            &self.scratch_state,
            &mut self.scratch_deriv,
        );
        for i in 0..n {
            self.scratch_accum[i] += 2.0 * self.scratch_deriv[i];
            self.scratch_state[i] = self.phases[i] + dt * self.scratch_deriv[i];
        }
        Self::eval_derivatives(
            &self.natural_frequencies,
            &self.couplings,
            self.coupling,
            &self.scratch_state,
            &mut self.scratch_deriv,
        );
        let dt_sixth = dt / 6.0;
        for i in 0..n {
            let next = self.phases[i] + dt_sixth * (self.scratch_accum[i] + self.scratch_deriv[i]);
            if next.is_finite() {
                self.phases[i] = next.rem_euclid(TAU);
            }
        }
        self.time += dt;
    }

    /// Computes the Kuramoto synchronization order parameter $(r(t), \psi(t))$ with zero allocations.
    ///
    /// Returns $r \in [0, 1]$ ($0$ = incoherent, $1$ = phase-locked) and mean phase $\psi \in (-\pi, \pi]$.
    #[verified_engine::verified]
    pub fn order_parameter(&self) -> (f64, f64) {
        let n = self.phases.len();
        if n == 0 {
            return (0.0, 0.0);
        }
        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;
        for &theta in &self.phases {
            let (s, c) = theta.sin_cos();
            sum_cos += c;
            sum_sin += s;
        }
        let inv_n = 1.0 / (n as f64);
        let r = (sum_cos.hypot(sum_sin) * inv_n).clamp(0.0, 1.0);
        let psi = sum_sin.atan2(sum_cos);
        (r, psi)
    }
}
