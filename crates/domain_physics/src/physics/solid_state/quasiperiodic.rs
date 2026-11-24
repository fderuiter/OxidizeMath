//! 1D quasiperiodic tight-binding Hamiltonians: Fibonacci chains and Aubry-André localization.
//!
//! Solves real symmetric tridiagonal Hamiltonians using a preallocated implicit Wilkinson-shifted
//! QL eigensolver with zero dynamic heap allocations, computing Cantor-like eigenspectra,
//! devil's-staircase Integrated Density of States (IDOS), and eigenstate Inverse Participation Ratios (IPR).

// [cite:solid_state]

use super::quasicrystal::GOLDEN_RATIO;
use math_commons::math_kernel::types::flatten_2d_index;
use std::f64::consts::{PI, TAU};

/// Maximum finite chain length $N$ supported by [`QuasiperiodicWorkspace`] within registry bounds.
pub const MAX_CHAIN_SITES: usize = 256;

/// Inverse golden ratio $\beta = \tau^{-1} = (\sqrt{5} - 1) / 2$ governing incommensurate modulation.
pub const INVERSE_GOLDEN_RATIO: f64 = GOLDEN_RATIO - 1.0;

/// Tight-binding Hamiltonian specification for finite 1D quasiperiodic chains.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuasiperiodicChainModel {
    /// Off-diagonal Fibonacci hopping chain $H = \sum_i E_0 |i\rangle\langle i| - \sum_i t_i (|i\rangle\langle i+1| + \text{h.c.})$
    /// with $t_i = t(1 + \delta)$ on $A$ bonds and $t_i = t(1 - \delta)$ on $B$ bonds.
    Fibonacci {
        /// Number of atomic sites $N \in [2, 256]$.
        sites: usize,
        /// Mean nearest-neighbor hopping amplitude $t$.
        t_avg: f64,
        /// Dimensionless hopping modulation $\delta \in [0.0, 0.95]$.
        modulation: f64,
        /// Uniform on-site reference energy $E_0$.
        e0: f64,
    },
    /// Aubry-André (Harper) chain $H = \sum_i [E_0 + \lambda \cos(2\pi \tau^{-1} i + \theta)] |i\rangle\langle i| - t \sum_i (|i\rangle\langle i+1| + \text{h.c.})$
    /// exhibiting a self-dual delocalization-localization transition at $\lambda = 2|t|$.
    AubryAndre {
        /// Number of atomic sites $N \in [2, 256]$.
        sites: usize,
        /// Nearest-neighbor hopping amplitude $t$.
        t: f64,
        /// Incommensurate on-site potential amplitude $\lambda \ge 0$.
        lambda: f64,
        /// Phason offset $\theta$ of the cosine potential.
        phase: f64,
        /// Uniform on-site energy offset $E_0$.
        e0: f64,
    },
}

/// Evaluates the $n$-th binary symbol ($true \equiv A$, $false \equiv B$) of the canonical Fibonacci Sturmian word.
#[must_use]
#[verified_engine::verified]
pub fn fibonacci_word_is_a(index: usize) -> bool {
    let a = (((index + 2) as f64) * INVERSE_GOLDEN_RATIO).floor() as i64;
    let b = (((index + 1) as f64) * INVERSE_GOLDEN_RATIO).floor() as i64;
    (a - b) == 1
}

/// Preallocated zero-allocation eigensolver workspace for 1D quasiperiodic chains.
#[derive(Debug, Clone)]
pub struct QuasiperiodicWorkspace {
    max_sites: usize,
    sites: usize,
    diag: Vec<f64>,
    off_diag: Vec<f64>,
    eigvecs: Vec<f64>,
    idos_buf: Vec<f64>,
    ipr_buf: Vec<f64>,
    mean_ipr: f64,
}

impl Default for QuasiperiodicWorkspace {
    fn default() -> Self {
        Self::new(MAX_CHAIN_SITES)
    }
}

impl QuasiperiodicWorkspace {
    /// Allocates tridiagonal and eigenvector buffers for chains up to `max_sites` ($\le 256$).
    #[must_use]
    pub fn new(max_sites: usize) -> Self {
        let cap = max_sites.clamp(2, MAX_CHAIN_SITES);
        Self {
            max_sites: cap,
            sites: 0,
            diag: vec![0.0; cap],
            off_diag: vec![0.0; cap],
            eigvecs: vec![0.0; cap * cap],
            idos_buf: vec![0.0; cap],
            ipr_buf: vec![0.0; cap],
            mean_ipr: 0.0,
        }
    }

    /// Sorted ascending eigenvalues $E_0 \le E_1 \le \dots \le E_{N-1}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn eigenvalues(&self) -> &[f64] {
        &self.diag[..self.sites]
    }

    /// Integrated Density of States $\text{IDOS}(E_n) = (n + 1) / N \in (0, 1]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn idos(&self) -> &[f64] {
        &self.idos_buf[..self.sites]
    }

    /// Per-eigenstate Inverse Participation Ratio $\text{IPR}_n = \sum_{i=0}^{N-1} |\psi_n(i)|^4 \in [1/N, 1]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn ipr(&self) -> &[f64] {
        &self.ipr_buf[..self.sites]
    }

    /// Arithmetic mean Inverse Participation Ratio across all $N$ eigenstates.
    #[must_use]
    #[verified_engine::verified]
    pub fn mean_ipr(&self) -> f64 {
        self.mean_ipr
    }

    /// Counts open spectral gaps $E_{n+1} - E_n > \text{min\_gap}$ across the sorted spectrum.
    #[must_use]
    #[verified_engine::verified]
    pub fn count_spectral_gaps(&self, min_gap: f64) -> usize {
        if self.sites < 2 {
            return 0;
        }
        let threshold = min_gap.max(1e-9);
        let evals = &self.diag[..self.sites];
        evals
            .windows(2)
            .filter(|w| (w[1] - w[0]) > threshold)
            .count()
    }

    /// Counts fractal superband gaps exceeding `prominence_factor` ($\ge 1.25$) times the
    /// open-boundary 1D Bloch level spacing at the same integrated state index.
    ///
    /// A uniform periodic 1D tight-binding chain satisfies $\Delta E_i = \Delta E_i^{\text{Bloch}}$
    /// identically (zero prominent gaps), whereas Fibonacci chains open an increasing self-similar
    /// Cantor hierarchy of prominent gaps across successive approximants $F_k$.
    #[must_use]
    #[verified_engine::verified]
    pub fn count_prominent_gaps(&self, prominence_factor: f64) -> usize {
        if self.sites < 3 {
            return 0;
        }
        let n = self.sites;
        let evals = &self.diag[..n];
        let span = (evals[n - 1] - evals[0]).max(1e-12);
        let n_plus_1 = (n + 1) as f64;
        let denom = (PI / n_plus_1).cos().max(1e-6);
        let base = (span / denom) * (PI / (2.0 * n_plus_1)).sin();
        let factor = prominence_factor.max(1.25);
        evals
            .windows(2)
            .enumerate()
            .filter(|&(i, w)| {
                let bloch_spacing = base * (PI * ((2 * i + 3) as f64) / (2.0 * n_plus_1)).sin();
                (w[1] - w[0]) > factor * bloch_spacing
            })
            .count()
    }

    /// Assembles and diagonalizes `model` in place with zero dynamic heap allocations.
    ///
    /// # Errors
    ///
    /// Returns `Err` if QR/QL tridiagonal iteration fails to converge within bounded sweeps.
    #[verified_engine::verified]
    pub fn solve_into(&mut self, model: QuasiperiodicChainModel) -> Result<(), &'static str> {
        let n = self.populate_tridiagonal(model);
        solve_symmetric_tridiagonal_ql(
            &mut self.diag[..n],
            &mut self.off_diag[..n],
            &mut self.eigvecs[..n * n],
            n,
        )?;
        sort_eigenpairs_and_compute_metrics(self, n);
        Ok(())
    }

    fn populate_tridiagonal(&mut self, model: QuasiperiodicChainModel) -> usize {
        match model {
            QuasiperiodicChainModel::Fibonacci {
                sites,
                t_avg,
                modulation,
                e0,
            } => {
                let n = sites.clamp(2, self.max_sites);
                self.sites = n;
                let delta = modulation.clamp(-0.95, 0.95);
                let t_a = t_avg * (1.0 + delta);
                let t_b = t_avg * (1.0 - delta);
                self.diag[..n].fill(e0);
                for (i, slot) in self.off_diag.iter_mut().enumerate().take(n - 1) {
                    *slot = if fibonacci_word_is_a(i) { -t_a } else { -t_b };
                }
                self.off_diag[n - 1] = 0.0;
                n
            }
            QuasiperiodicChainModel::AubryAndre {
                sites,
                t,
                lambda,
                phase,
                e0,
            } => {
                let n = sites.clamp(2, self.max_sites);
                self.sites = n;
                for (i, slot) in self.diag.iter_mut().enumerate().take(n) {
                    let angle = TAU * INVERSE_GOLDEN_RATIO * (i as f64) + phase;
                    *slot = e0 + lambda * angle.cos();
                }
                self.off_diag[..(n - 1)].fill(-t);
                self.off_diag[n - 1] = 0.0;
                n
            }
        }
    }
}

fn solve_symmetric_tridiagonal_ql(
    d: &mut [f64],
    e: &mut [f64],
    z: &mut [f64],
    n: usize,
) -> Result<(), &'static str> {
    z.fill(0.0);
    for i in 0..n {
        z[flatten_2d_index(i, i, n)] = 1.0;
    }
    e[n - 1] = 0.0;

    for l in 0..n {
        let mut iter = 0_usize;
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= 1e-14 * dd.max(1e-30) {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > 64 {
                return Err("Symmetric tridiagonal QL solver exceeded 64 iterations");
            }
            ql_implicit_step(d, e, z, n, l, m);
        }
    }
    Ok(())
}

fn ql_implicit_step(d: &mut [f64], e: &mut [f64], z: &mut [f64], n: usize, l: usize, m: usize) {
    let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
    let mut r = g.hypot(1.0);
    let denom = g + if g >= 0.0 { r.abs() } else { -r.abs() };
    g = d[m] - d[l] + e[l] / denom;

    let (mut s, mut c, mut p) = (1.0_f64, 1.0_f64, 0.0_f64);
    for i in (l..m).rev() {
        let f = s * e[i];
        let b = c * e[i];
        r = f.hypot(g);
        e[i + 1] = r;
        if r.abs() < 1e-30 {
            d[i + 1] -= p;
            e[m] = 0.0;
            return;
        }
        s = f / r;
        c = g / r;
        g = d[i + 1] - p;
        r = (d[i] - g) * s + 2.0 * c * b;
        p = s * r;
        d[i + 1] = g + p;
        g = c * r - b;
        for row in 0..n {
            let idx0 = flatten_2d_index(i, row, n);
            let idx1 = flatten_2d_index(i + 1, row, n);
            let z0 = z[idx0];
            let z1 = z[idx1];
            z[idx1] = s * z0 + c * z1;
            z[idx0] = c * z0 - s * z1;
        }
    }
    d[l] -= p;
    e[l] = g;
    e[m] = 0.0;
}

fn sort_eigenpairs_and_compute_metrics(ws: &mut QuasiperiodicWorkspace, n: usize) {
    for i in 0..n {
        let mut min_idx = i;
        let mut min_val = ws.diag[i];
        for (j, &val) in ws.diag.iter().enumerate().take(n).skip(i + 1) {
            if val < min_val {
                min_val = val;
                min_idx = j;
            }
        }
        if min_idx != i {
            ws.diag.swap(i, min_idx);
            for row in 0..n {
                let a = flatten_2d_index(i, row, n);
                let b = flatten_2d_index(min_idx, row, n);
                ws.eigvecs.swap(a, b);
            }
        }
    }

    let inv_n = 1.0 / (n as f64);
    let mut sum_ipr = 0.0_f64;
    for col in 0..n {
        ws.idos_buf[col] = ((col + 1) as f64) * inv_n;
        let mut ipr = 0.0_f64;
        for row in 0..n {
            let amp = ws.eigvecs[flatten_2d_index(col, row, n)];
            let prob = amp * amp;
            ipr += prob * prob;
        }
        ws.ipr_buf[col] = ipr;
        sum_ipr += ipr;
    }
    ws.mean_ipr = sum_ipr * inv_n;
}
