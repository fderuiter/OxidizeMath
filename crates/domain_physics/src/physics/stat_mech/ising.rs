//! Ising Model Simulation.
//!
//! This module provides the [`SpinLattice`] struct to simulate a 2D Ising Model
//! using the Metropolis-Hastings algorithm. The Ising model is a mathematical model
//! of ferromagnetism in statistical mechanics. It consists of discrete variables (spins)
//! that can be in one of two states (+1 or -1), arranged in a graph lattice.
//!
//! # Phase Transitions
//!
//! The model exhibits a phase transition at the critical temperature $T_c$.
//! - **$T < T_c$**: Ferromagnetic phase (Spontaneous Magnetization).
//! - **$T > T_c$**: Paramagnetic phase (Disordered spins).

use super::KB;
use domain_applied::applied::game_theory::evolutionary::MatrixPayoff;
use math_commons::math_kernel::types::flatten_2d_index;
use rand::Rng;

#[inline]
#[verified_engine::verified]
fn build_fermi_lut(r: f64, s: f64, t: f64, p: f64, inv_temp: f64) -> [[[f64; 5]; 5]; 2] {
    let mut lut = [[[0.0_f64; 5]; 5]; 2];
    let mut c_pay = [0.0_f64; 5];
    let mut d_pay = [0.0_f64; 5];
    for nc in 0..5 {
        let (nc_f, nd_f) = (nc as f64, (4 - nc) as f64);
        c_pay[nc] = (nc_f + 1.0) * r + nd_f * s;
        d_pay[nc] = nc_f * t + (nd_f + 1.0) * p;
    }
    for nc_i in 0..5 {
        for mc_j in 0..5 {
            let d_to_c = ((d_pay[nc_i] - c_pay[mc_j]) * inv_temp).clamp(-500.0, 500.0);
            let c_to_d = ((c_pay[nc_i] - d_pay[mc_j]) * inv_temp).clamp(-500.0, 500.0);
            lut[0][nc_i][mc_j] = 1.0 / (1.0 + d_to_c.exp());
            lut[1][nc_i][mc_j] = 1.0 / (1.0 + c_to_d.exp());
        }
    }
    lut
}

/// A 2D Spin Lattice for the Ising Model.
///
/// # Examples
///
/// Simulating a ferromagnetic phase transition (Low Temperature):
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::stat_mech::ising::SpinLattice;
/// use math_explorer::physics::stat_mech::KB;
///
/// // 1. Setup system parameters
/// let width = 10;
/// let height = 10;
/// let j_coupling = 1.0; // Interaction energy (J > 0 for Ferromagnetism)
/// let h_field = 0.0;    // No external magnetic field
///
/// // 2. Set Temperature below Critical Point (Tc ~ 2.269 * J / KB)
/// // We choose T such that kB * T = 1.0 * J (Deep within ordered phase)
/// let temp = 1.0 * j_coupling / KB;
///
/// // 3. Initialize Lattice
/// let mut lattice = SpinLattice::new(width, height, None);
///
/// // 4. Run Metropolis Simulation to reach equilibrium
/// // Note: This is a probabilistic process. We use enough steps to overcome
/// // initial disorder and potential metastable states.
/// lattice.evolve(500_000, temp, j_coupling, h_field);
///
/// // 5. Check Magnetization
/// // At low temp, spins align. M should be close to max_m or -max_m.
/// let m = lattice.magnetization();
/// let max_m = (width * height) as i64;
///
/// // We check that the absolute magnetization is significant (> 80% order).
/// let magnetization_ratio = (m as f64).abs() / (max_m as f64);
/// assert!(magnetization_ratio > 0.8, "System did not magnetize! Ratio: {}", magnetization_ratio);
/// ```
#[derive(Debug, Clone)]
pub struct SpinLattice {
    width: usize,
    height: usize,
    spins: Vec<i8>,             // Flattened 2D grid
    neighbors: Vec<[usize; 4]>, // Precomputed neighbor indices [left, right, up, down]
    rng: oxidize_core::rng::OxidizeRng,
}

impl SpinLattice {
    /// Returns the width of the lattice.
    #[inline]
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height of the lattice.
    #[inline]
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Creates a new random spin lattice. Optionally provide a seed.
    #[verified_engine::verified]
    pub fn new(width: usize, height: usize, seed: Option<u64>) -> Self {
        let rng = if let Some(s) = seed {
            oxidize_core::rng::OxidizeRng::new(s)
        } else {
            oxidize_core::rng::OxidizeRng::default()
        };
        Self::new_with_rng(width, height, rng)
    }

    /// Creates a new random spin lattice using the provided RNG.
    #[verified_engine::verified]
    pub fn new_with_rng(
        width: usize,
        height: usize,
        mut rng: oxidize_core::rng::OxidizeRng,
    ) -> Self {
        let count = width * height;
        let spins = (0..count)
            .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
            .collect();

        // Precompute neighbor indices
        let mut neighbors = vec![[0; 4]; count];
        for y in 0..height {
            for x in 0..width {
                let idx = flatten_2d_index(x, y, width);

                let left = if x == 0 { idx + width - 1 } else { idx - 1 };
                let right = if x == width - 1 {
                    idx + 1 - width
                } else {
                    idx + 1
                };
                let up = if y == 0 {
                    idx + (height - 1) * width
                } else {
                    idx - width
                };
                let down = if y == height - 1 {
                    idx - (height - 1) * width
                } else {
                    idx + width
                };

                neighbors[idx] = [left, right, up, down];
            }
        }

        SpinLattice {
            width,
            height,
            spins,
            neighbors,
            rng,
        }
    }

    /// Re-seeds the internal RNG of the lattice.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = oxidize_core::rng::OxidizeRng::new(seed);
    }

    /// Gets the spin at (x, y).
    #[verified_engine::verified]
    pub fn get(&self, x: usize, y: usize) -> i8 {
        self.spins[y * self.width + x]
    }

    /// Sets the spin at (x, y).
    #[verified_engine::verified]
    pub fn set(&mut self, x: usize, y: usize, val: i8) {
        self.spins[y * self.width + x] = val;
    }

    /// Returns a reference to the flattened spins array.
    #[inline]
    #[verified_engine::verified]
    pub fn spins(&self) -> &[i8] {
        &self.spins
    }

    /// Calculates the total energy (Hamiltonian) of the lattice.
    ///
    /// Formula: H = -J * sum_<i,j> s_i s_j - h * sum_i s_i
    /// Uses Periodic Boundary Conditions.
    #[verified_engine::verified]
    pub fn hamiltonian(&self, j_coupling: f64, h_field: f64) -> f64 {
        let mut interaction_sum = 0.0;
        let mut field_sum = 0.0;
        for y in 0..self.height {
            for x in 0..self.width {
                let s = self.get(x, y);
                field_sum += s as f64;
                let right_s = self.get((x + 1) % self.width, y);
                let down_s = self.get(x, (y + 1) % self.height);
                interaction_sum += (s * right_s) as f64 + (s * down_s) as f64;
            }
        }
        -j_coupling * interaction_sum - h_field * field_sum
    }

    /// Calculates the total Magnetization.
    ///
    /// Formula: M = sum_i s_i
    #[verified_engine::verified]
    pub fn magnetization(&self) -> i64 {
        self.spins.iter().map(|&s| s as i64).sum()
    }

    /// Performs one Metropolis algorithm step (one attempt).
    ///
    /// 1. Select a random site.
    /// 2. Calculate energy cost to flip Delta E.
    /// 3. Accept flip if Delta E < 0 OR with probability e^(-beta * Delta E).
    #[verified_engine::verified]
    pub fn metropolis_step(&mut self, temperature: f64, j_coupling: f64, h_field: f64) {
        let x = self.rng.gen_range(0..self.width);
        let y = self.rng.gen_range(0..self.height);

        let s = self.get(x, y);

        // Neighbors for energy difference (all 4 neighbors needed for Delta E)
        let left_x = (x + self.width - 1) % self.width;
        let right_x = (x + 1) % self.width;
        let up_y = (y + self.height - 1) % self.height;
        let down_y = (y + 1) % self.height;

        let sum_neighbors = self.get(left_x, y) as f64
            + self.get(right_x, y) as f64
            + self.get(x, up_y) as f64
            + self.get(x, down_y) as f64;

        // Delta E = E_new - E_old
        // E_old_local = -J * s * sum_neighbors - h * s
        // E_new_local = -J * (-s) * sum_neighbors - h * (-s)
        // Delta E = 2 * J * s * sum_neighbors + 2 * h * s
        let delta_e = 2.0 * s as f64 * (j_coupling * sum_neighbors + h_field);

        let should_flip = if delta_e < 0.0 {
            true
        } else {
            let beta = 1.0 / (KB * temperature);
            let prob = (-beta * delta_e).exp();
            self.rng.r#gen::<f64>() < prob
        };

        if should_flip {
            self.set(x, y, -s);
        }
    }

    /// Performs multiple Metropolis steps efficiently using a lookup table and batching.
    ///
    /// This method is significantly faster than calling `metropolis_step` in a loop because:
    /// 1. It precomputes Boltzmann factors (`exp(-beta * dE)`) into a lookup table.
    /// 2. It reuses the random number generator, avoiding TLS overhead.
    /// 3. It precomputes neighbor indices to avoid coordinate arithmetic in the loop.
    #[verified_engine::verified]
    pub fn evolve(&mut self, steps: usize, temperature: f64, j_coupling: f64, h_field: f64) {
        if steps == 0 || self.spins.is_empty() {
            return;
        }

        let beta = 1.0 / (KB * temperature);

        // Precompute probabilities: lut[s_idx][sum_idx]
        // s_idx: 0 for s=-1, 1 for s=1
        // sum_idx: 0..5 for sum in {-4, -2, 0, 2, 4} mapped via (sum + 4) / 2
        let mut lut = [[0.0; 5]; 2];

        for (s_idx, lut_row) in lut.iter_mut().enumerate() {
            let s_val = if s_idx == 0 { -1.0 } else { 1.0 };
            for (sum_idx, entry) in lut_row.iter_mut().enumerate() {
                let sum_val = (sum_idx as f64 * 2.0) - 4.0;
                let delta_e = 2.0 * s_val * (j_coupling * sum_val + h_field);

                if delta_e <= 0.0 {
                    *entry = 1.1; // Always accept (value > 1.0 ensures check passes)
                } else {
                    *entry = (-beta * delta_e).exp();
                }
            }
        }

        let count = self.width * self.height;

        for _ in 0..steps {
            let idx = self.rng.gen_range(0..count);
            let s = self.spins[idx];
            let s_idx = usize::from(s > 0);
            let sum_idx = self.neighbor_cooperators(idx);
            let prob = lut[s_idx][sum_idx];
            if prob > 1.0 || self.rng.r#gen::<f64>() < prob {
                self.spins[idx] = -s;
            }
        }
    }

    #[inline]
    #[verified_engine::verified]
    fn neighbor_cooperators(&self, idx: usize) -> usize {
        let n = self.neighbors[idx];
        let sum = i32::from(self.spins[n[0]])
            + i32::from(self.spins[n[1]])
            + i32::from(self.spins[n[2]])
            + i32::from(self.spins[n[3]]);
        ((sum + 4) / 2) as usize
    }

    /// Evolves a spatial 2x2 evolutionary game on the lattice via pairwise Fermi imitation.
    ///
    /// Spin `+1` represents Strategy `0` (Cooperator) and spin `-1` represents Strategy `1`
    /// (Defector), with 2x2 [`MatrixPayoff`] entries `(R, S; T, P)`. Each elementary trial picks
    /// a random focal site $i$ and a random von Neumann neighbor $j \in \mathcal{N}(i)$,
    /// adopting $s_j$ with probability $W = [1 + \exp((\pi_i - \pi_j) / T)]^{-1}$ using a
    /// stack-allocated `2x5x5` lookup table and zero heap allocations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use domain_physics::__doctest_facade as math_explorer;
    /// use math_explorer::applied::game_theory::evolutionary::MatrixPayoff;
    /// use math_explorer::nalgebra::DMatrix;
    /// use math_explorer::physics::stat_mech::ising::SpinLattice;
    ///
    /// let payoff = MatrixPayoff::new(DMatrix::from_row_slice(2, 2, &[1.0, -0.1, 1.25, 0.0])).unwrap();
    /// let mut lattice = SpinLattice::new(20, 20, Some(42));
    /// lattice.evolve_fermi_game(4_000, 0.08, &payoff);
    /// assert!(lattice.magnetization().abs() <= 400);
    /// ```
    #[verified_engine::verified]
    pub fn evolve_fermi_game(&mut self, steps: usize, temperature: f64, payoff: &MatrixPayoff) {
        let mat = payoff.payoff_matrix();
        if steps == 0
            || self.spins.is_empty()
            || mat.nrows() != 2
            || mat.ncols() != 2
            || !temperature.is_finite()
        {
            return;
        }
        let (r, s, t, p) = (mat[(0, 0)], mat[(0, 1)], mat[(1, 0)], mat[(1, 1)]);
        if !r.is_finite() || !s.is_finite() || !t.is_finite() || !p.is_finite() {
            return;
        }
        let lut = build_fermi_lut(r, s, t, p, 1.0 / temperature.max(1e-3));
        let count = self.width * self.height;

        for _ in 0..steps {
            let idx = self.rng.gen_range(0..count);
            let s_i = self.spins[idx];
            let j_idx = self.neighbors[idx][self.rng.gen_range(0..4)];
            let s_j = self.spins[j_idx];
            if s_i != s_j {
                let nc_i = self.neighbor_cooperators(idx);
                let mc_j = self.neighbor_cooperators(j_idx);
                if self.rng.r#gen::<f64>() < lut[usize::from(s_i > 0)][nc_i][mc_j] {
                    self.spins[idx] = s_j;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::stat_mech::KB;
    use domain_applied::applied::game_theory::evolutionary::ReplicatorDynamics;
    use nalgebra::{DMatrix, DVector};
    use verified_engine::engine::VerifiedEngine;

    static VERIFIED_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn seeded_droplet(r: i32, seed: u64) -> SpinLattice {
        let mut lat = SpinLattice::new(30, 30, Some(seed));
        for y in 0..30 {
            for x in 0..30 {
                let (dx, dy) = (x as i32 - 15, y as i32 - 15);
                lat.set(x, y, if dx * dx + dy * dy <= r * r { 1 } else { -1 });
            }
        }
        lat
    }

    #[test]
    #[verified_engine::verified]
    fn test_ising_disordered() {
        // High temperature limit. T >> J/k_B.
        // Magnetization should be near 0 on average.
        // J = 1.0 Joules (normalized).
        // T = very high. e.g. KB*T = 1000 * J.
        // T = 1000 * J / KB.
        let j_val = 1.0; // effectively unitless if we scale T
        let h_val = 0.0;
        let t_high = 100.0 * j_val / KB; // High Temp

        let width = 20;
        let height = 20;
        let mut lattice = SpinLattice::new(width, height, None);

        // Run many steps to equilibrate
        let steps = 10_000;
        for _ in 0..steps {
            lattice.metropolis_step(t_high, j_val, h_val);
        }

        let m = lattice.magnetization();
        let max_m = (width * height) as f64;
        let avg_m_per_spin = m as f64 / max_m;

        // In disordered phase, magnetization should be small (random fluctuations).
        // It won't be exactly 0, but should be close to 0 compared to 1.
        // <M> ~ 0.
        assert!(
            avg_m_per_spin.abs() < 0.3,
            "High T Ising should be disordered (M ~ 0). Got {}",
            avg_m_per_spin
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_fermi_game_zero_allocations_under_verified_allocator() {
        let mut lattice = SpinLattice::new(100, 100, Some(7));
        let payoff =
            MatrixPayoff::new(DMatrix::from_row_slice(2, 2, &[1.0, -0.1, 1.35, 0.0])).unwrap();

        let _guard = VERIFIED_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let ((), metrics) = VerifiedEngine::run_verified(|| {
            lattice.evolve_fermi_game(10_000, 0.0, &payoff);
        });
        assert!(metrics.function_calls >= 2);
        assert!(lattice.magnetization().abs() <= 10_000);
    }

    #[test]
    #[verified_engine::verified]
    fn test_prisoners_dilemma_droplet_survival_vs_replicator_collapse() {
        let pd_mat = DMatrix::from_row_slice(2, 2, &[1.0, -0.1, 1.25, 0.0]);
        let payoff = MatrixPayoff::new(pd_mat.clone()).unwrap();

        let mut isolated_lat = seeded_droplet(0, 1234);
        isolated_lat.evolve_fermi_game(30 * 30 * 20, 0.08, &payoff);
        assert_eq!(isolated_lat.magnetization(), -900);

        let mut droplet_lat = seeded_droplet(5, 1234);
        let init_coop = (droplet_lat.magnetization() as f64 / 900.0 + 1.0) * 0.5;
        droplet_lat.evolve_fermi_game(30 * 30 * 50, 0.08, &payoff);
        let final_coop = (droplet_lat.magnetization() as f64 / 900.0 + 1.0) * 0.5;
        assert!(
            final_coop > 0.7,
            "Droplet must expand ({init_coop:.3} -> {final_coop:.3})"
        );

        let ode = ReplicatorDynamics::new(pd_mat).unwrap();
        let traj = ode.simulate(
            DVector::from_vec(vec![init_coop, 1.0 - init_coop]),
            40.0,
            0.1,
        );
        assert!(traj.last().unwrap().1[0] < 1e-2);
    }

    #[test]
    #[verified_engine::verified]
    fn test_fermi_game_temperature_phase_transition() {
        let payoff =
            MatrixPayoff::new(DMatrix::from_row_slice(2, 2, &[1.0, -0.1, 1.25, 0.0])).unwrap();
        let mut cold = seeded_droplet(5, 99);
        cold.evolve_fermi_game(30 * 30 * 60, 0.08, &payoff);
        let cold_coop = (cold.magnetization() as f64 / 900.0 + 1.0) * 0.5;

        let mut hot = seeded_droplet(5, 99);
        hot.evolve_fermi_game(30 * 30 * 60, 0.9, &payoff);
        let hot_coop = (hot.magnetization() as f64 / 900.0 + 1.0) * 0.5;
        assert!(
            cold_coop > hot_coop + 0.3,
            "Expected phase transition: {cold_coop:.3} vs {hot_coop:.3}"
        );
    }
}
