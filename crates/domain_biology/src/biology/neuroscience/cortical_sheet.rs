//! Thickness-Weighted Cortical Hodgkin-Huxley Sheet Solver over Cotangent Triangle Meshes.
//!
//! Couples vertex-wise [`HodgkinHuxleyNeuron`] membrane potentials through diffusive surface
//! currents scaled by local FreeSurfer white-to-pial cortical thickness over [`CotangentLaplacian`].
// [cite:freesurfer]

use super::neuron::HodgkinHuxleyNeuron;
use pure_math::pure_math::differential_geometry::cotangent_mesh::CotangentLaplacian;

const MIN_THICKNESS_SCALE: f64 = 0.1;
const MAX_THICKNESS_SCALE: f64 = 4.0;

/// Coupled Hodgkin-Huxley excitable neural sheet over a 3D triangulated manifold.
///
/// Pre-allocates membrane potential, external brush stimulus, and normalized cortical thickness
/// buffers so `step` executes with zero heap allocations under `VerifiedEngine`.
#[derive(Debug, Clone)]
pub struct CorticalSheetRunner {
    neurons: Vec<HodgkinHuxleyNeuron>,
    potentials_mv: Vec<f64>,
    stimulus_ua_cm2: Vec<f64>,
    thickness_weights: Vec<f64>,
    laplacian: CotangentLaplacian,
    sheet_conductance: f64,
}

#[verified_engine::verified]
fn normalize_thickness_weight(thickness_mm: f64, reference_mm: f64) -> f64 {
    let ref_safe = if reference_mm.is_finite() && reference_mm > 1e-6 {
        reference_mm
    } else {
        2.5
    };
    if thickness_mm.is_finite() && thickness_mm > 0.0 {
        (thickness_mm / ref_safe).clamp(MIN_THICKNESS_SCALE, MAX_THICKNESS_SCALE)
    } else {
        1.0
    }
}

impl CorticalSheetRunner {
    /// Creates a new cortical sheet over `laplacian` with local `cortical_thickness_mm` values.
    #[must_use]
    pub fn new(
        laplacian: CotangentLaplacian,
        cortical_thickness_mm: &[f64],
        reference_thickness_mm: f64,
        sheet_conductance: f64,
        v_rest_mv: f64,
    ) -> Self {
        let n = laplacian.num_vertices();
        let mut neurons = Vec::with_capacity(n);
        let mut potentials_mv = vec![v_rest_mv; n];
        let mut thickness_weights = vec![1.0; n];
        for i in 0..n {
            let neuron = HodgkinHuxleyNeuron::new(v_rest_mv);
            potentials_mv[i] = neuron.v();
            neurons.push(neuron);
            if let Some(&t_mm) = cortical_thickness_mm.get(i) {
                thickness_weights[i] = normalize_thickness_weight(t_mm, reference_thickness_mm);
            }
        }
        Self {
            neurons,
            potentials_mv,
            stimulus_ua_cm2: vec![0.0; n],
            thickness_weights,
            laplacian,
            sheet_conductance: sheet_conductance.max(0.0),
        }
    }

    /// Updates per-vertex cortical thickness conductance weights in place without heap allocation.
    #[verified_engine::verified]
    pub fn update_thickness_map(
        &mut self,
        cortical_thickness_mm: &[f64],
        reference_thickness_mm: f64,
    ) {
        let count = self
            .thickness_weights
            .len()
            .min(cortical_thickness_mm.len());
        for (dst, &t_mm) in self
            .thickness_weights
            .iter_mut()
            .zip(cortical_thickness_mm.iter())
            .take(count)
        {
            *dst = normalize_thickness_weight(t_mm, reference_thickness_mm);
        }
    }

    /// Number of cortical vertices in the sheet.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn num_vertices(&self) -> usize {
        self.neurons.len()
    }

    /// Underlying CSR [`CotangentLaplacian`] operator.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn laplacian(&self) -> &CotangentLaplacian {
        &self.laplacian
    }

    /// Replaces the CSR [`CotangentLaplacian`] when vertex positions deform (must preserve vertex count).
    #[verified_engine::verified]
    pub fn set_laplacian(&mut self, laplacian: CotangentLaplacian) {
        assert_eq!(
            laplacian.num_vertices(),
            self.neurons.len(),
            "Updated CotangentLaplacian vertex count mismatch"
        );
        self.laplacian = laplacian;
    }

    /// Sheet lateral coupling conductance ($\text{mS}/\text{cm}^2 \cdot \text{mm}^2$).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn sheet_conductance(&self) -> f64 {
        self.sheet_conductance
    }

    /// Sets the lateral surface coupling conductance.
    #[inline]
    #[verified_engine::verified]
    pub fn set_sheet_conductance(&mut self, conductance: f64) {
        if conductance.is_finite() {
            self.sheet_conductance = conductance.max(0.0);
        }
    }

    /// Contiguous slice of vertex membrane potentials (mV).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn potentials(&self) -> &[f64] {
        &self.potentials_mv
    }

    /// Normalized dimensionless cortical thickness weights $T(v_i) / T_{\text{ref}}$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn thickness_weights(&self) -> &[f64] {
        &self.thickness_weights
    }

    /// Borrows the [`HodgkinHuxleyNeuron`] at `vertex`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn neuron(&self, vertex: usize) -> &HodgkinHuxleyNeuron {
        &self.neurons[vertex]
    }

    /// Sets sustained external stimulus current ($\mu\text{A}/\text{cm}^2$) at `vertex`.
    #[verified_engine::verified]
    pub fn set_stimulus(&mut self, vertex: usize, current_ua_cm2: f64) {
        if vertex < self.stimulus_ua_cm2.len() && current_ua_cm2.is_finite() {
            self.stimulus_ua_cm2[vertex] = current_ua_cm2;
        }
    }

    /// Adds external stimulus current ($\mu\text{A}/\text{cm}^2$) at `vertex`.
    #[verified_engine::verified]
    pub fn add_stimulus(&mut self, vertex: usize, delta_ua_cm2: f64) {
        if vertex < self.stimulus_ua_cm2.len() && delta_ua_cm2.is_finite() {
            self.stimulus_ua_cm2[vertex] =
                (self.stimulus_ua_cm2[vertex] + delta_ua_cm2).clamp(-100.0, 100.0);
        }
    }

    /// Clears all external brush stimulus currents across the sheet.
    #[verified_engine::verified]
    pub fn clear_stimulus(&mut self) {
        self.stimulus_ua_cm2.fill(0.0);
    }

    /// Perturbs or clamps membrane potential at `vertex` (mV) without heap allocation.
    #[verified_engine::verified]
    pub fn depolarize_vertex(&mut self, vertex: usize, target_mv: f64) {
        if vertex < self.neurons.len() && target_mv.is_finite() {
            let clamped = target_mv.clamp(-90.0, 60.0);
            self.neurons[vertex].set_v(clamped);
            self.potentials_mv[vertex] = clamped;
        }
    }

    /// Advances all coupled [`HodgkinHuxleyNeuron`] cells by `dt_ms` with zero heap allocations.
    #[verified_engine::verified]
    pub fn step(&mut self, dt_ms: f64) {
        let n = self.neurons.len();
        if n == 0 || !dt_ms.is_finite() || dt_ms <= 0.0 {
            return;
        }
        let clamped_dt = dt_ms.min(0.32);
        let substeps = ((clamped_dt / 0.01).ceil() as usize).clamp(1, 32);
        let sub_dt = clamped_dt / (substeps as f64);
        let g_sheet = self.sheet_conductance;

        for _ in 0..substeps {
            for (dst, neuron) in self.potentials_mv.iter_mut().zip(self.neurons.iter()) {
                *dst = neuron.v();
            }
            for v in 0..n {
                let v_curr = self.potentials_mv[v];
                let s_v = self.thickness_weights[v];
                let neighbors = self.laplacian.neighbors(v);
                let weights = self.laplacian.row_weights(v);
                let mut flux = 0.0;
                for (&j, &w) in neighbors.iter().zip(weights.iter()) {
                    let edge_scale = 0.5 * (s_v + self.thickness_weights[j]);
                    flux += w * edge_scale * (self.potentials_mv[j] - v_curr);
                }
                let i_diff =
                    (g_sheet * flux * self.laplacian.inv_vertex_area(v)).clamp(-250.0, 250.0);
                let i_total = (self.stimulus_ua_cm2[v] + i_diff).clamp(-350.0, 350.0);
                self.neurons[v].update(sub_dt, i_total);
            }
        }
        for (dst, neuron) in self.potentials_mv.iter_mut().zip(self.neurons.iter()) {
            *dst = neuron.v();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use domain_applied::applied::freesurfer::{Surface, cortical_thickness_batch};
    use std::f64::consts::PI;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_cortical_sheet_runner_100_steps_zero_alloc_and_wave_propagation() {
        let (nu, nv) = (16, 12);
        let mut white_verts = vec![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
        let mut pial_verts = vec![[0.0, 0.0, 1.03], [0.0, 0.0, -1.02]];
        for j in 1..(nv - 1) {
            let (sv, cv) = (PI * (j as f64) / ((nv - 1) as f64)).sin_cos();
            for i in 0..nu {
                let u = 2.0 * PI * (i as f64) / (nu as f64);
                let (su, cu) = u.sin_cos();
                let dir = [sv * cu, sv * su, cv];
                let scale = 1.0 + 0.02 + 0.015 * (3.0 * u).cos().max(0.0);
                white_verts.push(dir);
                pial_verts.push([dir[0] * scale, dir[1] * scale, dir[2] * scale]);
            }
        }
        let ring = |i: usize, j: usize| 2 + (j - 1) * nu + (i % nu);
        let mut tris = Vec::new();
        for i in 0..nu {
            tris.push([0, ring(i, 1), ring(i + 1, 1)]);
        }
        for j in 1..(nv - 2) {
            for i in 0..nu {
                tris.push([ring(i, j), ring(i, j + 1), ring(i + 1, j + 1)]);
                tris.push([ring(i, j), ring(i + 1, j + 1), ring(i + 1, j)]);
            }
        }
        for i in 0..nu {
            tris.push([ring(i, nv - 2), 1, ring(i + 1, nv - 2)]);
        }

        let white_surf = Surface::new(white_verts.clone(), tris.clone());
        let pial_surf = Surface::new(pial_verts, tris.clone());
        let thickness_map = cortical_thickness_batch(&white_surf, &pial_surf);
        let lap = CotangentLaplacian::from_indexed(&white_verts, &tris);
        let mut runner = CorticalSheetRunner::new(lap, &thickness_map, 0.025, 0.25, -65.0);
        runner.depolarize_vertex(0, 15.0);
        runner.set_stimulus(0, 20.0);

        let ((), _metrics) = VerifiedEngine::run_verified(|| {
            for _ in 0..100 {
                runner.step(0.02);
            }
        });

        assert!(runner.potentials().iter().all(|v| v.is_finite()));
        let neighbor = runner.laplacian().neighbors(0)[0];
        assert!(
            runner.potentials()[neighbor] > -64.5,
            "Depolarizing wave failed to couple to neighbor: {}",
            runner.potentials()[neighbor]
        );
    }
}
