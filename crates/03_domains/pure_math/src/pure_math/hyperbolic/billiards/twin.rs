use super::physics::{
    FixedRing, find_next_table_hit, propagate_by_tau, reflect_direction_on_edge,
    tangent_bundle_separation, uniform_unit_f64,
};
use super::*;
use oxidize_core::rng::OxidizeRng;
use std::f64::consts::TAU;

/// Zero-allocation twin-ray hyperbolic billiard laboratory tracking Lyapunov divergence $\ln(\delta_t / \delta_0)$.
#[derive(Clone, Debug)]
pub struct TwinBilliardSystem {
    table: BilliardTable,
    primary: GeodesicRay,
    twin: GeodesicRay,
    delta0: f64,
    elapsed_length: f64,
    total_bounces: u64,
    primary_segments: FixedRing<GeodesicSegment>,
    twin_segments: FixedRing<GeodesicSegment>,
    edge_hits: FixedRing<u8>,
    separation_log: FixedRing<[f64; 2]>,
}

impl TwinBilliardSystem {
    /// Creates a pre-allocated [`TwinBilliardSystem`] with deterministic [`OxidizeRng`] seeding.
    #[verified_engine::verified]
    pub fn new(
        kind: BilliardDomainKind,
        seed: u64,
        angular_perturbation: f64,
    ) -> Result<Self, HyperbolicError> {
        if !angular_perturbation.is_finite() || angular_perturbation <= 0.0 {
            return Err(HyperbolicError::InvalidBilliardConfig {
                reason: "angular perturbation delta_0 must be finite and strictly positive",
            });
        }
        let table = BilliardTable::new(kind)?;
        let mut sys = Self {
            table,
            primary: GeodesicRay {
                position: DiskPoint::origin(),
                direction: Complex64::new(1.0, 0.0),
                last_edge: None,
            },
            twin: GeodesicRay {
                position: DiskPoint::origin(),
                direction: Complex64::new(1.0, 0.0),
                last_edge: None,
            },
            delta0: angular_perturbation,
            elapsed_length: 0.0,
            total_bounces: 0,
            primary_segments: FixedRing::with_default(GeodesicSegment::default()),
            twin_segments: FixedRing::with_default(GeodesicSegment::default()),
            edge_hits: FixedRing::with_default(0),
            separation_log: FixedRing::with_default([0.0, 0.0]),
        };
        sys.reseed(seed, angular_perturbation)?;
        assert!(sys.delta0 > 0.0);
        Ok(sys)
    }

    /// Re-initializes the billiard table and launch state in-place without heap allocation.
    #[verified_engine::verified]
    pub fn reconfigure(
        &mut self,
        kind: BilliardDomainKind,
        seed: u64,
        angular_perturbation: f64,
    ) -> Result<(), HyperbolicError> {
        self.table = BilliardTable::new(kind)?;
        self.reseed(seed, angular_perturbation)?;
        assert!(self.delta0 > 0.0);
        Ok(())
    }

    /// Resets ray states deterministically from `seed` and `angular_perturbation` using [`OxidizeRng`].
    #[verified_engine::verified]
    pub fn reseed(&mut self, seed: u64, angular_perturbation: f64) -> Result<(), HyperbolicError> {
        if !angular_perturbation.is_finite() || angular_perturbation <= 0.0 {
            return Err(HyperbolicError::InvalidBilliardConfig {
                reason: "angular perturbation delta_0 must be finite and strictly positive",
            });
        }
        let mut rng = OxidizeRng::new(seed);
        let max_r = match self.table.kind {
            BilliardDomainKind::RegularPolygon { .. } => {
                (0.35 * self.table.vertices[0].re.hypot(self.table.vertices[0].im)).min(0.3)
            }
            BilliardDomainKind::IdealPolygon { .. } => 0.22,
        };
        let r = 0.05 + (max_r - 0.05).max(0.02) * uniform_unit_f64(&mut rng);
        let phi = TAU * uniform_unit_f64(&mut rng);
        let theta = TAU * uniform_unit_f64(&mut rng);
        let eps = angular_perturbation.clamp(1.0e-12, 0.25);
        let pos = DiskPoint::new_clamped(r * phi.cos(), r * phi.sin());
        self.primary = GeodesicRay::new(pos, Complex64::new(theta.cos(), theta.sin()))?;
        self.twin = GeodesicRay::new(
            pos,
            Complex64::new((theta + eps).cos(), (theta + eps).sin()),
        )?;
        self.delta0 = tangent_bundle_separation(self.primary, self.twin).max(1.0e-14);
        self.elapsed_length = 0.0;
        self.total_bounces = 0;
        self.primary_segments.clear();
        self.twin_segments.clear();
        self.edge_hits.clear();
        self.separation_log.clear();
        self.separation_log.push([0.0, 0.0]);
        Ok(())
    }

    /// Resets the twin rays to explicit `(position, angle_rad, angular_perturbation)` with zero heap allocation.
    #[verified_engine::verified]
    pub fn set_launch_state(
        &mut self,
        position: DiskPoint,
        angle_rad: f64,
        angular_perturbation: f64,
    ) -> Result<(), HyperbolicError> {
        if !angle_rad.is_finite()
            || !angular_perturbation.is_finite()
            || angular_perturbation <= 0.0
        {
            return Err(HyperbolicError::InvalidBilliardConfig {
                reason: "launch angle must be finite and perturbation strictly positive",
            });
        }
        let eps = angular_perturbation.clamp(1.0e-12, 0.25);
        self.primary =
            GeodesicRay::new(position, Complex64::new(angle_rad.cos(), angle_rad.sin()))?;
        self.twin = GeodesicRay::new(
            position,
            Complex64::new((angle_rad + eps).cos(), (angle_rad + eps).sin()),
        )?;
        self.delta0 = tangent_bundle_separation(self.primary, self.twin).max(1.0e-14);
        self.elapsed_length = 0.0;
        self.total_bounces = 0;
        self.primary_segments.clear();
        self.twin_segments.clear();
        self.edge_hits.clear();
        self.separation_log.clear();
        self.separation_log.push([0.0, 0.0]);
        debug_assert!(self.delta0 > 0.0 && self.elapsed_length == 0.0);
        debug_assert_eq!(self.total_bounces, 0);
        Ok(())
    }

    /// Steps a single ray by hyperbolic arc length `ds`, recording completed segments in `ring`.
    #[verified_engine::verified]
    fn advance_ray_by_length(
        table: &BilliardTable,
        ray: &mut GeodesicRay,
        ds: f64,
        seg_ring: &mut FixedRing<GeodesicSegment>,
        hit_ring: Option<&mut FixedRing<u8>>,
        bounces: &mut u64,
    ) {
        assert!(ds >= 0.0 && ds.is_finite());
        let mut remaining = ds;
        let mut hit_log = hit_ring;
        for _ in 0..64 {
            if remaining <= 1.0e-12 {
                break;
            }
            let exit = find_next_table_hit(table, *ray);
            if remaining < exit.arc_len {
                let tau_part = (0.5 * remaining).tanh();
                let (next_pos, next_dir) = propagate_by_tau(ray.position, ray.direction, tau_part);
                ray.position = next_pos;
                ray.direction = next_dir;
                break;
            }
            let start = ray.position;
            let (hit_pos, u_in) = propagate_by_tau(ray.position, ray.direction, exit.tau);
            let (snapped_pos, u_out) =
                reflect_direction_on_edge(table, hit_pos, u_in, exit.hit_edge);
            seg_ring.push(GeodesicSegment {
                start,
                end: snapped_pos,
                hit_edge: exit.hit_edge,
                hyperbolic_length: exit.arc_len,
            });
            if let Some(h) = hit_log.as_deref_mut() {
                h.push(exit.hit_edge);
            }
            *bounces += 1;
            ray.position = snapped_pos;
            ray.direction = u_out;
            ray.last_edge = Some(exit.hit_edge);
            remaining -= exit.arc_len;
        }
    }

    /// Fallible companion to [`step`](Self::step).
    #[verified_engine::verified]
    pub fn try_step(
        &mut self,
        ds: f64,
    ) -> core::result::Result<(), math_commons::error::MathError> {
        if !ds.is_finite() || ds <= 0.0 {
            return Ok(());
        }
        let step_len = ds.min(4.0);
        let mut primary_bounces = 0u64;
        let mut dummy_bounces = 0u64;
        Self::advance_ray_by_length(
            &self.table,
            &mut self.primary,
            step_len,
            &mut self.primary_segments,
            Some(&mut self.edge_hits),
            &mut primary_bounces,
        );
        Self::advance_ray_by_length(
            &self.table,
            &mut self.twin,
            step_len,
            &mut self.twin_segments,
            None,
            &mut dummy_bounces,
        );
        self.total_bounces += primary_bounces;
        self.elapsed_length += step_len;
        let log_sep = self.current_log_separation();
        self.separation_log.push([self.elapsed_length, log_sep]);
        crate::error::ensure_math(
            log_sep.is_finite(),
            "step: precondition or invariant violated",
        )?;
        debug_assert!(self.elapsed_length >= step_len);
        debug_assert!(log_sep.is_finite());
        Ok(())
    }

    /// Advances both rays by hyperbolic arc-length `ds > 0` with zero dynamic heap allocations.
    #[verified_engine::verified]
    pub fn step(&mut self, ds: f64) {
        self.try_step(ds).expect("step preconditions must hold");
    }

    /// Advances the system through `reflections` exact primary-ray boundary bounces with zero allocations.
    #[verified_engine::verified]
    pub fn advance_reflections(&mut self, reflections: usize) {
        let capped = reflections.min(MAX_BILLIARD_SEGMENTS);
        for _ in 0..capped {
            let exit = find_next_table_hit(&self.table, self.primary);
            self.step(exit.arc_len + 1.0e-9);
        }
        debug_assert!(self.elapsed_length >= 0.0);
    }

    /// Fallible companion to [`current_log_separation`](Self::current_log_separation).
    #[verified_engine::verified]
    pub fn try_current_log_separation(
        &self,
    ) -> core::result::Result<f64, math_commons::error::MathError> {
        let delta_t = tangent_bundle_separation(self.primary, self.twin).max(1.0e-16);
        let val = (delta_t / self.delta0).ln();
        crate::error::ensure_math(
            val.is_finite(),
            "current_log_separation: precondition or invariant violated",
        )?;
        debug_assert!(delta_t >= 1.0e-16 && val.is_finite());
        Ok(val)
    }

    /// Returns the instantaneous logarithmic separation $\ln(\delta_t / \delta_0)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn current_log_separation(&self) -> f64 {
        self.try_current_log_separation()
            .expect("current_log_separation preconditions must hold")
    }

    /// Returns a reference to the [`BilliardTable`].
    #[must_use]
    #[verified_engine::verified]
    pub fn table(&self) -> &BilliardTable {
        debug_assert!(self.table.vertex_count >= 3);
        &self.table
    }

    /// Returns the primary [`GeodesicRay`].
    #[must_use]
    #[verified_engine::verified]
    pub fn primary_ray(&self) -> GeodesicRay {
        debug_assert!((self.primary.direction.norm_sqr() - 1.0).abs() < 1.0e-8);
        self.primary
    }

    /// Returns the perturbed twin [`GeodesicRay`].
    #[must_use]
    #[verified_engine::verified]
    pub fn twin_ray(&self) -> GeodesicRay {
        debug_assert!((self.twin.direction.norm_sqr() - 1.0).abs() < 1.0e-8);
        self.twin
    }

    /// Fallible companion to [`elapsed_length`](Self::elapsed_length).
    #[verified_engine::verified]
    pub fn try_elapsed_length(&self) -> core::result::Result<f64, math_commons::error::MathError> {
        crate::error::ensure_math(
            self.elapsed_length >= 0.0,
            "elapsed_length: precondition or invariant violated",
        )?;
        debug_assert!(self.elapsed_length >= 0.0);
        Ok(self.elapsed_length)
    }

    /// Returns cumulative hyperbolic arc length elapsed.
    #[must_use]
    #[verified_engine::verified]
    pub fn elapsed_length(&self) -> f64 {
        self.try_elapsed_length()
            .expect("elapsed_length preconditions must hold")
    }

    /// Returns total primary-ray boundary reflections recorded.
    #[must_use]
    #[verified_engine::verified]
    pub fn total_bounces(&self) -> u64 {
        debug_assert!(self.elapsed_length >= 0.0);
        self.total_bounces
    }

    /// Copies buffered primary ray segments chronologically into `out` without heap allocation.
    #[verified_engine::verified]
    pub fn copy_primary_segments_into(&self, out: &mut [GeodesicSegment]) -> usize {
        let n = self.primary_segments.copy_chronological_into(out);
        debug_assert!(n <= out.len());
        n
    }

    /// Copies buffered twin ray segments chronologically into `out` without heap allocation.
    #[verified_engine::verified]
    pub fn copy_twin_segments_into(&self, out: &mut [GeodesicSegment]) -> usize {
        let n = self.twin_segments.copy_chronological_into(out);
        debug_assert!(n <= out.len());
        n
    }

    /// Copies buffered symbolic edge-hit codes chronologically into `out` without heap allocation.
    #[verified_engine::verified]
    pub fn copy_edge_hits_into(&self, out: &mut [u8]) -> usize {
        let n = self.edge_hits.copy_chronological_into(out);
        debug_assert!(n <= out.len());
        n
    }

    /// Copies buffered `[hyperbolic_length, ln(delta_t / delta_0)]` samples into `out` without heap allocation.
    #[verified_engine::verified]
    pub fn copy_separation_series_into(&self, out: &mut [[f64; 2]]) -> usize {
        let n = self.separation_log.copy_chronological_into(out);
        debug_assert!(n <= out.len());
        n
    }
}
