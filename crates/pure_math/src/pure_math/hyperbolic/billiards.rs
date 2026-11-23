// [cite:pure_math]
mod physics;

use self::physics::{
    FixedRing, find_next_table_hit, propagate_by_tau, reflect_direction_on_edge,
    tangent_bundle_separation, uniform_unit_f64,
};
use super::{
    DiskPoint, GeodesicDisk, HyperbolicError, MAX_POLYGON_VERTICES, ideal_polygon_vertices,
    regular_polygon_vertices,
};
use num_complex::Complex64;
use oxidize_core::rng::OxidizeRng;
use std::f64::consts::TAU;

/// Maximum number of geodesic segments retained in pre-allocated billiard ring buffers.
pub const MAX_BILLIARD_SEGMENTS: usize = 2048;

/// Convex hyperbolic billiard table geometry in $\overline{\mathbb{D}}$.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BilliardDomainKind {
    /// Compact regular `{p, q}` polygon with $(p - 2)(q - 2) > 4$.
    RegularPolygon {
        /// Polygon edge count ($3 \le p \le 16$).
        p: u32,
        /// Vertex valence ($q \ge 3$).
        q: u32,
    },
    /// Ideal regular $n$-gon with vertices on $\partial\mathbb{D}$ ($3 \le n \le 16$).
    IdealPolygon {
        /// Number of ideal sides ($3 \le \text{sides} \le 16$).
        sides: u32,
    },
}

/// Precomputed convex hyperbolic polygon table for zero-allocation ray reflections.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BilliardTable {
    kind: BilliardDomainKind,
    vertices: [Complex64; MAX_POLYGON_VERTICES],
    edge_centers: [Complex64; MAX_POLYGON_VERTICES],
    edge_radii: [f64; MAX_POLYGON_VERTICES],
    vertex_count: usize,
}

impl BilliardTable {
    /// Builds a validated [`BilliardTable`] from `kind`.
    #[verified_engine::verified]
    pub fn new(kind: BilliardDomainKind) -> Result<Self, HyperbolicError> {
        let raw_verts = match kind {
            BilliardDomainKind::RegularPolygon { p, q } => {
                if (p as usize) > MAX_POLYGON_VERTICES {
                    return Err(HyperbolicError::InvalidBilliardConfig {
                        reason: "billiard polygon side count exceeds MAX_POLYGON_VERTICES (16)",
                    });
                }
                regular_polygon_vertices(p, q)?
            }
            BilliardDomainKind::IdealPolygon { sides } => ideal_polygon_vertices(sides)?,
        };
        let n = raw_verts.len();
        let mut vertices = [Complex64::new(0.0, 0.0); MAX_POLYGON_VERTICES];
        let mut edge_centers = [Complex64::new(0.0, 0.0); MAX_POLYGON_VERTICES];
        let mut edge_radii = [0.0; MAX_POLYGON_VERTICES];
        for k in 0..n {
            vertices[k] = raw_verts[k];
        }
        for k in 0..n {
            let v0 = vertices[k];
            let v1 = vertices[(k + 1) % n];
            match GeodesicDisk::through_points(v0, v1)? {
                GeodesicDisk::OrthogonalArc { center, radius } => {
                    edge_centers[k] = center;
                    edge_radii[k] = radius;
                }
                GeodesicDisk::Diameter { .. } => {
                    return Err(HyperbolicError::InvalidBilliardConfig {
                        reason: "central billiard table edge unexpectedly passed through origin",
                    });
                }
            }
        }
        assert!((3..=MAX_POLYGON_VERTICES).contains(&n));
        Ok(Self {
            kind,
            vertices,
            edge_centers,
            edge_radii,
            vertex_count: n,
        })
    }

    /// Returns the [`BilliardDomainKind`] of this table.
    #[must_use]
    #[verified_engine::verified]
    pub fn kind(&self) -> BilliardDomainKind {
        debug_assert!(self.vertex_count >= 3);
        self.kind
    }

    /// Returns the active counterclockwise polygon vertices `&[Complex64]`.
    #[must_use]
    #[verified_engine::verified]
    pub fn vertices(&self) -> &[Complex64] {
        assert!(self.vertex_count <= MAX_POLYGON_VERTICES);
        &self.vertices[..self.vertex_count]
    }
}

/// Instantaneous state of a unit-speed geodesic billiard particle on $\mathbb{D}$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeodesicRay {
    /// Interior position $z \in \mathbb{D}$.
    pub position: DiskPoint,
    /// Unit Euclidean tangent direction $u$ ($|u| = 1$) at `position`.
    pub direction: Complex64,
    /// Index of the most recently struck boundary edge, if any.
    pub last_edge: Option<u8>,
}

impl GeodesicRay {
    /// Constructs a validated [`GeodesicRay`] normalizing `direction`.
    #[verified_engine::verified]
    pub fn new(position: DiskPoint, direction: Complex64) -> Result<Self, HyperbolicError> {
        let norm = direction.re.hypot(direction.im);
        if !norm.is_finite() || norm <= 1.0e-12 {
            return Err(HyperbolicError::InvalidBilliardConfig {
                reason: "ray direction vector must be finite and non-zero",
            });
        }
        let u = Complex64::new(direction.re / norm, direction.im / norm);
        assert!((u.norm_sqr() - 1.0).abs() < 1.0e-10);
        Ok(Self {
            position,
            direction: u,
            last_edge: None,
        })
    }
}

/// Single chordal geodesic flight segment between billiard boundary impacts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeodesicSegment {
    /// Segment start coordinate in $\mathbb{D}$.
    pub start: DiskPoint,
    /// Segment end coordinate in $\mathbb{D}$.
    pub end: DiskPoint,
    /// Struck boundary edge index (`0..vertex_count`).
    pub hit_edge: u8,
    /// Hyperbolic arc length $\Delta s > 0$ traversed along this segment.
    pub hyperbolic_length: f64,
}

impl Default for GeodesicSegment {
    #[verified_engine::verified]
    fn default() -> Self {
        let seg = Self {
            start: DiskPoint::origin(),
            end: DiskPoint::origin(),
            hit_edge: 0,
            hyperbolic_length: 0.0,
        };
        debug_assert_eq!(seg.hyperbolic_length, 0.0);
        seg
    }
}

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

    /// Advances both rays by hyperbolic arc-length `ds > 0` with zero dynamic heap allocations.
    #[verified_engine::verified]
    pub fn step(&mut self, ds: f64) {
        if !ds.is_finite() || ds <= 0.0 {
            return;
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
        assert!(log_sep.is_finite());
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

    /// Returns the instantaneous logarithmic separation $\ln(\delta_t / \delta_0)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn current_log_separation(&self) -> f64 {
        let delta_t = tangent_bundle_separation(self.primary, self.twin).max(1.0e-16);
        let val = (delta_t / self.delta0).ln();
        assert!(val.is_finite());
        val
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

    /// Returns cumulative hyperbolic arc length elapsed.
    #[must_use]
    #[verified_engine::verified]
    pub fn elapsed_length(&self) -> f64 {
        assert!(self.elapsed_length >= 0.0);
        self.elapsed_length
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
        self.primary_segments.copy_chronological_into(out)
    }

    /// Copies buffered twin ray segments chronologically into `out` without heap allocation.
    #[verified_engine::verified]
    pub fn copy_twin_segments_into(&self, out: &mut [GeodesicSegment]) -> usize {
        self.twin_segments.copy_chronological_into(out)
    }

    /// Copies buffered symbolic edge-hit codes chronologically into `out` without heap allocation.
    #[verified_engine::verified]
    pub fn copy_edge_hits_into(&self, out: &mut [u8]) -> usize {
        self.edge_hits.copy_chronological_into(out)
    }

    /// Copies buffered `[hyperbolic_length, ln(delta_t / delta_0)]` samples into `out` without heap allocation.
    #[verified_engine::verified]
    pub fn copy_separation_series_into(&self, out: &mut [[f64; 2]]) -> usize {
        self.separation_log.copy_chronological_into(out)
    }
}
