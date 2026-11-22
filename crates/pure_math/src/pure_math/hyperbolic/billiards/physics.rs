// [cite:pure_math]
use super::{BilliardTable, GeodesicRay, MAX_BILLIARD_SEGMENTS};
use crate::pure_math::hyperbolic::{DiskPoint, GeodesicDisk, Psu11, hyperbolic_distance_disk};
use num_complex::Complex64;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

/// Exact exit collision data for a ray leaving `ray.position` inside `table`.
#[derive(Clone, Copy, Debug)]
pub(super) struct ExitHit {
    pub(super) tau: f64,
    pub(super) arc_len: f64,
    pub(super) hit_edge: u8,
}

/// Computes the exact next collision parameter $\tau \in (0, 1)$ of `ray` with `table`.
#[verified_engine::verified]
pub(super) fn find_next_table_hit(table: &BilliardTable, ray: GeodesicRay) -> ExitHit {
    let to_orig = Psu11::translation_to_origin(ray.position);
    let n = table.vertex_count;
    let mut best_tau = 1.0 - 1.0e-9;
    let mut best_edge = 0u8;
    let mut found = false;
    for k in 0..n {
        if ray.last_edge == Some(k as u8) {
            continue;
        }
        let v0 = to_orig.apply_complex(table.vertices[k]);
        let v1 = to_orig.apply_complex(table.vertices[(k + 1) % n]);
        if let Ok(GeodesicDisk::OrthogonalArc { center, .. }) = GeodesicDisk::through_points(v0, v1)
        {
            let b = center.re * ray.direction.re + center.im * ray.direction.im;
            if b > 1.0 + 1.0e-11 {
                let tau = 1.0 / (b + (b * b - 1.0).sqrt());
                if tau > 1.0e-12 && tau < best_tau {
                    best_tau = tau;
                    best_edge = k as u8;
                    found = true;
                }
            }
        }
    }
    if !found {
        best_edge = ray.last_edge.map_or(0, |e| (e + 1) % (n as u8));
    }
    let clamped_tau = best_tau.clamp(1.0e-11, 1.0 - 1.0e-9);
    let arc_len = 2.0 * clamped_tau.atanh();
    assert!(arc_len > 0.0 && arc_len.is_finite());
    ExitHit {
        tau: clamped_tau,
        arc_len,
        hit_edge: best_edge,
    }
}

/// Advances `(z0, u0)` along a unit-speed geodesic by parameter $\tau \in [0, 1)$.
#[verified_engine::verified]
pub(super) fn propagate_by_tau(z0: DiskPoint, u0: Complex64, tau: f64) -> (DiskPoint, Complex64) {
    assert!((0.0..1.0).contains(&tau));
    let z = z0.as_complex();
    let zeta = Complex64::new(1.0, 0.0) + z.conj() * (u0 * tau);
    let zeta_norm = zeta.re.hypot(zeta.im).max(1.0e-15);
    let phase = Complex64::new(zeta.re / zeta_norm, -zeta.im / zeta_norm);
    let u_new_raw = u0 * phase * phase;
    let u_norm = u_new_raw.re.hypot(u_new_raw.im).max(1.0e-15);
    let u_new = Complex64::new(u_new_raw.re / u_norm, u_new_raw.im / u_norm);
    let pos_new = DiskPoint::from_complex_clamped((u0 * tau + z) / zeta);
    debug_assert!((u_new.norm_sqr() - 1.0).abs() < 1.0e-9);
    (pos_new, u_new)
}

/// Applies exact conformal reflection of `u_in` off `hit_edge` of `table` at `hit_pos`.
#[verified_engine::verified]
pub(super) fn reflect_direction_on_edge(
    table: &BilliardTable,
    hit_pos: DiskPoint,
    u_in: Complex64,
    hit_edge: u8,
) -> (DiskPoint, Complex64) {
    let idx = usize::from(hit_edge) % table.vertex_count;
    let center = table.edge_centers[idx];
    let radius = table.edge_radii[idx];
    let radial = hit_pos.as_complex() - center;
    let r_norm = radial.re.hypot(radial.im).max(1.0e-15);
    let normal = Complex64::new(radial.re / r_norm, radial.im / r_norm);
    let snapped = DiskPoint::from_complex_clamped(center + normal * (radius + 1.0e-13));
    let dot = normal.re * u_in.re + normal.im * u_in.im;
    let reflected = u_in - normal * (2.0 * dot);
    let ref_norm = reflected.re.hypot(reflected.im).max(1.0e-15);
    let unit_out = Complex64::new(reflected.re / ref_norm, reflected.im / ref_norm);
    assert!((unit_out.norm_sqr() - 1.0).abs() < 1.0e-9);
    (snapped, unit_out)
}

/// Evaluates the `PSU(1,1)`-invariant tangent-bundle distance $\delta = \sqrt{d_{\mathbb{D}}^2 + \Delta\theta^2}$.
#[verified_engine::verified]
pub(super) fn tangent_bundle_separation(r1: GeodesicRay, r2: GeodesicRay) -> f64 {
    let d_pos = hyperbolic_distance_disk(r1.position, r2.position);
    let z1 = r1.position.as_complex();
    let z2 = r2.position.as_complex();
    let denom = Complex64::new(1.0, 0.0) - z1.conj() * z2;
    let den_norm = denom.re.hypot(denom.im).max(1.0e-15);
    let phase = Complex64::new(denom.re / den_norm, -denom.im / den_norm);
    let u2_transported = r2.direction * phase * phase;
    let cross = r1.direction.re * u2_transported.im - r1.direction.im * u2_transported.re;
    let dot = r1.direction.re * u2_transported.re + r1.direction.im * u2_transported.im;
    let d_theta = cross.atan2(dot);
    let delta = d_pos.hypot(d_theta);
    assert!(delta.is_finite() && delta >= 0.0);
    delta
}

/// Samples a deterministic uniform `f64` in `[0.0, 1.0)` from `OxidizeRng`.
#[verified_engine::verified]
pub(super) fn uniform_unit_f64(rng: &mut OxidizeRng) -> f64 {
    const INV_2POW53: f64 = 1.0 / ((1u64 << 53) as f64);
    let val = ((rng.next_u64() >> 11) as f64) * INV_2POW53;
    assert!((0.0..1.0).contains(&val));
    val
}

/// Pre-allocated fixed-capacity ring buffer (`MAX_BILLIARD_SEGMENTS`) for zero-allocation storage.
#[derive(Clone, Debug)]
pub(super) struct FixedRing<T: Copy> {
    buf: Vec<T>,
    head: usize,
    len: usize,
}

impl<T: Copy> FixedRing<T> {
    #[verified_engine::verified]
    pub(super) fn with_default(fill: T) -> Self {
        let buf = vec![fill; MAX_BILLIARD_SEGMENTS];
        assert_eq!(buf.len(), MAX_BILLIARD_SEGMENTS);
        Self {
            buf,
            head: 0,
            len: 0,
        }
    }

    #[verified_engine::verified]
    pub(super) fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
        debug_assert_eq!(self.len, 0);
    }

    #[verified_engine::verified]
    pub(super) fn push(&mut self, item: T) {
        assert_eq!(self.buf.len(), MAX_BILLIARD_SEGMENTS);
        if self.len < MAX_BILLIARD_SEGMENTS {
            let write_idx = (self.head + self.len) % MAX_BILLIARD_SEGMENTS;
            self.buf[write_idx] = item;
            self.len += 1;
        } else {
            self.buf[self.head] = item;
            self.head = (self.head + 1) % MAX_BILLIARD_SEGMENTS;
        }
    }

    #[verified_engine::verified]
    pub(super) fn copy_chronological_into(&self, out: &mut [T]) -> usize {
        assert_eq!(self.buf.len(), MAX_BILLIARD_SEGMENTS);
        let count = self.len.min(out.len());
        let skip = self.len - count;
        for (i, slot) in out.iter_mut().take(count).enumerate() {
            let idx = (self.head + skip + i) % MAX_BILLIARD_SEGMENTS;
            *slot = self.buf[idx];
        }
        count
    }
}
