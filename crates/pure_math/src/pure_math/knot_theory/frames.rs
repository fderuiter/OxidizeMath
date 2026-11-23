//! Rotation-minimizing Bishop parallel-transport frames on closed spatial curves.

// [cite:knot_braid_topology]

use super::KnotError;
use nalgebra::{Point3, Vector3};

/// Orthonormal right-handed Bishop frame $(T, U, V)$ at a curve sample point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BishopFrame {
    /// Unit tangent vector $T$.
    pub tangent: Vector3<f64>,
    /// First parallel-transported unit normal $U \perp T$.
    pub normal_u: Vector3<f64>,
    /// Second parallel-transported unit normal $V = T \times U$.
    pub normal_v: Vector3<f64>,
}

impl Default for BishopFrame {
    fn default() -> Self {
        Self {
            tangent: Vector3::x(),
            normal_u: Vector3::y(),
            normal_v: Vector3::z(),
        }
    }
}

#[verified_engine::verified]
fn choose_initial_normal(t0: &Vector3<f64>) -> Vector3<f64> {
    let ref_axis = if t0.x.abs() <= t0.y.abs() && t0.x.abs() <= t0.z.abs() {
        Vector3::x()
    } else if t0.y.abs() <= t0.z.abs() {
        Vector3::y()
    } else {
        Vector3::z()
    };
    let proj = ref_axis - t0 * ref_axis.dot(t0);
    proj.normalize()
}

/// Transports `u` from unit tangent `t_prev` to `t_next` using Wang's double-reflection method
/// (exact discrete parallel transport free of Frenet inflection flips).
#[verified_engine::verified]
fn parallel_transport_normal(
    t_prev: &Vector3<f64>,
    t_next: &Vector3<f64>,
    u_prev: &Vector3<f64>,
) -> Vector3<f64> {
    let sum = t_prev + t_next;
    let sum_sq = sum.norm_squared();
    let transported = if sum_sq > 1e-20 {
        // Minimal rotation sending t_prev -> t_next via reflection across (t_prev + t_next) then t_next
        let u_mid = u_prev - sum * (2.0 * u_prev.dot(&sum) / sum_sq);
        -u_mid + t_next * (2.0 * u_mid.dot(t_next))
    } else {
        // Near-antipodal fallback: project onto orthogonal plane
        choose_initial_normal(t_next)
    };
    let ortho = transported - t_next * transported.dot(t_next);
    ortho.normalize()
}

/// Computes closed-loop rotation-minimizing Bishop frames into `out` with zero heap allocations.
///
/// Any holonomy angle accumulated around the closed curve is distributed uniformly by arc-index
/// so that `out[n - 1]` transports seamlessly into `out[0]` with zero seam twist.
///
/// # Errors
/// Returns [`KnotError`] if `points.len() < 3`, `out.len() != points.len()`, or any chord is degenerate.
#[verified_engine::verified]
pub fn compute_bishop_frames_in_place(
    points: &[Point3<f64>],
    out: &mut [BishopFrame],
) -> Result<(), KnotError> {
    let n = points.len();
    if n < 3 {
        return Err(KnotError::InsufficientSamples {
            required: 3,
            actual: n,
        });
    }
    if out.len() != n {
        return Err(KnotError::BufferLengthMismatch {
            expected: n,
            actual: out.len(),
        });
    }

    for i in 0..n {
        let prev = points[(i + n - 1) % n];
        let next = points[(i + 1) % n];
        let chord = next - prev;
        let len = chord.norm();
        if !len.is_finite() || len < 1e-12 {
            return Err(KnotError::DegenerateTangent { index: i });
        }
        out[i].tangent = chord / len;
    }

    let u0 = choose_initial_normal(&out[0].tangent);
    out[0].normal_u = u0;
    out[0].normal_v = out[0].tangent.cross(&u0);

    for i in 1..n {
        let t_prev = out[i - 1].tangent;
        let t_curr = out[i].tangent;
        let u_prev = out[i - 1].normal_u;
        let u_curr = parallel_transport_normal(&t_prev, &t_curr, &u_prev);
        out[i].normal_u = u_curr;
        out[i].normal_v = t_curr.cross(&u_curr);
    }

    // Close the Bishop frame around segment (n - 1) -> 0 by cancelling holonomy smoothly
    let u_end_to_0 =
        parallel_transport_normal(&out[n - 1].tangent, &out[0].tangent, &out[n - 1].normal_u);
    let cos_hol = u_end_to_0.dot(&out[0].normal_u).clamp(-1.0, 1.0);
    let sin_hol = u_end_to_0.dot(&out[0].normal_v).clamp(-1.0, 1.0);
    let holonomy = sin_hol.atan2(cos_hol);

    let inv_n = 1.0 / (n as f64);
    for (i, frame) in out.iter_mut().enumerate().skip(1) {
        let angle = -holonomy * (i as f64) * inv_n;
        let (sa, ca) = angle.sin_cos();
        let u_rot = frame.normal_u * ca + frame.normal_v * sa;
        let u_unit = (u_rot - frame.tangent * u_rot.dot(&frame.tangent)).normalize();
        frame.normal_u = u_unit;
        frame.normal_v = frame.tangent.cross(&u_unit);
    }
    Ok(())
}

/// Convenience wrapper allocating and returning closed [`BishopFrame`]s for `points`.
///
/// # Errors
/// Propagates any [`KnotError`] from [`compute_bishop_frames_in_place`].
#[verified_engine::verified]
pub fn compute_bishop_frames(points: &[Point3<f64>]) -> Result<Vec<BishopFrame>, KnotError> {
    let mut frames = vec![BishopFrame::default(); points.len()];
    compute_bishop_frames_in_place(points, &mut frames)?;
    Ok(frames)
}
