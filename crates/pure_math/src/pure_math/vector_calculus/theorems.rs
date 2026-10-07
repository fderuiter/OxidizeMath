use super::coordinates::OrthogonalCoordinateSystem;
use super::operators::divergence;
use nalgebra::Vector3;

/// A rectangular domain in the coordinate space defined by min and max bounds.
pub struct Domain {
    #[allow(missing_docs)]
    pub min: Vector3<f64>,
    #[allow(missing_docs)]
    pub max: Vector3<f64>,
}

/// Verifies the Divergence Theorem: $\int_V (\nabla \cdot \mathbf{A}) dV = \oint_S (\mathbf{A} \cdot \mathbf{n}) dS$.
/// Returns (lhs, rhs, diff).
#[verified_engine::verified]
pub fn verify_divergence_theorem<S, F>(
    coords: &S,
    domain: &Domain,
    field: F,
    steps: usize,
) -> (f64, f64, f64)
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&Vector3<f64>) -> Vector3<f64> + Copy,
{
    debug_assert!(steps > 0);
    let lhs = integrate_volume(coords, domain, steps, |p| divergence(coords, field, p));

    let rhs = integrate_surface_flux(coords, domain, steps, field);

    (lhs, rhs, (lhs - rhs).abs())
}

#[verified_engine::verified]
fn integrate_volume<S, F>(coords: &S, domain: &Domain, steps: usize, func: F) -> f64
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&Vector3<f64>) -> f64,
{
    debug_assert!(steps > 0);
    let d = (domain.max - domain.min) / (steps as f64);
    let mut sum = 0.0;

    for i in 0..steps {
        for j in 0..steps {
            for k in 0..steps {
                let u1 = domain.min[0] + (i as f64 + 0.5) * d[0];
                let u2 = domain.min[1] + (j as f64 + 0.5) * d[1];
                let u3 = domain.min[2] + (k as f64 + 0.5) * d[2];
                let p = Vector3::new(u1, u2, u3);

                let h = coords.scale_factors(&p);
                let volume_element = h[0] * h[1] * h[2] * d[0] * d[1] * d[2];

                sum += func(&p) * volume_element;
            }
        }
    }
    sum
}

#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
#[verified_engine::verified]
fn integrate_surface_flux<S, F>(coords: &S, domain: &Domain, steps: usize, field: F) -> f64
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&Vector3<f64>) -> Vector3<f64>,
{
    debug_assert!(steps > 0);
    let d = (domain.max - domain.min) / (steps as f64);
    let mut flux = 0.0;

    // Face u1 = min (Normal -e1)
    for j in 0..steps {
        for k in 0..steps {
            let u1 = domain.min[0];
            let u2 = domain.min[1] + (j as f64 + 0.5) * d[1];
            let u3 = domain.min[2] + (k as f64 + 0.5) * d[2];
            let p = Vector3::new(u1, u2, u3);
            let h = coords.scale_factors(&p);
            let area = h[1] * h[2] * d[1] * d[2];

            // A dot n. n is -e1. Field components are local. So A_1.
            // Flux = A . (-e1) * area = -A_1 * area.
            flux -= field(&p)[0] * area;
        }
    }
    // Face u1 = max (Normal +e1)
    for j in 0..steps {
        for k in 0..steps {
            let u1 = domain.max[0];
            let u2 = domain.min[1] + (j as f64 + 0.5) * d[1];
            let u3 = domain.min[2] + (k as f64 + 0.5) * d[2];
            let p = Vector3::new(u1, u2, u3);
            let h = coords.scale_factors(&p);
            let area = h[1] * h[2] * d[1] * d[2];

            flux += field(&p)[0] * area;
        }
    }

    // Face u2 = min (Normal -e2)
    for i in 0..steps {
        for k in 0..steps {
            let u1 = domain.min[0] + (i as f64 + 0.5) * d[0];
            let u2 = domain.min[1];
            let u3 = domain.min[2] + (k as f64 + 0.5) * d[2];
            let p = Vector3::new(u1, u2, u3);
            let h = coords.scale_factors(&p);
            let area = h[0] * h[2] * d[0] * d[2];

            flux -= field(&p)[1] * area;
        }
    }
    // Face u2 = max (Normal +e2)
    for i in 0..steps {
        for k in 0..steps {
            let u1 = domain.min[0] + (i as f64 + 0.5) * d[0];
            let u2 = domain.max[1];
            let u3 = domain.min[2] + (k as f64 + 0.5) * d[2];
            let p = Vector3::new(u1, u2, u3);
            let h = coords.scale_factors(&p);
            let area = h[0] * h[2] * d[0] * d[2];

            flux += field(&p)[1] * area;
        }
    }

    // Face u3 = min (Normal -e3)
    for i in 0..steps {
        for j in 0..steps {
            let u1 = domain.min[0] + (i as f64 + 0.5) * d[0];
            let u2 = domain.min[1] + (j as f64 + 0.5) * d[1];
            let u3 = domain.min[2];
            let p = Vector3::new(u1, u2, u3);
            let h = coords.scale_factors(&p);
            let area = h[0] * h[1] * d[0] * d[1];

            flux -= field(&p)[2] * area;
        }
    }
    // Face u3 = max (Normal +e3)
    for i in 0..steps {
        for j in 0..steps {
            let u1 = domain.min[0] + (i as f64 + 0.5) * d[0];
            let u2 = domain.min[1] + (j as f64 + 0.5) * d[1];
            let u3 = domain.max[2];
            let p = Vector3::new(u1, u2, u3);
            let h = coords.scale_factors(&p);
            let area = h[0] * h[1] * d[0] * d[1];

            flux += field(&p)[2] * area;
        }
    }

    flux
}
