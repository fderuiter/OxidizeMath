//! Analytical 2D MRI Tissue Phantoms (Modified Shepp-Logan).

use std::f64::consts::PI;

/// MRI relaxation and proton-density parameters at a single spatial voxel.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TissueParameters {
    /// Equilibrium proton spin density $\rho \in [0, 1]$.
    pub rho: f64,
    /// Longitudinal spin-lattice relaxation time $T_1$ in seconds.
    pub t1: f64,
    /// Transverse spin-spin relaxation time $T_2$ in seconds.
    pub t2: f64,
}

/// Analytical ellipse specification for the Modified Shepp-Logan phantom.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SheppLoganEllipse {
    /// Additive grey-level intensity contribution.
    pub additive_intensity: f64,
    /// Horizontal semi-axis $a$ in normalized $[-1, 1]$ FOV units.
    pub semi_axis_x: f64,
    /// Vertical semi-axis $b$ in normalized $[-1, 1]$ FOV units.
    pub semi_axis_y: f64,
    /// Ellipse center $x_0$ in $[-1, 1]$.
    pub center_x: f64,
    /// Ellipse center $y_0$ in $[-1, 1]$.
    pub center_y: f64,
    /// Counter-clockwise rotation angle $\phi$ in degrees.
    pub rotation_deg: f64,
}

/// Standard 10-ellipse Modified Shepp-Logan head phantom (Toft, 1996).
pub const MODIFIED_SHEPP_LOGAN_ELLIPSES: [SheppLoganEllipse; 10] = [
    SheppLoganEllipse {
        additive_intensity: 1.0,
        semi_axis_x: 0.6900,
        semi_axis_y: 0.9200,
        center_x: 0.0,
        center_y: 0.0,
        rotation_deg: 0.0,
    },
    SheppLoganEllipse {
        additive_intensity: -0.8,
        semi_axis_x: 0.6624,
        semi_axis_y: 0.8740,
        center_x: 0.0,
        center_y: -0.0184,
        rotation_deg: 0.0,
    },
    SheppLoganEllipse {
        additive_intensity: -0.2,
        semi_axis_x: 0.1100,
        semi_axis_y: 0.3100,
        center_x: 0.22,
        center_y: 0.0,
        rotation_deg: -18.0,
    },
    SheppLoganEllipse {
        additive_intensity: -0.2,
        semi_axis_x: 0.1600,
        semi_axis_y: 0.4100,
        center_x: -0.22,
        center_y: 0.0,
        rotation_deg: 18.0,
    },
    SheppLoganEllipse {
        additive_intensity: 0.1,
        semi_axis_x: 0.2100,
        semi_axis_y: 0.2500,
        center_x: 0.0,
        center_y: 0.35,
        rotation_deg: 0.0,
    },
    SheppLoganEllipse {
        additive_intensity: 0.1,
        semi_axis_x: 0.0460,
        semi_axis_y: 0.0460,
        center_x: 0.0,
        center_y: 0.10,
        rotation_deg: 0.0,
    },
    SheppLoganEllipse {
        additive_intensity: 0.1,
        semi_axis_x: 0.0460,
        semi_axis_y: 0.0460,
        center_x: 0.0,
        center_y: -0.10,
        rotation_deg: 0.0,
    },
    SheppLoganEllipse {
        additive_intensity: 0.1,
        semi_axis_x: 0.0460,
        semi_axis_y: 0.0230,
        center_x: -0.08,
        center_y: -0.605,
        rotation_deg: 0.0,
    },
    SheppLoganEllipse {
        additive_intensity: 0.1,
        semi_axis_x: 0.0230,
        semi_axis_y: 0.0230,
        center_x: 0.0,
        center_y: -0.606,
        rotation_deg: 0.0,
    },
    SheppLoganEllipse {
        additive_intensity: 0.1,
        semi_axis_x: 0.0230,
        semi_axis_y: 0.0460,
        center_x: 0.06,
        center_y: -0.605,
        rotation_deg: 0.0,
    },
];

#[verified_engine::verified]
fn point_in_ellipse(x: f64, y: f64, e: &SheppLoganEllipse) -> bool {
    let dx = x - e.center_x;
    let dy = y - e.center_y;
    let phi = e.rotation_deg * (PI / 180.0);
    let cos_p = phi.cos();
    let sin_p = phi.sin();
    let xr = dx * cos_p + dy * sin_p;
    let yr = -dx * sin_p + dy * cos_p;
    let nx = xr / e.semi_axis_x;
    let ny = yr / e.semi_axis_y;
    nx * nx + ny * ny <= 1.0
}

/// Evaluates Modified Shepp-Logan tissue properties $(\rho, T_1, T_2)$ at normalized coordinates $(x, y) \in [-1, 1]^2$.
///
/// # Example
///
/// ```rust
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::mri::phantom::sample_shepp_logan_tissue;
///
/// let center = sample_shepp_logan_tissue(0.0, 0.0);
/// let outside = sample_shepp_logan_tissue(0.95, 0.95);
/// assert!(center.rho > 0.5);
/// assert_eq!(outside.rho, 0.0);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn sample_shepp_logan_tissue(x_norm: f64, y_norm: f64) -> TissueParameters {
    if !point_in_ellipse(x_norm, y_norm, &MODIFIED_SHEPP_LOGAN_ELLIPSES[0]) {
        return TissueParameters {
            rho: 0.0,
            t1: 2.50,
            t2: 0.35,
        };
    }

    let mut intensity = 0.0;
    for ellipse in &MODIFIED_SHEPP_LOGAN_ELLIPSES {
        if point_in_ellipse(x_norm, y_norm, ellipse) {
            intensity += ellipse.additive_intensity;
        }
    }

    if intensity >= 0.85 {
        // Outer skull / scalp rim
        TissueParameters {
            rho: 0.95,
            t1: 0.50,
            t2: 0.055,
        }
    } else if intensity < 0.12 {
        // Lateral ventricles (CSF-like: high proton density, long T1 and T2)
        TissueParameters {
            rho: 0.92,
            t1: 2.40,
            t2: 0.34,
        }
    } else if intensity > 0.25 {
        // Focal soft-tissue inclusions (shorter T1, moderate-short T2)
        let extra = (intensity - 0.25).clamp(0.0, 0.35);
        TissueParameters {
            rho: (0.82 + 0.4 * extra).clamp(0.0, 1.0),
            t1: 0.44,
            t2: 0.062,
        }
    } else {
        // Brain parenchyma baseline
        TissueParameters {
            rho: 0.74,
            t1: 0.95,
            t2: 0.105,
        }
    }
}

/// Fills pre-allocated row-major $(\rho, T_1, T_2)$ slices of length `rows * cols` with the Modified Shepp-Logan phantom.
#[verified_engine::verified]
pub fn fill_shepp_logan_grid(
    rows: usize,
    cols: usize,
    out_rho: &mut [f64],
    out_t1: &mut [f64],
    out_t2: &mut [f64],
) {
    let total = rows * cols;
    assert_eq!(out_rho.len(), total, "out_rho length mismatch");
    assert_eq!(out_t1.len(), total, "out_t1 length mismatch");
    assert_eq!(out_t2.len(), total, "out_t2 length mismatch");
    if total == 0 {
        return;
    }

    let inv_r = if rows > 1 {
        2.0 / ((rows - 1) as f64)
    } else {
        0.0
    };
    let inv_c = if cols > 1 {
        2.0 / ((cols - 1) as f64)
    } else {
        0.0
    };

    for r in 0..rows {
        let y_norm = 1.0 - (r as f64) * inv_r;
        let row_offset = r * cols;
        for c in 0..cols {
            let x_norm = -1.0 + (c as f64) * inv_c;
            let tissue = sample_shepp_logan_tissue(x_norm, y_norm);
            let idx = row_offset + c;
            out_rho[idx] = tissue.rho;
            out_t1[idx] = tissue.t1;
            out_t2[idx] = tissue.t2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_shepp_logan_compartments_and_zero_alloc_fill() {
        let mut rho = vec![0.0; 64 * 64];
        let mut t1 = vec![0.0; 64 * 64];
        let mut t2 = vec![0.0; 64 * 64];

        let ((), _metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            fill_shepp_logan_grid(64, 64, &mut rho, &mut t1, &mut t2);
        });

        assert_eq!(rho[0], 0.0, "Corner outside skull must have zero density");
        let center_idx = 32 * 64 + 32;
        assert!(rho[center_idx] > 0.7, "Interior brain must be non-zero");
        let csf = sample_shepp_logan_tissue(0.22, 0.0);
        let focus = sample_shepp_logan_tissue(0.0, 0.35);
        assert!(csf.t1 > focus.t1 && csf.t2 > focus.t2);
    }
}
