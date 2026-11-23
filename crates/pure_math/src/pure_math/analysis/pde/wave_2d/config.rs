//! Configuration enums, parameter structs, and analytic reference formulas for 2D wave & plate PDEs.

// [cite:advanced_linear_algebra]

use std::f64::consts::PI;

/// Maximum Courant-Friedrichs-Lewy number enforced on 2D wave steps.
pub const MAX_COURANT_NUMBER: f64 = 0.65;

/// Maximum biharmonic Fourier-Courant stability parameter for explicit Verlet plate substeps.
pub const MAX_BIHARMONIC_NUMBER: f64 = 0.18;

/// Hard upper bound on substeps per macro step to preserve deterministic frame budgets.
pub const MAX_VERLET_SUBSTEPS: usize = 64;

/// Governing 2D partial differential equation medium.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Wave2DMedium {
    /// Second-order hyperbolic wave equation $u_{tt} + \gamma u_t = c(x,y)^2 \nabla^2 u + F$.
    AcousticWave,
    /// Fourth-order Kirchhoff-Love thin plate equation $u_{tt} + \gamma u_t = -D \nabla^4 u + c^2 \nabla^2 u + F$.
    BiharmonicPlate,
}

/// Perimeter boundary condition applied on the rectangular grid boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BoundaryMode2D {
    /// Fixed zero displacement ($u = 0$) and zero boundary moment ($\nabla^2 u = 0$).
    Dirichlet,
    /// Clamped plate boundary ($u = 0$ and $\partial_n u = 0$ on all four edges).
    Clamped,
    /// Free / zero-normal-derivative boundary ($\partial_n u = 0$).
    Neumann,
    /// Absorbing perimeter sponge layer with fixed outer edge.
    Sponge,
}

/// Harmonic point or line excitation source configuration.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DriverConfig2D {
    /// Whether continuous harmonic forcing is active.
    pub enabled: bool,
    /// If `true`, drives a vertical plane-wave line at `x_norm`; otherwise drives a localized spot at `(x_norm, y_norm)`.
    pub line_source: bool,
    /// Normalized X position in `[0.0, 1.0]`.
    pub x_norm: f64,
    /// Normalized Y position in `[0.0, 1.0]`.
    pub y_norm: f64,
    /// Driving frequency $f$ in Hz (cycles per simulation time unit).
    pub frequency: f64,
    /// Peak forcing acceleration amplitude.
    pub amplitude: f64,
    /// Spatial Gaussian radius in grid cells for localized point forcing.
    pub radius_cells: f64,
}

impl Default for DriverConfig2D {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            enabled: true,
            line_source: false,
            x_norm: 0.37,
            y_norm: 0.29,
            frequency: 12.0,
            amplitude: 40.0,
            radius_cells: 2.2,
        }
    }
}

/// Physical parameters governing a 2D wave or biharmonic plate step.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Wave2DParams {
    /// Active PDE medium (`AcousticWave` or `BiharmonicPlate`).
    pub medium: Wave2DMedium,
    /// Perimeter boundary condition.
    pub boundary: BoundaryMode2D,
    /// Reference wave speed $c_0 \ge 0$.
    pub wave_speed: f64,
    /// Biharmonic flexural stiffness $D_{\text{plate}} \ge 0$.
    pub plate_stiffness: f64,
    /// Uniform viscous damping coefficient $\gamma \ge 0$.
    pub damping: f64,
}

impl Default for Wave2DParams {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            medium: Wave2DMedium::BiharmonicPlate,
            boundary: BoundaryMode2D::Neumann,
            wave_speed: 0.0,
            plate_stiffness: 1.0e-4,
            damping: 0.15,
        }
    }
}

/// Analytic angular eigenfrequency $\omega_{m,n}$ (rad/s) of a square clamped Kirchhoff-Love plate
/// of side length `length` and flexural stiffness `plate_stiffness` ($D / (\rho h)$),
/// tabulated from Leissa (1969, NASA SP-160 Table 4.1) and Warburton's Rayleigh-Ritz beam formula.
#[verified_engine::verified]
pub fn analytic_clamped_square_omega(
    mode_m: usize,
    mode_n: usize,
    length: f64,
    plate_stiffness: f64,
) -> f64 {
    assert!(mode_m >= 1 && mode_n >= 1, "Mode indices must be >= 1");
    assert!(
        length > 0.0 && plate_stiffness >= 0.0,
        "Invalid plate geometry"
    );
    let pair = if mode_m <= mode_n {
        (mode_m, mode_n)
    } else {
        (mode_n, mode_m)
    };
    let lambda_sq = match pair {
        (1, 1) => 35.9852,
        (1, 2) => 73.3938,
        (2, 2) => 108.2165,
        (1, 3) => 131.5808,
        (2, 3) => 165.0004,
        (3, 3) => 220.06,
        _ => {
            let bm = (mode_m as f64 + 0.5) * PI;
            let bn = (mode_n as f64 + 0.5) * PI;
            (bm * bm * bm * bm
                + bn * bn * bn * bn
                + 2.0 * (bm * bm - 2.0 * bm) * (bn * bn - 2.0 * bn))
                .sqrt()
        }
    };
    plate_stiffness.sqrt() * lambda_sq / (length * length)
}

/// Analytic Young double-slit first-order bright fringe spacing $y_1$ satisfying
/// $\sqrt{D^2 + (y_1 + d/2)^2} - \sqrt{D^2 + (y_1 - d/2)^2} = \lambda$
/// (which asymptotes to $\lambda D / d$ for $D \gg d \gg \lambda$).
#[verified_engine::verified]
pub fn young_double_slit_fringe_spacing(
    wavelength: f64,
    slit_separation: f64,
    screen_distance: f64,
) -> f64 {
    assert!(
        wavelength > 0.0 && slit_separation > 0.0 && screen_distance > 0.0,
        "Young interference parameters must be strictly positive"
    );
    let diff_sq = slit_separation * slit_separation - wavelength * wavelength;
    if diff_sq > 1.0e-9 {
        let numer = 4.0 * screen_distance * screen_distance + diff_sq;
        0.5 * wavelength * (numer / diff_sq).sqrt()
    } else {
        wavelength * screen_distance / slit_separation
    }
}
