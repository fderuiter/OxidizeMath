//! Aerodynamic geometry stamps, vorticity curl diagnostics, and boundary-force
//! integration for 2D D2Q9 Lattice Boltzmann wind-tunnel simulations.

use super::model::{BgkCollision, CollisionModel, D2Q9, Lattice2D, LatticeBoltzmannD2Q9};

/// Maximum lattice speed permitted for interactive velocity-jet injection to
/// preserve low-Mach BGK compressibility and numerical stability.
pub const MAX_LATTICE_JET_SPEED: f64 = 0.16;

/// Integrated 2D aerodynamic force and dimensionless lift/drag coefficients.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AerodynamicCoefficients {
    /// Dimensionless drag coefficient $C_D = F_x / (\frac{1}{2} \rho_0 U_\infty^2 L)$.
    pub drag_coeff: f64,
    /// Dimensionless lift coefficient $C_L = F_y / (\frac{1}{2} \rho_0 U_\infty^2 L)$.
    pub lift_coeff: f64,
    /// Raw horizontal fluid-to-obstacle momentum-exchange force $F_x$ in lattice units.
    pub force_x: f64,
    /// Raw vertical fluid-to-obstacle momentum-exchange force $F_y$ in lattice units.
    pub force_y: f64,
}

/// Parametric NACA 4-digit airfoil section placed on a 2D lattice grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Naca4Airfoil {
    /// Four NACA digits `[m, p, t_tens, t_ones]` (for example `[2, 4, 1, 2]` for NACA 2412).
    pub digits: [u8; 4],
    /// Chord length $c$ in lattice cells.
    pub chord: f64,
    /// Quarter-chord pivot X coordinate in solver grid coordinates.
    pub center_x: f64,
    /// Quarter-chord pivot Y coordinate in solver grid coordinates.
    pub center_y: f64,
    /// Geometric angle of attack $\alpha$ in degrees (positive pitches nose up into $+x$ flow).
    pub angle_of_attack_deg: f64,
}

impl Naca4Airfoil {
    /// Constructs a NACA 4-digit airfoil from an integer code such as `12` (`0012`), `2412`, or `4412`.
    #[verified_engine::verified]
    pub fn from_code(
        code: u16,
        chord: f64,
        center_x: f64,
        center_y: f64,
        angle_of_attack_deg: f64,
    ) -> Self {
        let clamped = code.min(9999);
        let d0 = ((clamped / 1000) % 10) as u8;
        let d1 = ((clamped / 100) % 10) as u8;
        let d2 = ((clamped / 10) % 10) as u8;
        let d3 = (clamped % 10) as u8;
        Self {
            digits: [d0, d1, d2, d3],
            chord: chord.max(4.0),
            center_x,
            center_y,
            angle_of_attack_deg,
        }
    }

    /// Evaluates whether a lattice point `(x, y)` lies inside the rotated NACA 4-digit profile.
    #[verified_engine::verified]
    pub fn contains_point(&self, x: f64, y: f64) -> bool {
        let m = f64::from(self.digits[0]) * 0.01;
        let p = f64::from(self.digits[1]) * 0.1;
        let t = (f64::from(self.digits[2]) * 10.0 + f64::from(self.digits[3])) * 0.01;
        let thickness = t.clamp(0.04, 0.40);

        // Positive AoA pitches the leading edge toward -y or +y? In standard wind-tunnel
        // coordinates (+x right, +y up), positive AoA rotates the nose down or trailing edge down:
        // Nose up (+y) relative to +x freestream corresponds to chord line tilted by -alpha
        // (leading edge at higher y, trailing edge at lower y).
        let alpha = self.angle_of_attack_deg.to_radians();
        let cos_a = alpha.cos();
        let sin_a = alpha.sin();

        let dx = x - self.center_x;
        let dy = y - self.center_y;

        // Rotate into chord-aligned frame anchored at quarter-chord (s = 0.25).
        let x_local = dx * cos_a - dy * sin_a + 0.25 * self.chord;
        let y_local = dx * sin_a + dy * cos_a;

        let s = x_local / self.chord;
        if !(0.0..=1.0).contains(&s) {
            return false;
        }

        let (yc, dyc_ds) = if m > 0.0 && p > 0.0 && p < 1.0 {
            if s < p {
                let yc_val = (m / (p * p)) * (2.0 * p * s - s * s);
                let slope = (2.0 * m / (p * p)) * (p - s);
                (yc_val, slope)
            } else {
                let denom = (1.0 - p) * (1.0 - p);
                let yc_val = (m / denom) * ((1.0 - 2.0 * p) + 2.0 * p * s - s * s);
                let slope = (2.0 * m / denom) * (p - s);
                (yc_val, slope)
            }
        } else {
            (0.0, 0.0)
        };

        let yt = 5.0
            * thickness
            * (0.2969 * s.sqrt() - 0.1260 * s - 0.3516 * s.powi(2) + 0.2843 * s.powi(3)
                - 0.1015 * s.powi(4));

        let camber_y = yc * self.chord;
        let normal_cos = 1.0 / (1.0 + dyc_ds * dyc_ds).sqrt();
        let half_thick = (yt * self.chord * normal_cos).max(0.55);

        (y_local - camber_y).abs() <= half_thick
    }
}

/// Stamps a solid NACA 4-digit airfoil onto the lattice obstacle mask.
#[verified_engine::verified]
pub fn stamp_naca4_airfoil<C: CollisionModel<9, D2Q9>>(
    solver: &mut LatticeBoltzmannD2Q9<C>,
    airfoil: &Naca4Airfoil,
) {
    let width = solver.width();
    let height = solver.height();
    for y in 0..height {
        for x in 0..width {
            if airfoil.contains_point(x as f64 + 0.5, y as f64 + 0.5) {
                solver.set_obstacle(x, y, true);
            }
        }
    }
}

/// Stamps a symmetric converging-diverging Venturi nozzle along the top and bottom channel walls.
///
/// `throat_ratio` specifies the open throat height as a fraction of total channel height
/// (clamped to `[0.25, 0.85]`).
#[verified_engine::verified]
pub fn stamp_venturi_nozzle<C: CollisionModel<9, D2Q9>>(
    solver: &mut LatticeBoltzmannD2Q9<C>,
    throat_ratio: f64,
) {
    let width = solver.width();
    let height = solver.height();
    let clamped_ratio = throat_ratio.clamp(0.25, 0.85);
    let max_wall = 0.5 * (1.0 - clamped_ratio) * height as f64;
    let x_throat = 0.45 * width as f64;
    let sigma = (0.18 * width as f64).max(2.0);

    for x in 4..width.saturating_sub(2) {
        let z = (x as f64 - x_throat) / sigma;
        let wall_thickness = (1.0 + (max_wall - 1.0) * (-0.5 * z * z).exp()).round() as usize;
        for y in 0..height {
            if y < wall_thickness || y >= height.saturating_sub(wall_thickness) {
                solver.set_obstacle(x, y, true);
            }
        }
    }
}

/// Injects a bounded macroscopic momentum jet into non-obstacle lattice cells within a disk.
#[verified_engine::verified]
pub fn inject_velocity_jet<C: CollisionModel<9, D2Q9>>(
    solver: &mut LatticeBoltzmannD2Q9<C>,
    cx: i32,
    cy: i32,
    radius: i32,
    jet_ux: f64,
    jet_uy: f64,
) {
    let speed = jet_ux.hypot(jet_uy);
    let (clamped_ux, clamped_uy) = if speed > MAX_LATTICE_JET_SPEED {
        let scale = MAX_LATTICE_JET_SPEED / speed;
        (jet_ux * scale, jet_uy * scale)
    } else if speed <= 1e-12 {
        (0.12, 0.0)
    } else {
        (jet_ux, jet_uy)
    };

    let r = radius.max(1);
    let r_sq = r * r;
    let w = solver.width() as i32;
    let h = solver.height() as i32;

    for y in (cy - r)..=(cy + r) {
        for x in (cx - r)..=(cx + r) {
            if x >= 0 && x < w && y >= 0 && y < h && (x - cx).pow(2) + (y - cy).pow(2) <= r_sq {
                let ux_idx = (y as usize, x as usize);
                if !solver.state.obstacles[ux_idx] {
                    let rho = solver.state.rho[ux_idx].clamp(0.85, 1.15);
                    solver.state.ux[ux_idx] = clamped_ux;
                    solver.state.uy[ux_idx] = clamped_uy;
                    let eq = D2Q9::equilibrium(rho, clamped_ux, clamped_uy);
                    solver.state.f[ux_idx] = eq;
                    solver.state.f_new[ux_idx] = eq;
                }
            }
        }
    }
}

/// Applies a zero-gradient convective outlet condition on the right boundary (`x = width - 1`)
/// so shed vortices exit cleanly instead of wrapping periodically into the inlet.
#[verified_engine::verified]
pub fn apply_outflow_right_boundary<C: CollisionModel<9, D2Q9>>(
    solver: &mut LatticeBoltzmannD2Q9<C>,
) {
    let width = solver.width();
    let height = solver.height();
    if width < 3 {
        return;
    }
    let src_x = width - 2;
    let dst_x = width - 1;
    for y in 0..height {
        if !solver.state.obstacles[(y, dst_x)] && !solver.state.obstacles[(y, src_x)] {
            solver.state.f[(y, dst_x)] = solver.state.f[(y, src_x)];
            solver.state.rho[(y, dst_x)] = solver.state.rho[(y, src_x)];
            solver.state.ux[(y, dst_x)] = solver.state.ux[(y, src_x)];
            solver.state.uy[(y, dst_x)] = solver.state.uy[(y, src_x)];
        }
    }
}

/// Advances one complete wind-tunnel step with zero heap allocations:
/// enforces left-boundary freestream inflow, streams + collides, swaps buffers,
/// and applies a zero-gradient right-boundary outlet.
#[verified_engine::verified]
pub fn advance_wind_tunnel_step(solver: &mut LatticeBoltzmannD2Q9<BgkCollision>, inlet_u: f64) {
    let safe_u = inlet_u.clamp(0.0, MAX_LATTICE_JET_SPEED);
    solver.set_inlet(0, 0, 2, solver.height(), safe_u, 0.0);
    solver.step();
    apply_outflow_right_boundary(solver);
}

/// Computes discrete 2D vorticity $\omega_z = \partial_x u_y - \partial_y u_x$ into
/// `out_vorticity` (row-major `y * width + x`) with zero heap allocations.
#[verified_engine::verified]
pub fn compute_vorticity_into<C: CollisionModel<9, D2Q9>>(
    solver: &LatticeBoltzmannD2Q9<C>,
    out_vorticity: &mut [f64],
) {
    let width = solver.width();
    let height = solver.height();
    let total = width * height;
    assert_eq!(
        out_vorticity.len(),
        total,
        "Vorticity buffer length must equal width * height"
    );
    out_vorticity.fill(0.0);
    if width < 3 || height < 3 {
        return;
    }

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = y * width + x;
            if solver.state.obstacles[(y, x)] {
                continue;
            }
            let duy_dx = 0.5 * (solver.state.uy[(y, x + 1)] - solver.state.uy[(y, x - 1)]);
            let dux_dy = 0.5 * (solver.state.ux[(y + 1, x)] - solver.state.ux[(y - 1, x)]);
            out_vorticity[idx] = duy_dx - dux_dy;
        }
    }
}

/// Integrates hydrodynamic drag and lift on interior solid obstacles via the D2Q9
/// bounce-back momentum-exchange method with zero heap allocations.
///
/// Cells within `y_margin` of the top and bottom domain edges are excluded so channel/nozzle
/// walls near the outer boundaries do not mask body lift.
#[verified_engine::verified]
pub fn compute_aerodynamic_coefficients<C: CollisionModel<9, D2Q9>>(
    solver: &LatticeBoltzmannD2Q9<C>,
    u_inf: f64,
    ref_length: f64,
    y_margin: usize,
) -> AerodynamicCoefficients {
    let width = solver.width();
    let height = solver.height();
    if width < 3 || height <= 2 * y_margin {
        return AerodynamicCoefficients::default();
    }

    let cx = D2Q9::directions_x();
    let cy = D2Q9::directions_y();
    let mut fx = 0.0;
    let mut fy = 0.0;

    let y_start = y_margin.max(1);
    let y_end = height.saturating_sub(y_margin.max(1));

    for y in y_start..y_end {
        for x in 2..(width - 1) {
            if solver.state.obstacles[(y, x)] {
                continue;
            }
            let f_fluid = solver.state.f[(y, x)];
            for k in 1..9 {
                let nb_x = (x as i32 + cx[k]) as usize;
                let nb_y = (y as i32 + cy[k]) as usize;
                if nb_y >= y_start && nb_y < y_end && solver.state.obstacles[(nb_y, nb_x)] {
                    // Bounce-back reflects outgoing momentum c_k * f_k into -c_k * f_k,
                    // transferring 2 * f_k * c_k from fluid to solid across link k.
                    let impulse = 2.0 * f_fluid[k];
                    fx += impulse * f64::from(cx[k]);
                    fy += impulse * f64::from(cy[k]);
                }
            }
        }
    }

    let denom = (0.5 * u_inf * u_inf * ref_length.max(1.0)).max(1e-9);
    AerodynamicCoefficients {
        drag_coeff: fx / denom,
        lift_coeff: fy / denom,
        force_x: fx,
        force_y: fy,
    }
}
