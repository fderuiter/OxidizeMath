#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use std::f64::consts::{PI, TAU};

use eframe::egui;
use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::analysis::pde::wave_2d::masks::clamped_beam_eigenfunction;
use math_explorer::pure_math::analysis::pde::wave_2d::{
    BoundaryMode2D, Wave2DMedium, Wave2DSolver, analytic_clamped_square_omega,
};

use super::{ChladniWaveTankState, ChladniWaveTankTool, FREQ_MAX, FREQ_MIN, GRID_SIZE, TankPreset};

const DRUM_RADIUS_NORM: f64 = 0.45;

/// Roots $j_{m, n}$ of $J_m(x) = 0$ for azimuthal order $m \in 0..6$ and radial root $n \in 1..=6$.
const BESSEL_J_ZEROS: [[f64; 6]; 6] = [
    [
        2.404_825_557_7,
        5.520_078_110_3,
        8.653_727_912_9,
        11.791_534_439_0,
        14.930_917_708_5,
        18.071_063_967_9,
    ],
    [
        3.831_705_970_2,
        7.015_586_669_8,
        10.173_468_135_1,
        13.323_691_936_3,
        16.470_630_050_9,
        19.615_858_510_5,
    ],
    [
        5.135_622_301_8,
        8.417_244_140_4,
        11.619_841_172_1,
        14.795_951_782_4,
        17.959_819_495_0,
        21.116_997_053_0,
    ],
    [
        6.380_161_895_9,
        9.761_023_130_0,
        13.015_200_721_7,
        16.223_466_160_3,
        19.409_415_226_4,
        22.582_729_593_1,
    ],
    [
        7.588_342_434_5,
        11.064_709_488_5,
        14.372_536_671_6,
        17.615_966_049_8,
        20.826_932_957_0,
        24.019_019_524_8,
    ],
    [
        8.771_483_816_0,
        12.338_604_197_2,
        15.700_174_079_7,
        18.980_133_875_2,
        22.217_799_896_6,
        25.430_340_702_2,
    ],
];

pub(crate) fn default_mode_for_preset(preset: TankPreset) -> (usize, usize) {
    match preset {
        TankPreset::SquareChladniPlate => (1, 3),
        TankPreset::CircularDrumhead => (2, 2),
        TankPreset::DoubleSlitInterference | TankPreset::AcousticLens => (1, 1),
    }
}

pub(crate) fn bessel_j_zero(m: usize, n: usize) -> f64 {
    let m_idx = m.min(BESSEL_J_ZEROS.len() - 1);
    let n_idx = n.saturating_sub(1).min(BESSEL_J_ZEROS[0].len() - 1);
    BESSEL_J_ZEROS[m_idx][n_idx]
}

/// Evaluates $J_m(x) = \frac{1}{\pi}\int_0^\pi \cos(m\tau - x\sin\tau)\,d\tau$ via 48-point midpoint quadrature.
fn bessel_j(m: usize, x: f64) -> f64 {
    const STEPS: usize = 48;
    let d_tau = PI / (STEPS as f64);
    let m_f = m as f64;
    let mut sum = 0.0_f64;
    for k in 0..STEPS {
        let tau = ((k as f64) + 0.5) * d_tau;
        sum += (m_f * tau - x * tau.sin()).cos();
    }
    sum / (STEPS as f64)
}

fn domain_length(dx: f64) -> f64 {
    (((GRID_SIZE - 1) as f64) * dx).max(1.0e-3)
}

pub(crate) fn lattice_axes(state: &ChladniWaveTankState) -> ([usize; 6], [usize; 6]) {
    if state.preset == TankPreset::CircularDrumhead {
        ([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
    } else {
        ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
    }
}

pub(crate) fn normalized_mode(
    state: &ChladniWaveTankState,
    mode: (usize, usize),
) -> (usize, usize) {
    if state.preset == TankPreset::CircularDrumhead {
        (mode.0.min(5), mode.1.clamp(1, 6))
    } else {
        (mode.0.clamp(1, 6), mode.1.clamp(1, 6))
    }
}

pub(crate) fn mode_metric_label(state: &ChladniWaveTankState, m: usize, n: usize) -> String {
    let (mc, nc) = normalized_mode(state, (m, n));
    if state.preset == TankPreset::CircularDrumhead {
        format!("Bessel f({mc},{nc})")
    } else {
        match state.boundary {
            BoundaryMode2D::Clamped => format!("Clamped f({mc},{nc})"),
            BoundaryMode2D::Neumann => format!("Neumann f({mc},{nc})"),
            BoundaryMode2D::Dirichlet => format!("Dirichlet f({mc},{nc})"),
            BoundaryMode2D::Sponge => format!("Sponge f({mc},{nc})"),
        }
    }
}

pub(crate) fn mode_frequency_hz(state: &ChladniWaveTankState, dx: f64, m: usize, n: usize) -> f64 {
    let (mc, nc) = normalized_mode(state, (m, n));
    let len = domain_length(dx);
    if state.preset == TankPreset::CircularDrumhead {
        let radius = (DRUM_RADIUS_NORM * len).max(1.0e-4);
        let j_mn = bessel_j_zero(mc, nc);
        if state.medium == Wave2DMedium::BiharmonicPlate {
            let d = state.plate_stiffness.max(1.0e-9);
            d.sqrt() * (j_mn / radius).powi(2) / TAU
        } else {
            state.wave_speed.max(1.0e-4) * j_mn / (TAU * radius)
        }
    } else if state.medium == Wave2DMedium::BiharmonicPlate {
        let d = state.plate_stiffness.max(1.0e-9);
        if state.boundary == BoundaryMode2D::Clamped {
            analytic_clamped_square_omega(mc, nc, len, d) / TAU
        } else {
            let modal_sum = (mc * mc + nc * nc) as f64;
            d.sqrt() * PI * modal_sum / (2.0 * len * len)
        }
    } else {
        let modal_hypot = ((mc * mc + nc * nc) as f64).sqrt();
        state.wave_speed.max(1.0e-4) * modal_hypot / (2.0 * len)
    }
}

pub(crate) fn eigenmode_sample(
    state: &ChladniWaveTankState,
    m: usize,
    n: usize,
    xn: f64,
    yn: f64,
) -> f64 {
    let (mc, nc) = normalized_mode(state, (m, n));
    if state.preset == TankPreset::CircularDrumhead {
        let (dx, dy) = (xn - 0.5, yn - 0.5);
        let r_norm = (dx * dx + dy * dy).sqrt() / DRUM_RADIUS_NORM;
        if r_norm >= 1.0 {
            return 0.0;
        }
        let j_mn = bessel_j_zero(mc, nc);
        let theta = dy.atan2(dx);
        return 0.55 * bessel_j(mc, j_mn * r_norm) * ((mc as f64) * theta).cos();
    }
    match state.boundary {
        BoundaryMode2D::Clamped => clamped_square_sample(mc, nc, xn, yn),
        BoundaryMode2D::Neumann => trig_square_sample(mc, nc, xn, yn, true),
        BoundaryMode2D::Dirichlet | BoundaryMode2D::Sponge => {
            trig_square_sample(mc, nc, xn, yn, false)
        }
    }
}

fn clamped_square_sample(m: usize, n: usize, xn: f64, yn: f64) -> f64 {
    let xm = clamped_beam_eigenfunction(m, xn);
    let ym = clamped_beam_eigenfunction(m, yn);
    if m == n {
        return 0.30 * xm * ym;
    }
    let xn_mode = clamped_beam_eigenfunction(n, xn);
    let yn_mode = clamped_beam_eigenfunction(n, yn);
    if m < n {
        0.24 * (xn_mode * ym - xm * yn_mode)
    } else {
        0.24 * (xm * yn_mode + xn_mode * ym)
    }
}

fn trig_square_sample(m: usize, n: usize, xn: f64, yn: f64, use_cos: bool) -> f64 {
    let f = |k: usize, u: f64| -> f64 {
        let arg = (k as f64) * PI * u;
        if use_cos { arg.cos() } else { arg.sin() }
    };
    if m == n {
        0.45 * f(m, xn) * f(n, yn)
    } else if m < n {
        0.45 * (f(n, xn) * f(m, yn) - f(m, xn) * f(n, yn))
    } else {
        0.45 * (f(m, xn) * f(n, yn) + f(n, xn) * f(m, yn))
    }
}

pub(crate) fn seed_eigenmode_field(
    solver: &mut Wave2DSolver,
    state: &ChladniWaveTankState,
    m: usize,
    n: usize,
) {
    if state.line_source {
        return;
    }
    let (w, h) = (solver.width(), solver.height());
    let (denom_x, denom_y) = ((w - 1).max(1) as f64, (h - 1).max(1) as f64);
    let mut init = vec![0.0; w * h];
    for y in 0..h {
        let yn = (y as f64) / denom_y;
        for x in 0..w {
            let idx = flatten_2d_index(x, y, w);
            if solver.wall_mask[idx] > 0.5 {
                init[idx] = eigenmode_sample(state, m, n, (x as f64) / denom_x, yn);
            }
        }
    }
    if init.iter().any(|&v| v.abs() > 1.0e-9) {
        solver.set_stationary_displacement(&init);
    } else {
        solver.inject_gaussian_pulse(state.driver_x, state.driver_y, 0.07, 0.45);
    }
}

pub(crate) fn find_antinode_driver_coords(
    solver: &Wave2DSolver,
    state: &ChladniWaveTankState,
    m: usize,
    n: usize,
) -> (f64, f64) {
    let (w, h) = (solver.width(), solver.height());
    let (denom_x, denom_y) = ((w - 1).max(1) as f64, (h - 1).max(1) as f64);
    let margin = 6.min(w / 4);
    let mut best = (0.35_f64, 0.08_f64);
    let mut best_score = -1.0_f64;

    for y in margin..(h.saturating_sub(margin)) {
        let yn = (y as f64) / denom_y;
        for x in margin..(w.saturating_sub(margin)) {
            let idx = flatten_2d_index(x, y, w);
            if solver.wall_mask[idx] <= 0.5
                || solver.wall_mask[flatten_2d_index(x - 2, y, w)] <= 0.5
                || solver.wall_mask[flatten_2d_index(x + 2, y, w)] <= 0.5
                || solver.wall_mask[flatten_2d_index(x, y - 2, w)] <= 0.5
                || solver.wall_mask[flatten_2d_index(x, y + 2, w)] <= 0.5
            {
                continue;
            }
            let xn = (x as f64) / denom_x;
            let amp = eigenmode_sample(state, m, n, xn, yn).abs();
            let tie_break = 1.0 - 0.015 * ((xn - 0.35).abs() + (yn - 0.08).abs());
            let score = amp * tie_break;
            if score > best_score {
                best_score = score;
                best = (xn.clamp(0.04, 0.96), yn.clamp(0.04, 0.96));
            }
        }
    }
    best
}

impl ChladniWaveTankTool {
    pub(crate) fn select_eigenmode(&mut self, m: usize, n: usize) {
        let (mc, nc) = normalized_mode(&self.state, (m, n));
        self.active_mode = (mc, nc);
        self.state.driver_enabled = true;
        self.state.line_source = false;
        self.rebuild_masks();
        let freq =
            mode_frequency_hz(&self.state, self.solver.dx(), mc, nc).clamp(FREQ_MIN, FREQ_MAX);
        self.state.driver_frequency = (freq * 10_000.0).round() / 10_000.0;
        let (nx, ny) = find_antinode_driver_coords(&self.solver, &self.state, mc, nc);
        self.state.driver_x = (nx * 1_000.0).round() / 1_000.0;
        self.state.driver_y = (ny * 1_000.0).round() / 1_000.0;
        self.solver.reset_all();
        self.rebuild_masks();
        seed_eigenmode_field(&mut self.solver, &self.state, mc, nc);
        self.sand.reseed(
            self.state.rng_seed,
            GRID_SIZE,
            GRID_SIZE,
            &self.solver.wall_mask,
        );
        self.state_rev = self.state_rev.wrapping_add(1);
        self.refresh_analysis_outputs();
    }

    pub(super) fn draw_eigenmode_lattice_ui(&mut self, ui: &mut egui::Ui) {
        let (m_vals, n_vals) = lattice_axes(&self.state);
        let active = normalized_mode(&self.state, self.active_mode);
        let axis_hint = if self.state.preset == TankPreset::CircularDrumhead {
            "Bessel Modes"
        } else {
            "Eigenmode Lattice"
        };
        let mut clicked_mode: Option<(usize, usize)> = None;
        egui::CollapsingHeader::new(format!("{axis_hint} ({}, {})", active.0, active.1))
            .id_salt("chladni_eigenmode_lattice")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("chladni_eigenmode_6x6_grid")
                    .spacing([2.0, 2.0])
                    .show(ui, |ui| {
                        for &m in &m_vals {
                            for &n in &n_vals {
                                let hz = mode_frequency_hz(&self.state, self.solver.dx(), m, n);
                                let selected = active == (m, n);
                                let resp = ui
                                    .selectable_label(selected, format!("{m},{n}"))
                                    .on_hover_text(format!(
                                        "{} = {hz:.4} Hz (click to tune & seed antinode)",
                                        mode_metric_label(&self.state, m, n)
                                    ));
                                if resp.clicked() {
                                    clicked_mode = Some((m, n));
                                }
                            }
                            ui.end_row();
                        }
                    });
            });
        if let Some((m, n)) = clicked_mode {
            self.select_eigenmode(m, n);
        }
    }
}
