#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::preset::IsingInteractionMode;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::DiagnosticReport;
use eframe::egui;
use math_explorer::applied::game_theory::evolutionary::MatrixPayoff;
use math_explorer::physics::stat_mech::ising::SpinLattice;
use math_explorer::physics::stat_mech::KB;
use nalgebra::DMatrix;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[allow(missing_docs)]
pub struct IsingSharedParams {
    pub temperature: f64,
    pub j_coupling: f64,
    pub h_field: f64,
    pub mode: IsingInteractionMode,
    pub game_payoffs: [f64; 4],
    pub steps_per_frame: usize,
    pub theme: crate::app::ThemeMode,
    pub total_trials: AtomicU64,
}

impl Default for IsingSharedParams {
    fn default() -> Self {
        Self {
            temperature: 2.269,
            j_coupling: 1.0,
            h_field: 0.0,
            mode: IsingInteractionMode::Ferromagnetic,
            game_payoffs: super::preset::default_game_payoffs(),
            steps_per_frame: 5000,
            theme: crate::app::ThemeMode::Dark,
            total_trials: AtomicU64::new(0),
        }
    }
}

/// Maps an Ising lattice spin (`+1` or `-1`) to a WCAG `>= 3.0:1` [`crate::app::SemanticPalette`] color.
#[inline]
#[must_use]
pub fn ising_spin_color(spin: i8, palette: &crate::app::SemanticPalette) -> egui::Color32 {
    if spin > 0 {
        palette.axis_x
    } else {
        palette.axis_z
    }
}

/// Paints a clamped circular droplet onto `lattice` using `SpinLattice::set`.
pub fn apply_circular_droplet(lattice: &mut SpinLattice, cx: i32, cy: i32, r: i32, spin_val: i8) {
    let width = lattice.width();
    let height = lattice.height();
    if width == 0 || height == 0 {
        return;
    }
    let max_x = (width as i32) - 1;
    let max_y = (height as i32) - 1;
    let cx = cx.clamp(0, max_x);
    let cy = cy.clamp(0, max_y);
    let r = r.clamp(0, max_x.max(max_y));
    let r_sq = i64::from(r) * i64::from(r);

    let y_min = (cy - r).max(0);
    let y_max = (cy + r).min(max_y);
    let x_min = (cx - r).max(0);
    let x_max = (cx + r).min(max_x);

    for y in y_min..=y_max {
        let dy = i64::from(y - cy);
        for x in x_min..=x_max {
            let dx = i64::from(x - cx);
            if dx * dx + dy * dy <= r_sq {
                lattice.set(x as usize, y as usize, spin_val);
            }
        }
    }
}

#[allow(missing_docs)]
pub struct IsingRunner {
    pub lattice: SpinLattice,
    pub params: Arc<std::sync::RwLock<IsingSharedParams>>,
}

impl IsingRunner {
    fn evolve_trials(&mut self, num_trials: usize) {
        let (mode, temp, j, h, payoffs) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.mode, p.temperature, p.j_coupling, p.h_field, p.game_payoffs)
        };
        if !temp.is_finite() {
            return;
        }
        match mode {
            IsingInteractionMode::Ferromagnetic => {
                if !j.is_finite() || !h.is_finite() {
                    return;
                }
                let j_scale = if j.abs() < 1e-6 { 1.0 } else { j.abs() };
                let t_phys = temp.max(1e-3) * j_scale / KB;
                self.lattice.evolve(num_trials, t_phys, j, h);
            }
            IsingInteractionMode::FermiGame => {
                if !payoffs.iter().all(|v| v.is_finite()) {
                    return;
                }
                let mat = DMatrix::from_row_slice(2, 2, &payoffs);
                if let Ok(payoff) = MatrixPayoff::new(mat) {
                    self.lattice.evolve_fermi_game(num_trials, temp, &payoff);
                }
            }
        }
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        p.total_trials.fetch_add(num_trials as u64, Ordering::Relaxed);
    }
}

impl SimulationRunner for IsingRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => {
                let w = self.lattice.width();
                let h = self.lattice.height();
                self.lattice = SpinLattice::new(w, h, None);
                self.params
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .total_trials
                    .store(0, Ordering::Relaxed);
            }
            SimCommand::SetSpeed(speed) => {
                self.params
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .steps_per_frame = speed;
            }
            SimCommand::ApplyBrush {
                cx,
                cy,
                r,
                is_obstacle,
            } => {
                let spin_val = if is_obstacle { 1 } else { -1 };
                apply_circular_droplet(&mut self.lattice, cx, cy, r, spin_val);
            }
            SimCommand::ClearObstacles => {
                for y in 0..self.lattice.height() {
                    for x in 0..self.lattice.width() {
                        self.lattice.set(x, y, -1);
                    }
                }
                self.params
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .total_trials
                    .store(0, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        let steps = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .steps_per_frame;
        self.evolve_trials(steps);
    }

    fn step_once(&mut self) {
        let one_sweep = self.lattice.width() * self.lattice.height();
        self.evolve_trials(one_sweep);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let width = self.lattice.width();
        let height = self.lattice.height();
        let (j, h, theme) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.j_coupling, p.h_field, p.theme)
        };
        let palette = theme.palette();
        let spins_vec: Vec<i8> = self.lattice.spins().to_vec();
        let mut pixels = Vec::with_capacity(width * height);
        for &s in &spins_vec {
            pixels.push(ising_spin_color(s, &palette));
        }
        let m = self.lattice.magnetization();
        let max_m = (width * height) as f64;
        let m_avg = m as f64 / max_m;
        let energy = self.lattice.hamiltonian(j, h);

        StateSnapshot {
            width,
            height,
            pixels: Arc::new(std::sync::RwLock::new(pixels)),
            custom_data: vec![m_avg, energy],
            structured_data: Some(Box::new(spins_vec)),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_finite_state(&self) -> bool {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        p.temperature.is_finite()
            && p.j_coupling.is_finite()
            && p.h_field.is_finite()
            && p.game_payoffs.iter().all(|v| v.is_finite())
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        if self.is_finite_state() {
            None
        } else {
            Some(DiagnosticReport::new(
                0.0,
                0,
                "ising_parameters",
                "Non-finite Ising model parameter encountered",
            ))
        }
    }
}
