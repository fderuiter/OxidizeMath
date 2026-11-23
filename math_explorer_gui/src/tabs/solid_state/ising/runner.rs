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

/// Maximum persisted circular droplet seeds stored in `IsingState`.
pub const MAX_DROPLET_SEEDS: usize = 32;
/// Deterministic RNG seed used when auto-reseeding absorbed Fermi lattices.
pub const FERMI_AUTO_RESEED: u64 = 0xF321_5EED;

/// Bounded circular spin disk painted onto [`SpinLattice`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DropletSeed {
    /// Horizontal lattice coordinate `0..=99`.
    pub cx: i32,
    /// Vertical lattice coordinate `0..=99`.
    pub cy: i32,
    /// Disk radius `0..=99`.
    pub r: i32,
    /// Spin value (`+1` or `-1`).
    pub spin: i8,
}

#[allow(missing_docs)]
pub struct IsingSharedParams {
    pub temperature: f64,
    pub j_coupling: f64,
    pub h_field: f64,
    pub mutation_rate: f64,
    pub mode: IsingInteractionMode,
    pub game_payoffs: [f64; 4],
    pub defector_sea: bool,
    pub droplet_seeds: Vec<DropletSeed>,
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
            mutation_rate: 0.0,
            mode: IsingInteractionMode::Ferromagnetic,
            game_payoffs: super::preset::default_game_payoffs(),
            defector_sea: false,
            droplet_seeds: Vec::new(),
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
        let (mode, temp, j, h, payoffs, mutation_rate) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (
                p.mode,
                p.temperature,
                p.j_coupling,
                p.h_field,
                p.game_payoffs,
                p.mutation_rate,
            )
        };
        if !temp.is_finite() || !mutation_rate.is_finite() {
            return;
        }
        match mode {
            IsingInteractionMode::Ferromagnetic => {
                if !j.is_finite() || !h.is_finite() {
                    return;
                }
                let t_phys = temp.max(1e-3) / KB;
                self.lattice.evolve(num_trials, t_phys, j, h);
            }
            IsingInteractionMode::FermiGame => {
                if !payoffs.iter().all(|v| v.is_finite()) {
                    return;
                }
                let mat = DMatrix::from_row_slice(2, 2, &payoffs);
                if let Ok(payoff) = MatrixPayoff::new(mat) {
                    self.lattice.evolve_fermi_game_with_mutation(
                        num_trials,
                        temp,
                        &payoff,
                        mutation_rate,
                    );
                }
            }
        }
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        p.total_trials
            .fetch_add(num_trials as u64, Ordering::Relaxed);
    }

    fn apply_shared_seed_layout(&mut self) {
        let (defector_sea, seeds) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            p.total_trials.store(0, Ordering::Relaxed);
            (p.defector_sea, p.droplet_seeds.clone())
        };
        if defector_sea {
            self.lattice.reset_uniform_with_seed(-1, FERMI_AUTO_RESEED);
        } else {
            let (w, h) = (self.lattice.width(), self.lattice.height());
            self.lattice = SpinLattice::new(w, h, Some(FERMI_AUTO_RESEED));
        }
        for s in seeds.into_iter().take(MAX_DROPLET_SEEDS) {
            apply_circular_droplet(
                &mut self.lattice,
                s.cx,
                s.cy,
                s.r,
                if s.spin >= 0 { 1 } else { -1 },
            );
        }
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
                self.lattice.reset_uniform_with_seed(-1, FERMI_AUTO_RESEED);
                self.params
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .total_trials
                    .store(0, Ordering::Relaxed);
            }
            SimCommand::Custom(ref tag) if tag == "reseed_random" => {
                let (w, h) = (self.lattice.width(), self.lattice.height());
                self.lattice = SpinLattice::new(w, h, Some(FERMI_AUTO_RESEED));
                self.params
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .total_trials
                    .store(0, Ordering::Relaxed);
            }
            SimCommand::Custom(ref tag) if tag == "apply_seed_layout" => {
                self.apply_shared_seed_layout();
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
        let (mode, j, h, theme) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.mode, p.j_coupling, p.h_field, p.theme)
        };
        let palette = theme.palette();
        let spins_vec: Vec<i8> = self.lattice.spins().to_vec();
        let mut pixels = Vec::with_capacity(width * height);
        for &s in &spins_vec {
            pixels.push(ising_spin_color(s, &palette));
        }
        let max_m = (width * height).max(1) as f64;
        let m_order = if mode == IsingInteractionMode::Ferromagnetic && j < 0.0 {
            self.lattice.staggered_magnetization() as f64 / max_m
        } else {
            self.lattice.magnetization() as f64 / max_m
        };
        let energy = self.lattice.hamiltonian(j, h);

        StateSnapshot {
            width,
            height,
            pixels: Arc::new(std::sync::RwLock::new(pixels)),
            custom_data: vec![m_order, energy],
            structured_data: Some(Box::new(spins_vec)),
            sonification: crate::audio::SonificationFrame::silent(),
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
            && p.mutation_rate.is_finite()
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
