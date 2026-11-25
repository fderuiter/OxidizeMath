#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub use super::brush::{
    apply_circular_droplet, apply_ising_brush_cells, apply_pinned_rle_to_lattice,
};
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

/// Interactive brush modes for painting spin domains and boundary pinning in [`super::IsingModelTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IsingBrushMode {
    /// Sets spin to `+1` without pinning.
    #[default]
    SpinUp,
    /// Sets spin to `-1` without pinning.
    SpinDown,
    /// Sets spin to `+1` and pins against Metropolis flips.
    PinUp,
    /// Sets spin to `-1` and pins against Metropolis flips.
    PinDown,
    /// Unpins touched sites so thermal sweeps can flip them again.
    Unpin,
}

impl IsingBrushMode {
    /// All selectable Ising brush modes in UI display order.
    pub const ALL: [Self; 5] = [
        Self::SpinUp,
        Self::SpinDown,
        Self::PinUp,
        Self::PinDown,
        Self::Unpin,
    ];

    /// User-facing label for side-panel controls.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SpinUp => "Paint Spin Up (+1)",
            Self::SpinDown => "Paint Spin Down (-1)",
            Self::PinUp => "Pin Spin Up (+1 Fixed)",
            Self::PinDown => "Pin Spin Down (-1 Fixed)",
            Self::Unpin => "Unpin Sites",
        }
    }

    /// Wire command byte passed via [`SimCommand::ApplyBrushCells`].
    #[must_use]
    pub const fn to_code(self) -> u8 {
        match self {
            Self::Unpin => 0,
            Self::SpinUp => 1,
            Self::SpinDown => 2,
            Self::PinUp => 3,
            Self::PinDown => 4,
        }
    }

    /// Decodes a wire command byte back to [`IsingBrushMode`].
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::SpinUp,
            2 => Self::SpinDown,
            3 => Self::PinUp,
            4 => Self::PinDown,
            _ => Self::Unpin,
        }
    }
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
    pub pinned_rle: Option<String>,
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
            pinned_rle: None,
            total_trials: AtomicU64::new(0),
        }
    }
}

/// Maps an Ising lattice spin (`±1` free, `±2` pinned) to a WCAG `>= 3.0:1` [`crate::app::SemanticPalette`] color.
#[inline]
#[must_use]
pub fn ising_spin_color(spin: i8, palette: &crate::app::SemanticPalette) -> egui::Color32 {
    palette.ising_spin_color(spin)
}

#[allow(missing_docs)]
pub struct IsingRunner {
    pub lattice: SpinLattice,
    pub params: Arc<std::sync::RwLock<IsingSharedParams>>,
}

const ISING_SLICE_TRIAL_CHUNK: usize = 512;

impl IsingRunner {
    fn evolve_trials_sliced(&mut self, num_trials: usize, max_slice: Option<std::time::Duration>) {
        if num_trials == 0 {
            return;
        }
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
        let completed = match mode {
            IsingInteractionMode::Ferromagnetic => {
                if !j.is_finite() || !h.is_finite() {
                    return;
                }
                let t_phys = temp.max(1e-3) / KB;
                self.evolve_ferromagnetic_chunks(num_trials, t_phys, j, h, max_slice)
            }
            IsingInteractionMode::FermiGame => {
                if !payoffs.iter().all(|v| v.is_finite()) {
                    return;
                }
                let mat = DMatrix::from_row_slice(2, 2, &payoffs);
                let Ok(payoff) = MatrixPayoff::new(mat) else {
                    return;
                };
                self.evolve_fermi_chunks(num_trials, temp, &payoff, mutation_rate, max_slice)
            }
        };
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        p.total_trials
            .fetch_add(completed as u64, Ordering::Relaxed);
    }

    fn evolve_ferromagnetic_chunks(
        &mut self,
        num_trials: usize,
        t_phys: f64,
        j: f64,
        h: f64,
        max_slice: Option<std::time::Duration>,
    ) -> usize {
        let Some(raw_budget) = max_slice else {
            self.lattice.evolve(num_trials, t_phys, j, h);
            return num_trials;
        };
        let reserve = std::time::Duration::from_micros(150).min(raw_budget / 4);
        let budget = raw_budget.saturating_sub(reserve);
        let start = crate::web_io::PlatformInstant::now();
        let mut done = 0_usize;
        while done < num_trials {
            if done > 0 && start.elapsed() >= budget {
                break;
            }
            let batch = (num_trials - done).min(ISING_SLICE_TRIAL_CHUNK);
            self.lattice.evolve(batch, t_phys, j, h);
            done += batch;
        }
        done
    }

    fn evolve_fermi_chunks(
        &mut self,
        num_trials: usize,
        temp: f64,
        payoff: &MatrixPayoff,
        mutation_rate: f64,
        max_slice: Option<std::time::Duration>,
    ) -> usize {
        let Some(raw_budget) = max_slice else {
            self.lattice
                .evolve_fermi_game_with_mutation(num_trials, temp, payoff, mutation_rate);
            return num_trials;
        };
        let reserve = std::time::Duration::from_micros(150).min(raw_budget / 4);
        let budget = raw_budget.saturating_sub(reserve);
        let start = crate::web_io::PlatformInstant::now();
        let mut done = 0_usize;
        while done < num_trials {
            if done > 0 && start.elapsed() >= budget {
                break;
            }
            let batch = (num_trials - done).min(ISING_SLICE_TRIAL_CHUNK);
            self.lattice
                .evolve_fermi_game_with_mutation(batch, temp, payoff, mutation_rate);
            done += batch;
        }
        done
    }

    fn apply_shared_seed_layout(&mut self) {
        let (defector_sea, seeds, pinned_rle) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            p.total_trials.store(0, Ordering::Relaxed);
            (
                p.defector_sea,
                p.droplet_seeds.clone(),
                p.pinned_rle.clone(),
            )
        };
        if defector_sea {
            self.lattice.reset_uniform_with_seed(-1, FERMI_AUTO_RESEED);
        } else {
            let (w, h) = (self.lattice.width(), self.lattice.height());
            self.lattice = SpinLattice::new(w, h, Some(FERMI_AUTO_RESEED));
        }
        apply_pinned_rle_to_lattice(&mut self.lattice, pinned_rle.as_deref());
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
                let pinned_rle = {
                    let p = self.params.read().unwrap_or_else(|e| e.into_inner());
                    p.total_trials.store(0, Ordering::Relaxed);
                    p.pinned_rle.clone()
                };
                apply_pinned_rle_to_lattice(&mut self.lattice, pinned_rle.as_deref());
            }
            SimCommand::Custom(ref s) if s == "clear_pins" => {
                self.lattice.clear_pinned();
            }
            SimCommand::ApplyBrushCells { cells, mode } => {
                apply_ising_brush_cells(&mut self.lattice, &cells, IsingBrushMode::from_code(mode));
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
                let pinned_rle = {
                    let p = self.params.read().unwrap_or_else(|e| e.into_inner());
                    p.total_trials.store(0, Ordering::Relaxed);
                    p.pinned_rle.clone()
                };
                apply_pinned_rle_to_lattice(&mut self.lattice, pinned_rle.as_deref());
            }
            SimCommand::Custom(ref tag) if tag == "reseed_random" => {
                let (w, h) = (self.lattice.width(), self.lattice.height());
                self.lattice = SpinLattice::new(w, h, Some(FERMI_AUTO_RESEED));
                let pinned_rle = {
                    let p = self.params.read().unwrap_or_else(|e| e.into_inner());
                    p.total_trials.store(0, Ordering::Relaxed);
                    p.pinned_rle.clone()
                };
                apply_pinned_rle_to_lattice(&mut self.lattice, pinned_rle.as_deref());
            }
            SimCommand::Custom(ref tag) if tag == "apply_seed_layout" => {
                self.apply_shared_seed_layout();
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        self.step_sliced(None);
    }

    fn step_sliced(&mut self, max_slice: Option<std::time::Duration>) {
        let steps = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .steps_per_frame;
        self.evolve_trials_sliced(steps, max_slice);
    }

    fn step_once(&mut self) {
        let one_sweep = self.lattice.width() * self.lattice.height();
        self.evolve_trials_sliced(one_sweep, None);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let width = self.lattice.width();
        let height = self.lattice.height();
        let (mode, j, h, theme) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.mode, p.j_coupling, p.h_field, p.theme)
        };
        let palette = theme.palette();
        let mut spins_vec = Vec::with_capacity(width * height);
        let mut pixels = Vec::with_capacity(width * height);
        for (&s, &pin) in self
            .lattice
            .spins()
            .iter()
            .zip(self.lattice.pinned().iter())
        {
            let encoded = if pin { s.saturating_mul(2) } else { s };
            spins_vec.push(encoded);
            pixels.push(ising_spin_color(encoded, &palette));
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
