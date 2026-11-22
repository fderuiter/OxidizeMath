#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::DiagnosticReport;
use eframe::egui;
use math_explorer::physics::stat_mech::ising::SpinLattice;
use math_explorer::physics::stat_mech::KB;
use std::sync::Arc;

#[allow(missing_docs)]
pub struct IsingSharedParams {
    pub temperature: f64,
    pub j_coupling: f64,
    pub h_field: f64,
    pub steps_per_frame: usize,
    pub theme: crate::app::ThemeMode,
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

#[allow(missing_docs)]
pub struct IsingRunner {
    pub lattice: SpinLattice,
    pub params: Arc<std::sync::RwLock<IsingSharedParams>>,
}

impl IsingRunner {
    fn evolve_trials(&mut self, num_trials: usize) {
        let (temp, j, h) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.temperature, p.j_coupling, p.h_field)
        };
        if !temp.is_finite() || !j.is_finite() || !h.is_finite() {
            return;
        }
        let j_scale = if j.abs() < 1e-6 { 1.0 } else { j.abs() };
        let t_phys = temp * j_scale / KB;
        self.lattice.evolve(num_trials, t_phys, j, h);
    }
}

impl SimulationRunner for IsingRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => {
                let w = self.lattice.width();
                let h = self.lattice.height();
                self.lattice = SpinLattice::new(w, h, None);
            }
            SimCommand::SetSpeed(speed) => {
                self.params
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .steps_per_frame = speed;
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
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_finite_state(&self) -> bool {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        p.temperature.is_finite() && p.j_coupling.is_finite() && p.h_field.is_finite()
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
