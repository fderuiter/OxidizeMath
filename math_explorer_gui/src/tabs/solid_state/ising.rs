use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::{
    decode_u8_grid, encode_u8_grid, InteractiveTool, RasterBrushState, RasterBrushStroke,
};
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::stat_mech::ising::SpinLattice;
use math_explorer::physics::stat_mech::KB;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[path = "ising/runner.rs"]
mod runner;
#[path = "ising/ui.rs"]
mod ui;
pub use runner::*;

pub(crate) const DEFAULT_ISING_SIZE: usize = 100;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct IsingState {
    temperature: f64,
    j_coupling: f64,
    h_field: f64,
    #[serde(default)]
    state: Option<SimWorkerState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    running: Option<bool>,
    steps_per_frame: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pinned_rle: Option<String>,
}

pub struct IsingModelTool {
    pub(crate) temperature: f64,
    pub(crate) j_coupling: f64,
    pub(crate) h_field: f64,
    pub(crate) steps_per_frame: usize,
    pub(crate) magnetization: f64,
    pub(crate) energy: f64,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) pixels: Vec<egui::Color32>,
    pub(crate) stepped_pixels: Option<(usize, usize, Vec<egui::Color32>)>,
    pub(crate) last_theme: Option<crate::app::ThemeMode>,
    pub(crate) dirty: bool,
    pub(crate) brush_state: RasterBrushState,
    pub(crate) brush_mode: IsingBrushMode,
    pub(crate) pin_cells: Vec<u8>,
    pub(crate) params: Arc<std::sync::RwLock<IsingSharedParams>>,
    pub worker: BackgroundWorker<IsingRunner>,
}

impl IsingModelTool {
    fn build_runner(params: &Arc<std::sync::RwLock<IsingSharedParams>>) -> IsingRunner {
        let mut lattice = SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, None);
        let pinned_rle = params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .pinned_rle
            .clone();
        apply_pinned_rle_to_lattice(&mut lattice, pinned_rle.as_deref());
        IsingRunner {
            lattice,
            params: Arc::clone(params),
        }
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn restart_worker(&mut self) {
        let runner = Self::build_runner(&self.params);
        self.worker = BackgroundWorker::new(runner);
        self.stepped_pixels = None;
        self.sync_metrics_from_latest();
        self.dirty = true;
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn retry(&mut self) {
        self.restart_worker();
    }

    /// Resets the worker lattice in place or respawns a disconnected worker.
    pub fn reset(&mut self) {
        let params = Arc::clone(&self.params);
        self.worker
            .reset_or_respawn(|| Self::build_runner(&params));
        self.stepped_pixels = None;
        self.sync_metrics_from_latest();
        self.dirty = true;
    }

    /// Applies a [`RasterBrushStroke`] to spin domains and/or the boundary pin mask.
    pub fn apply_brush_stroke(&mut self, stroke: RasterBrushStroke, ctx: Option<&egui::Context>) {
        let effective_mode = if stroke.secondary {
            IsingBrushMode::Unpin
        } else {
            self.brush_mode
        };
        if self.pin_cells.len() != DEFAULT_ISING_SIZE * DEFAULT_ISING_SIZE {
            self.pin_cells
                .resize(DEFAULT_ISING_SIZE * DEFAULT_ISING_SIZE, 0);
        }
        let pin_sym = match effective_mode {
            IsingBrushMode::PinUp => Some(1u8),
            IsingBrushMode::PinDown => Some(2u8),
            IsingBrushMode::Unpin => Some(0u8),
            IsingBrushMode::SpinUp | IsingBrushMode::SpinDown => None,
        };
        if let Some(sym) = pin_sym {
            for &(x, y) in &stroke.cells {
                if x < DEFAULT_ISING_SIZE && y < DEFAULT_ISING_SIZE {
                    self.pin_cells[flatten_2d_index(x, y, DEFAULT_ISING_SIZE)] = sym;
                }
            }
            self.sync_pinned_rle();
        }
        let _ = self.worker.send_command(SimCommand::ApplyBrushCells {
            cells: stroke.cells,
            mode: effective_mode.to_code(),
        });
        self.stepped_pixels = None;
        self.dirty = true;
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    fn sync_pinned_rle(&mut self) {
        let next_rle = if self.pin_cells.iter().any(|&c| c != 0) {
            encode_u8_grid(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, &self.pin_cells, 2).ok()
        } else {
            None
        };
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .pinned_rle = next_rle;
    }

    /// Clears all pinned boundary sites across UI and worker threads.
    pub fn clear_pins(&mut self, ctx: Option<&egui::Context>) {
        self.pin_cells.fill(0);
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .pinned_rle = None;
        let _ = self
            .worker
            .send_command(SimCommand::Custom("clear_pins".to_string()));
        self.stepped_pixels = None;
        self.dirty = true;
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    fn sync_metrics_from_latest(&mut self) {
        if let Some(snapshot) = self.worker.latest_snapshot() {
            if snapshot.custom_data.len() >= 2 {
                self.magnetization = snapshot.custom_data[0];
                self.energy = snapshot.custom_data[1];
            }
        }
    }
}

impl Default for IsingModelTool {
    fn default() -> Self {
        let temperature = 2.269;
        let j_coupling = 1.0;
        let h_field = 0.0;
        let steps_per_frame = 5000;
        let params = Arc::new(std::sync::RwLock::new(IsingSharedParams {
            temperature,
            j_coupling,
            h_field,
            steps_per_frame,
            theme: crate::app::ThemeMode::Dark,
            pinned_rle: None,
        }));
        let runner = Self::build_runner(&params);
        let worker = BackgroundWorker::new(runner);
        let mut tool = Self {
            temperature,
            j_coupling,
            h_field,
            steps_per_frame,
            magnetization: 0.0,
            energy: 0.0,
            texture: None,
            pixels: Vec::new(),
            stepped_pixels: None,
            last_theme: None,
            dirty: true,
            brush_state: RasterBrushState::default(),
            brush_mode: IsingBrushMode::default(),
            pin_cells: vec![0; DEFAULT_ISING_SIZE * DEFAULT_ISING_SIZE],
            params,
            worker,
        };
        tool.sync_metrics_from_latest();
        tool
    }
}

impl InteractiveTool for IsingModelTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Ising Model"
    }
    fn domain(&self) -> &'static str {
        "solid_state"
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.brush_state.focused_id
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.temperature.to_bits().hash(&mut h);
        self.j_coupling.to_bits().hash(&mut h);
        self.h_field.to_bits().hash(&mut h);
        self.steps_per_frame.hash(&mut h);
        self.params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .pinned_rle
            .hash(&mut h);
        std::mem::discriminant(&self.worker.state()).hash(&mut h);
        h.finish().max(1)
    }

    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::raster_only()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        if let Some((w, h, ref px)) = self.stepped_pixels {
            if w > 0 && h > 0 && !px.is_empty() {
                return Some(egui::ColorImage::new([w, h], px.clone()));
            }
        }
        let snap = self.worker.latest_snapshot()?;
        let px = snap.pixels.try_read().ok()?;
        if px.is_empty() || snap.width == 0 || snap.height == 0 {
            return None;
        }
        Some(egui::ColorImage::new([snap.width, snap.height], px.clone()))
    }

    fn step_simulation(&mut self) {
        let mut lattice = SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, Some(42));
        let (theme, pinned_rle) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.theme, p.pinned_rle.clone())
        };
        apply_pinned_rle_to_lattice(&mut lattice, pinned_rle.as_deref());
        let j_scale = if self.j_coupling.abs() < 1e-6 {
            1.0
        } else {
            self.j_coupling.abs()
        };
        let t_phys = self.temperature * j_scale / KB;
        lattice.evolve(
            self.steps_per_frame.clamp(100, 5000),
            t_phys,
            self.j_coupling,
            self.h_field,
        );
        let palette = theme.palette();
        let pixels: Vec<egui::Color32> = lattice
            .spins()
            .iter()
            .zip(lattice.pinned().iter())
            .map(|(&s, &pin)| ising_spin_color(if pin { s.saturating_mul(2) } else { s }, &palette))
            .collect();
        self.stepped_pixels = Some((lattice.width(), lattice.height(), pixels));
        self.dirty = true;
    }

    fn save_state(&self) -> Option<String> {
        let pinned_rle = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .pinned_rle
            .clone();
        serde_json::to_string(&IsingState {
            temperature: self.temperature,
            j_coupling: self.j_coupling,
            h_field: self.h_field,
            state: Some(self.worker.state()),
            running: None,
            steps_per_frame: self.steps_per_frame,
            pinned_rle,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: IsingState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Ising state: {e}"))?;
        if !s.temperature.is_finite()
            || !(0.1..=10.0).contains(&s.temperature)
            || !s.j_coupling.is_finite()
            || !(-2.0..=2.0).contains(&s.j_coupling)
            || !s.h_field.is_finite()
            || !(-2.0..=2.0).contains(&s.h_field)
            || !(100..=50_000).contains(&s.steps_per_frame)
        {
            return Err("Ising parameters out of valid bounds".to_string());
        }
        let decoded_pins = match s.pinned_rle.as_deref() {
            Some(rle) => decode_u8_grid(rle, DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, 2)?,
            None => vec![0; DEFAULT_ISING_SIZE * DEFAULT_ISING_SIZE],
        };
        let pins_changed = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .pinned_rle
            != s.pinned_rle;
        self.temperature = s.temperature;
        self.j_coupling = s.j_coupling;
        self.h_field = s.h_field;
        self.steps_per_frame = s.steps_per_frame;
        self.pin_cells = decoded_pins;
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.temperature = self.temperature;
            p.j_coupling = self.j_coupling;
            p.h_field = self.h_field;
            p.steps_per_frame = self.steps_per_frame;
            p.pinned_rle = s.pinned_rle;
        }
        if pins_changed {
            self.reset();
        }
        let target_state = s.state.unwrap_or_else(|| {
            if s.running.unwrap_or(false) {
                SimWorkerState::Running
            } else {
                SimWorkerState::Ready
            }
        });
        if target_state == SimWorkerState::Running {
            if !self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Start);
            }
        } else if self.worker.is_running() {
            let _ = self.worker.send_command(SimCommand::Pause);
        }
        self.stepped_pixels = None;
        self.dirty = true;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let theme = crate::app::ThemeMode::from_visuals(ui.visuals());
        if self.last_theme != Some(theme) {
            self.last_theme = Some(theme);
            self.params.write().unwrap_or_else(|e| e.into_inner()).theme = theme;
            self.dirty = true;
        }
        if let Some(snapshot) = self.worker.update() {
            if snapshot.custom_data.len() >= 2 {
                self.magnetization = snapshot.custom_data[0];
                self.energy = snapshot.custom_data[1];
            }
            self.stepped_pixels = None;
            self.dirty = true;
        }

        if matches!(
            self.worker.state(),
            SimWorkerState::Running | SimWorkerState::Stepping
        ) {
            ctx.request_repaint();
        }

        self.draw_side_panel(ui, &ctx);
        self.draw_central_panel(ui, &ctx);
    }
}

#[path = "ising/preset.rs"]
mod preset;

#[cfg(test)]
use crate::async_sim::SimulationRunner;
#[cfg(test)]
#[path = "ising/tests.rs"]
mod tests;

// [cite:clinical_trials_statistics]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "IsingModelTool",
        display_name: "Ising Model",
        domain: "solid_state",
        tags: &[],
        build: || Box::new(IsingModelTool::default()),
    }
}
