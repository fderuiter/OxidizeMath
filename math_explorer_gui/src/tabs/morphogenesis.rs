// @explorer_feature = "biology"
use crate::async_sim::ui::{render_worker_fault_banner, WorkerFaultAction};
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard, RecoveryAction};
use crate::framework::{PlaybackAction, PlaybackTransport};
use crate::presets::PresetDialogState;
use crate::tabs::ExplorerTab;
use eframe::egui;
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{SchnakenbergKinetics, TuringSystem};
use std::sync::Arc;

#[derive(serde::Serialize, serde::Deserialize)]
struct MorphoState {
    dt: f64,
    #[serde(default)]
    state: Option<SimWorkerState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    paused: Option<bool>,
    #[serde(default)]
    show_starter_gallery: bool,
}

#[path = "morphogenesis/runner.rs"]
mod runner_impl;
pub use runner_impl::*;

#[path = "morphogenesis/ui.rs"]
mod ui_impl;

#[cfg_attr(any(), verified(opt_out = "gui_tool"))]
pub struct MorphogenesisTab {
    pub worker: BackgroundWorker<MorphogenesisRunner>,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    width: usize,
    height: usize,
    texture: Option<egui::TextureHandle>,
    cached_image: Option<Arc<egui::ColorImage>>,
    pub(crate) dialog_state: PresetDialogState,
    pixels: Vec<egui::Color32>,
    dirty: bool,
    pub show_starter_gallery: bool,
}

impl MorphogenesisTab {
    pub(crate) fn create_runner(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        width: usize,
        height: usize,
    ) -> MorphogenesisRunner {
        use math_explorer::biology::morphogenesis::DiffusionCoeff;
        use math_explorer::math_kernel::types::{Dimension, StepSize};
        let (a, b, d_u, d_v) = {
            let p = params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v)
        };
        let kinetics = SchnakenbergKinetics { a, b };
        let diffusion = FiniteDifference2D::new(
            Dimension(width),
            Dimension(height),
            StepSize(1.0),
            StepSize(1.0),
        );
        let mut system = TuringSystem::new_with_kinetics(
            Dimension(width * height),
            DiffusionCoeff(d_u),
            DiffusionCoeff(d_v),
            kinetics,
            diffusion,
        );
        Self::initialize_system(&mut system, width, height);
        MorphogenesisRunner {
            system,
            params: Arc::clone(params),
            width,
            height,
        }
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn restart_worker(&mut self) {
        let runner = Self::create_runner(&self.params, self.width, self.height);
        self.worker = BackgroundWorker::new(runner);
        self.dirty = true;
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn retry(&mut self) {
        self.restart_worker();
    }

    /// Resets the worker in place or respawns a disconnected worker in `Ready` state.
    pub fn reset(&mut self) {
        let params = Arc::clone(&self.params);
        let width = self.width;
        let height = self.height;
        self.worker
            .reset_or_respawn(|| Self::create_runner(&params, width, height));
        self.dirty = true;
    }
}

impl Default for MorphogenesisTab {
    fn default() -> Self {
        let width = 100;
        let height = 100;
        let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams {
            a: 0.1,
            b: 0.9,
            d_u: 1.0,
            d_v: 100.0,
            dt: 0.05,
        }));

        let runner = Self::create_runner(&params, width, height);
        let mut worker = BackgroundWorker::new(runner);
        let _ = worker.send_command(SimCommand::Start);

        Self {
            worker,
            params,
            width,
            height,
            texture: None,
            cached_image: None,
            dialog_state: PresetDialogState::default(),
            pixels: Vec::new(),
            dirty: true,
            show_starter_gallery: false,
        }
    }
}

impl ExplorerTab for MorphogenesisTab {
    fn name(&self) -> &'static str {
        "Morphogenesis"
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.params.read().unwrap_or_else(|e| e.into_inner()).dt.to_bits().hash(&mut h);
        std::mem::discriminant(&self.worker.state()).hash(&mut h);
        self.show_starter_gallery.hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&MorphoState {
            dt: self.params.read().unwrap_or_else(|e| e.into_inner()).dt,
            state: Some(self.worker.state()),
            paused: None,
            show_starter_gallery: self.show_starter_gallery,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(s) = serde_json::from_str::<MorphoState>(state) {
            self.params.write().unwrap_or_else(|e| e.into_inner()).dt = s.dt;
            self.show_starter_gallery = false;
            let target = s.state.unwrap_or_else(|| {
                if s.paused.unwrap_or(false) {
                    SimWorkerState::Ready
                } else {
                    SimWorkerState::Running
                }
            });
            if target == SimWorkerState::Running && !self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Start);
            } else if target != SimWorkerState::Running && self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Pause);
            }
            self.dirty = true;
        }
    }

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.worker.update().is_some() {
            self.dirty = true;
        }

        if matches!(
            self.worker.state(),
            SimWorkerState::Running | SimWorkerState::Stepping
        ) {
            ctx.request_repaint();
        }

        self.draw_controls(ctx);
        self.draw_canvas(ctx);

        if self.show_starter_gallery {
            let cards = crate::starter_gallery::get_starter_cards_for_domain("biology", &[]);
            let mut is_open = self.show_starter_gallery;
            if let Some(card) = crate::starter_gallery::show_starter_gallery_overlay(
                ctx,
                "morphogenesis",
                &cards,
                &mut is_open,
            ) {
                if self
                    .apply_starter_card(card.title, card.preset_json)
                    .is_ok()
                {
                    ctx.request_repaint();
                }
                is_open = false;
            }
            self.show_starter_gallery = is_open;
        }
    }
}

impl FiniteStateGuard for MorphogenesisTab {
    fn is_finite_state(&self) -> bool {
        self.worker.state() != SimWorkerState::Faulted && self.worker.diagnostic_report().is_none()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.worker.diagnostic_report()
    }

    fn diverged(&self) -> bool {
        !self.is_finite_state()
    }

    fn auto_scale_step_size(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.dt = (p.dt * 0.5).max(1e-4);
        drop(p);
        self.reset_system();
    }

    fn clamp_parameters(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.a = p.a.clamp(0.0, 1.0);
        p.b = p.b.clamp(0.0, 2.0);
        p.d_u = p.d_u.clamp(0.1, 5.0);
        p.d_v = p.d_v.clamp(10.0, 200.0);
        p.dt = p.dt.clamp(0.001, 0.1);
        drop(p);
        self.reset_system();
    }

    fn reset_parameters(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.a = 0.1;
        p.b = 0.9;
        p.d_u = 1.0;
        p.d_v = 100.0;
        p.dt = 0.05;
        drop(p);
        self.reset_system();
    }
}

impl MorphogenesisTab {
    pub(crate) fn reset_system(&mut self) {
        self.reset();
        let _ = self.worker.send_command(SimCommand::Start);
    }

    fn draw_parameter_sliders(&self, ui: &mut egui::Ui) {
        let mut params = self.params.write().unwrap_or_else(|e| e.into_inner());
        ui.add(egui::Slider::new(&mut params.a, 0.0..=1.0).text("a"));
        ui.add(egui::Slider::new(&mut params.b, 0.0..=2.0).text("b"));
        ui.add(egui::Slider::new(&mut params.d_u, 0.1..=5.0).text("D_u"));
        ui.add(egui::Slider::new(&mut params.d_v, 10.0..=200.0).text("D_v"));
    }

    fn draw_presets(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Presets:");
        if ui.button("Spots").clicked() {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v) = (0.1, 0.9, 1.0, 100.0);
        }
        if ui.button("Stripes").clicked() {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v) = (0.14, 0.86, 1.0, 50.0);
        }

        let mut dialog_state = std::mem::take(&mut self.dialog_state);
        let domain = self.domain();
        crate::presets::render_preset_buttons(
            &mut dialog_state,
            ui,
            self,
            domain,
            Self::to_json_string,
            Self::from_json_str,
        );
        self.dialog_state = dialog_state;
    }

    fn draw_fault_and_recovery(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if let Some(action) = render_worker_fault_banner(
            ui,
            self.worker.state(),
            self.worker.diagnostic_report(),
            true,
        ) {
            match action {
                WorkerFaultAction::Reset => self.reset(),
                WorkerFaultAction::ResetParameters => self.reset_parameters(),
                WorkerFaultAction::Recovery(RecoveryAction::AutoScaleStepSize) => {
                    self.auto_scale_step_size();
                }
                WorkerFaultAction::Recovery(RecoveryAction::ClampParameters) => {
                    self.clamp_parameters();
                }
                WorkerFaultAction::Recovery(RecoveryAction::ResetParameters) => {
                    self.reset_parameters();
                }
            }
            ctx.request_repaint();
            ui.separator();
        }
    }

    pub(crate) fn draw_controls(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("morphogenesis_controls").show(ctx, |ui| {
            ui.heading("Morphogenesis Controls");
            if ui.button("🎨 Starter Gallery").clicked() {
                self.show_starter_gallery = !self.show_starter_gallery;
            }
            ui.separator();

            self.draw_fault_and_recovery(ui, ctx);
            self.draw_parameter_sliders(ui);
            ui.separator();

            if let Some(action) = PlaybackTransport::from_worker_state(
                "morphogenesis_transport",
                "Morphogenesis",
                self.worker.state(),
            )
            .can_reset(true)
            .show(ui)
            {
                match action {
                    PlaybackAction::Play => {
                        let _ = self.worker.send_command(SimCommand::Start);
                    }
                    PlaybackAction::Pause => {
                        let _ = self.worker.send_command(SimCommand::Pause);
                    }
                    PlaybackAction::Step => {
                        let _ = self.worker.send_command(SimCommand::Step(1));
                    }
                    PlaybackAction::Reset => {
                        self.reset();
                        ctx.request_repaint();
                    }
                }
            }

            self.draw_presets(ui);
        });
    }
}

#[path = "morphogenesis/presets.rs"]
mod presets_impl;
pub use presets_impl::*;

#[cfg(test)]
use math_commons::math_kernel::colormap::heatmap_color;
#[cfg(test)]
#[path = "morphogenesis/tests.rs"]
mod tests;
