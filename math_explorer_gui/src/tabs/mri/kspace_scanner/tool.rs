//! [`InteractiveTool`] implementation and `inventory` registration for [`MriKSpaceScannerTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::phantom::synthesize_phantom;
use super::recon::{
    compose_quad_export_image, compute_forward_kspace, reconstruct_from_forward_cache,
    render_error_panel, render_grayscale_panel, render_kspace_panel, ForwardKSpaceCache,
    ReconOutput,
};
use super::state::{
    decode_mask_hex, encode_mask_hex, KSpaceBrushMode, KSpaceMaskFilter, KSpaceTrajectory,
    MriKSpaceScannerState, PhantomSource, SCANNER_STATE_BOUNDS,
};
use super::theory::MriKSpaceScannerTheory;
use super::trajectory::{apply_brush_stroke, build_parametric_mask};
use super::{KSPACE_CENTER, KSPACE_DIM};
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use crate::web_io::PlatformInstant;
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use scientific_metadata::theory::TheoryDescribable;
use std::hash::{Hash, Hasher};

/// Interactive cross-domain 2D MRI k-space Fourier imaging and frequency-sculpting tool.
pub struct MriKSpaceScannerTool {
    /// Validated persisted scanner configuration.
    pub state: MriKSpaceScannerState,
    pub(crate) ground_truth: Vec<f64>,
    pub(crate) forward_cache: ForwardKSpaceCache,
    pub(crate) parametric_mask: Vec<bool>,
    pub(crate) active_mask: Vec<bool>,
    pub(crate) recon: ReconOutput,
    pub(crate) reticle_row: usize,
    pub(crate) reticle_col: usize,
    pub(crate) phantom_dirty: bool,
    pub(crate) mask_dirty: bool,
    pub(crate) recon_dirty: bool,
    pub(crate) last_high_contrast: Option<bool>,
    pub(crate) cached_images: [egui::ColorImage; 4],
    pub(crate) textures: Option<[egui::TextureHandle; 4]>,
    pub(crate) kspace_canvas_id: Option<egui::Id>,
    pub(crate) focused_kspace_id: Option<egui::Id>,
    pub(crate) last_recon_ms: f64,
    theory: MriKSpaceScannerTheory,
}

impl Default for MriKSpaceScannerTool {
    fn default() -> Self {
        let state = MriKSpaceScannerState::default();
        let start = PlatformInstant::now();
        let ground_truth = synthesize_phantom(&state);
        let forward_cache = compute_forward_kspace(&ground_truth);
        let parametric_mask = build_parametric_mask(&state);
        let active_mask = parametric_mask.clone();
        let recon = reconstruct_from_forward_cache(&ground_truth, &forward_cache, &active_mask);
        let last_recon_ms = start.elapsed().as_secs_f64() * 1000.0;
        let cached_images = [
            render_grayscale_panel(&ground_truth, false),
            render_kspace_panel(&recon.centered_log_kspace, &active_mask, false),
            render_grayscale_panel(&recon.reconstructed, false),
            render_error_panel(&recon.abs_error, false),
        ];
        Self {
            state,
            ground_truth,
            forward_cache,
            parametric_mask,
            active_mask,
            recon,
            reticle_row: KSPACE_CENTER,
            reticle_col: KSPACE_CENTER,
            phantom_dirty: false,
            mask_dirty: false,
            recon_dirty: false,
            last_high_contrast: Some(false),
            cached_images,
            textures: None,
            kspace_canvas_id: None,
            focused_kspace_id: None,
            last_recon_ms,
            theory: MriKSpaceScannerTheory,
        }
    }
}

impl MriKSpaceScannerTool {
    /// Marks spatial phantom + forward/inverse 2D FFT caches dirty.
    pub fn mark_phantom_dirty(&mut self) {
        self.phantom_dirty = true;
        self.recon_dirty = true;
    }

    /// Clears manual brush overrides and marks the parametric trajectory/filter mask dirty.
    pub fn mark_parametric_mask_dirty(&mut self) {
        self.state.custom_mask_hex = None;
        self.mask_dirty = true;
        self.recon_dirty = true;
    }

    /// Recomputes phantom, forward FFT, k-space mask, inverse 2D FFT, and panel images only as needed.
    pub fn recompute_if_dirty(&mut self, high_contrast: bool) -> bool {
        let hc_changed = self.last_high_contrast != Some(high_contrast);
        if !self.phantom_dirty && !self.mask_dirty && !self.recon_dirty && !hc_changed {
            return false;
        }
        let gt_needs_render = self.phantom_dirty || hc_changed;
        if self.phantom_dirty {
            self.ground_truth = synthesize_phantom(&self.state);
            self.forward_cache = compute_forward_kspace(&self.ground_truth);
            self.phantom_dirty = false;
            self.recon_dirty = true;
        }
        if self.mask_dirty {
            self.parametric_mask = build_parametric_mask(&self.state);
            self.active_mask = match self.state.custom_mask_hex.as_deref() {
                Some(hex) => decode_mask_hex(hex).unwrap_or_else(|_| self.parametric_mask.clone()),
                None => self.parametric_mask.clone(),
            };
            self.mask_dirty = false;
            self.recon_dirty = true;
        }
        if self.recon_dirty {
            let start = PlatformInstant::now();
            self.recon = reconstruct_from_forward_cache(
                &self.ground_truth,
                &self.forward_cache,
                &self.active_mask,
            );
            self.recon_dirty = false;
            self.last_recon_ms = start.elapsed().as_secs_f64() * 1000.0;
        }
        if gt_needs_render {
            self.cached_images[0] = render_grayscale_panel(&self.ground_truth, high_contrast);
        }
        self.cached_images[1] =
            render_kspace_panel(&self.recon.centered_log_kspace, &self.active_mask, high_contrast);
        self.cached_images[2] = render_grayscale_panel(&self.recon.reconstructed, high_contrast);
        self.cached_images[3] = render_error_panel(&self.recon.abs_error, high_contrast);
        self.last_high_contrast = Some(high_contrast);
        if let Some(ref mut tex) = self.textures {
            for (idx, handle) in tex.iter_mut().enumerate() {
                if idx > 0 || gt_needs_render {
                    handle.set(self.cached_images[idx].clone(), egui::TextureOptions::NEAREST);
                }
            }
        }
        true
    }

    pub(crate) fn sync_custom_mask_hex(&mut self) {
        if self.active_mask == self.parametric_mask {
            self.state.custom_mask_hex = None;
        } else {
            self.state.custom_mask_hex = Some(encode_mask_hex(&self.active_mask));
        }
    }

    /// Applies the active `Paint` or `Erase` brush centered on `(row, col)`.
    pub fn paint_at_bin(&mut self, row: usize, col: usize) -> bool {
        let paint_val = self.state.brush_mode == KSpaceBrushMode::Paint;
        let changed = apply_brush_stroke(
            &mut self.active_mask,
            row.min(KSPACE_DIM - 1),
            col.min(KSPACE_DIM - 1),
            self.state.brush_radius,
            paint_val,
        );
        if changed {
            self.sync_custom_mask_hex();
            self.recon_dirty = true;
            let hc = self.last_high_contrast.unwrap_or(false);
            let _ = self.recompute_if_dirty(hc);
        }
        changed
    }

    /// Toggles the single k-space bin under `(reticle_row, reticle_col)`, returning its new state.
    pub fn toggle_reticle_bin(&mut self) -> bool {
        let r = self.reticle_row.min(KSPACE_DIM - 1);
        let c = self.reticle_col.min(KSPACE_DIM - 1);
        let idx = flatten_2d_index(c, r, KSPACE_DIM);
        self.active_mask[idx] = !self.active_mask[idx];
        let now_sampled = self.active_mask[idx];
        self.sync_custom_mask_hex();
        self.recon_dirty = true;
        let hc = self.last_high_contrast.unwrap_or(false);
        let _ = self.recompute_if_dirty(hc);
        now_sampled
    }

    /// Acquires all `4096` Nyquist k-space bins (`100%` Cartesian coverage).
    pub fn acquire_full_kspace(&mut self) {
        self.state.trajectory = KSpaceTrajectory::FullCartesian;
        self.state.mask_filter = KSpaceMaskFilter::TrajectoryOnly;
        self.state.custom_mask_hex = None;
        self.mask_dirty = true;
        self.recon_dirty = true;
        let _ = self.recompute_if_dirty(self.last_high_contrast.unwrap_or(false));
    }

    /// Zeros every k-space bin (`0%` coverage) so users can paint frequencies from scratch.
    pub fn clear_all_kspace(&mut self) {
        self.active_mask.fill(false);
        self.sync_custom_mask_hex();
        self.recon_dirty = true;
        let _ = self.recompute_if_dirty(self.last_high_contrast.unwrap_or(false));
    }

    /// Discards manual brush/reticle edits and restores the parametric trajectory + filter mask.
    pub fn reset_brush_edits(&mut self) {
        self.state.custom_mask_hex = None;
        self.mask_dirty = true;
        self.recon_dirty = true;
        let _ = self.recompute_if_dirty(self.last_high_contrast.unwrap_or(false));
    }

    /// Restores all scanner parameters, phantom, mask, and reticle position to factory defaults.
    pub fn reset_to_defaults(&mut self) {
        self.state = MriKSpaceScannerState::default();
        self.reticle_row = KSPACE_CENTER;
        self.reticle_col = KSPACE_CENTER;
        self.phantom_dirty = true;
        self.mask_dirty = true;
        self.recon_dirty = true;
        let _ = self.recompute_if_dirty(self.last_high_contrast.unwrap_or(false));
    }
}

impl InteractiveTool for MriKSpaceScannerTool {
    fn name(&self) -> &'static str {
        "MRI 2D k-Space Scanner"
    }

    fn domain(&self) -> &'static str {
        "mri"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.show_scanner_inside(ui);
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_kspace_id
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![
            (
                "Radial Mean Log10 k-Space Power".to_string(),
                self.recon.radial_power_curve.clone(),
            ),
            (
                "Cumulative Keyhole PSNR (dB)".to_string(),
                self.recon.radial_psnr_curve.clone(),
            ),
            (
                "Radial Sampling Fraction".to_string(),
                self.recon.radial_sampling_curve.clone(),
            ),
        ]
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        Vec::new()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        let refs = [
            &self.cached_images[0],
            &self.cached_images[1],
            &self.cached_images[2],
            &self.cached_images[3],
        ];
        Some(compose_quad_export_image(
            refs,
            self.last_high_contrast.unwrap_or(false),
        ))
    }

    fn step_simulation(&mut self) {
        if self.state.custom_mask_hex.is_none() {
            match self.state.trajectory {
                KSpaceTrajectory::LorenzChaotic => {
                    self.state.lorenz_steps = 256 + ((self.state.lorenz_steps + 240) % 4096);
                    self.mark_parametric_mask_dirty();
                }
                KSpaceTrajectory::RadialSpokes => {
                    self.state.radial_spokes = 6 + ((self.state.radial_spokes + 4) % 60);
                    self.mark_parametric_mask_dirty();
                }
                KSpaceTrajectory::ArchimedeanSpiral => {
                    self.state.spiral_turns = 4.0 + ((self.state.spiral_turns + 1.5) % 24.0);
                    self.mark_parametric_mask_dirty();
                }
                KSpaceTrajectory::FullCartesian | KSpaceTrajectory::CartesianLines => {
                    if matches!(
                        self.state.mask_filter,
                        KSpaceMaskFilter::Keyhole | KSpaceMaskFilter::HighPass
                    ) {
                        self.state.cutoff_radius = 2 + ((self.state.cutoff_radius + 2) % 30);
                        self.mark_parametric_mask_dirty();
                    } else if self.state.phantom == PhantomSource::SheppLogan {
                        let next_ms = (self.state.te_s * 1000.0 + 20.0) % 220.0;
                        self.state.te_s = (0.015 + next_ms / 1000.0).clamp(0.005, 0.5);
                        self.mark_phantom_dirty();
                    } else {
                        self.state.mask_filter = KSpaceMaskFilter::Keyhole;
                        self.state.cutoff_radius = 4;
                        self.mark_parametric_mask_dirty();
                    }
                }
            }
        }
        let _ = self.recompute_if_dirty(self.last_high_contrast.unwrap_or(false));
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.state.tr_s.to_bits().hash(&mut h);
        self.state.te_s.to_bits().hash(&mut h);
        self.state.t1_scale.to_bits().hash(&mut h);
        self.state.t2_scale.to_bits().hash(&mut h);
        self.state.phantom.hash(&mut h);
        self.state.trajectory.hash(&mut h);
        self.state.mask_filter.hash(&mut h);
        self.state.brush_mode.hash(&mut h);
        self.state.brush_radius.hash(&mut h);
        self.state.cartesian_stride.hash(&mut h);
        self.state.radial_spokes.hash(&mut h);
        self.state.spiral_turns.to_bits().hash(&mut h);
        self.state.lorenz_steps.hash(&mut h);
        self.state.lorenz_rho.to_bits().hash(&mut h);
        self.state.cutoff_radius.hash(&mut h);
        self.state.random_keep_ratio.to_bits().hash(&mut h);
        self.state.rng_seed.hash(&mut h);
        self.state.turing_steps.hash(&mut h);
        self.state.custom_mask_hex.hash(&mut h);
        h.finish().max(1)
    }

    fn state_bounds(&self) -> ToolStateBounds {
        SCANNER_STATE_BOUNDS
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: MriKSpaceScannerState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid k-Space scanner state: {e}"))?;
        parsed.validate()?;
        self.state = parsed;
        self.phantom_dirty = true;
        self.mask_dirty = true;
        self.recon_dirty = true;
        let _ = self.recompute_if_dirty(self.last_high_contrast.unwrap_or(false));
        Ok(())
    }

    fn has_finite_parameters(&self) -> bool {
        self.state.validate().is_ok()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.state.clamp_in_place();
        self.phantom_dirty = true;
        self.mask_dirty = true;
        self.recon_dirty = true;
        let _ = self.recompute_if_dirty(self.last_high_contrast.unwrap_or(false));
        true
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "MriKSpaceScannerTool",
        display_name: "MRI 2D k-Space Scanner",
        domain: "mri",
        tags: &[
            "mri",
            "k-space",
            "fourier",
            "fft",
            "shepp-logan",
            "turing",
            "lorenz",
            "compressed sensing",
        ],
        build: || Box::new(MriKSpaceScannerTool::default()),
    }
}
