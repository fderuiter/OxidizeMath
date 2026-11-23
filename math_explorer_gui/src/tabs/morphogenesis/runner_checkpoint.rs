//! Snapshot assembly and `K = 16` zero-allocation time-travel checkpoint helpers for [`MorphogenesisKineticsRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::overlay::render_morphogenesis_overlay;
use super::runner::{MorphoSnapshotData, MorphogenesisKineticsRunner};
use super::sculpt::overlay_cell_mask_colors;
use super::types::{MorphogenesisFlowMode, MorphogenesisKineticsMode, MorphogenesisOverlayMode};
use super::MorphogenesisTool;
use crate::app::ThemeMode;
use crate::async_sim::{CheckpointParams, CheckpointPayload, StateSnapshot, TimeTravelCheckpoint};
use eframe::egui::ColorImage;
use std::sync::Arc;

impl MorphogenesisKineticsRunner {
    fn render_snapshot_image(&self, field_vec: &[f64]) -> (ColorImage, Vec<u8>) {
        let high_contrast = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .high_contrast;
        let flow = self.read_flow();
        let theme_mode = if high_contrast {
            ThemeMode::HighContrast
        } else {
            flow.theme_mode
        };
        let mut image = if flow.flow_mode == MorphogenesisFlowMode::None
            && flow.overlay_mode == MorphogenesisOverlayMode::Concentration
        {
            MorphogenesisTool::plot_concentration_with_contrast(
                field_vec,
                self.width,
                self.height,
                high_contrast,
            )
        } else {
            render_morphogenesis_overlay(
                field_vec,
                &self.vorticity_buf,
                self.system.diffusion.obstacles(),
                self.width,
                self.height,
                flow.overlay_mode,
                theme_mode,
            )
        };
        let mask_roles = if self.system.diffusion.mask.has_active_cells() {
            let roles = self.system.diffusion.mask.roles().to_vec();
            overlay_cell_mask_colors(&mut image, &roles, high_contrast);
            roles
        } else {
            Vec::new()
        };
        (image, mask_roles)
    }

    pub(crate) fn build_snapshot(&self) -> StateSnapshot {
        let field_vec = match self.system.kinetics.mode {
            MorphogenesisKineticsMode::Schnakenberg => self.system.u().to_vec(),
            MorphogenesisKineticsMode::GrayScott => self.system.v().to_vec(),
        };
        let (image, mask_roles) = self.render_snapshot_image(&field_vec);
        StateSnapshot {
            width: self.width,
            height: self.height,
            pixels: Arc::new(std::sync::RwLock::new(image.pixels)),
            custom_data: vec![self.peclet, self.damkohler],
            structured_data: Some(Box::new(MorphoSnapshotData {
                u_data: field_vec,
                mask_roles,
                vorticity: self.vorticity_buf.clone(),
                obstacles: self.system.diffusion.obstacles().to_vec(),
                peclet: self.peclet,
                damkohler: self.damkohler,
            })),
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    pub(crate) fn capture_active_params_impl(&self) -> Option<CheckpointParams> {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        let ks = self.read_kinetics();
        Some(CheckpointParams::scalars(&[
            Self::mode_to_scalar(ks.mode),
            p.a,
            p.b,
            ks.f,
            ks.k,
            p.d_u,
            p.d_v,
            p.dt,
            self.noise_amplitude,
        ]))
    }

    pub(crate) fn capture_checkpoint_impl(&self) -> Option<TimeTravelCheckpoint> {
        let params = self.capture_active_params_impl()?;
        let mask = self
            .system
            .diffusion
            .mask
            .has_active_cells()
            .then(|| self.system.diffusion.mask.roles().into());
        Some(TimeTravelCheckpoint {
            step: self.step_count,
            rng: self.rng.clone(),
            params,
            state: CheckpointPayload::DualField2D {
                width: self.width,
                height: self.height,
                u: self.system.u().into(),
                v: self.system.v().into(),
                mask,
            },
        })
    }

    pub(crate) fn restore_checkpoint_impl(&mut self, checkpoint: &TimeTravelCheckpoint) {
        if let CheckpointPayload::DualField2D {
            width,
            height,
            ref u,
            ref v,
            ref mask,
        } = checkpoint.state
        {
            if width == self.width
                && height == self.height
                && u.len() == self.system.u().len()
                && v.len() == self.system.v().len()
            {
                self.system.u_mut().copy_from_slice(u);
                self.system.v_mut().copy_from_slice(v);
                match mask {
                    Some(bytes) if bytes.len() == width * height => {
                        let _ = self.system.diffusion.mask.set_roles_slice(bytes);
                    }
                    _ => self.system.diffusion.mask.clear(),
                }
                self.system.diffusion.sync_obstacles_from_mask();
            }
        }
        if let Some(p) = checkpoint.params.as_scalars() {
            let _ = self.apply_scalar_params(p);
        }
        self.step_count = checkpoint.step;
        self.rng = checkpoint.rng.clone();
    }

    pub(crate) fn replay_step_impl(&mut self, params: &CheckpointParams) {
        if let Some(dt) = params
            .as_scalars()
            .and_then(|p| self.apply_scalar_params(p))
        {
            self.advance_grid_step(dt, self.noise_amplitude);
        }
    }
}
