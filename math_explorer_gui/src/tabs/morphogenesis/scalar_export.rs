//! Cold-path 2D scalar-field extraction for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui::ColorImage;
use math_explorer::biology::diffusion::ReactionCellRole;

use super::counterfactual::active_morphogen_slice;
use super::runner::{MorphoSnapshotData, MorphogenesisKineticsRunner};
use super::tool::MorphogenesisTool;
use crate::framework::ScalarField2D;
use crate::presets::MorphogenesisKineticsMode;

fn extract_runner_field(runner: &MorphogenesisKineticsRunner, periodic: bool) -> ScalarField2D {
    let mode = runner
        .kinetics_state
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .mode;
    let values = match mode {
        MorphogenesisKineticsMode::Schnakenberg => runner.system.u().to_vec(),
        MorphogenesisKineticsMode::GrayScott => runner.system.v().to_vec(),
    };
    let mut field = ScalarField2D::continuous(runner.width, runner.height, values, periodic);
    if runner.system.diffusion.mask.has_active_cells() {
        let mask: Vec<bool> = runner
            .system
            .diffusion
            .mask
            .roles()
            .iter()
            .map(|&r| ReactionCellRole::from_u8(r) != ReactionCellRole::ZeroFluxWall)
            .collect();
        field = field.with_active_mask(mask);
    }
    field
}

impl MorphogenesisTool {
    pub(crate) fn build_raster_export_image(&self) -> Option<ColorImage> {
        if self.is_forked() {
            return Some(self.render_current_heatmap_image());
        }
        if let Some((w, h, ref px)) = self.stepped_pixels {
            if w > 0 && h > 0 && px.len() == w * h {
                return Some(ColorImage {
                    size: [w, h],
                    pixels: px.clone(),
                    ..Default::default()
                });
            }
        }
        if !self.dirty && self.pixels.len() == self.width * self.height && !self.pixels.is_empty() {
            return Some(ColorImage {
                size: [self.width, self.height],
                pixels: self.pixels.clone(),
                ..Default::default()
            });
        }
        let snapshot = self.worker.latest_snapshot()?;
        let px = snapshot.pixels.read().unwrap_or_else(|e| e.into_inner());
        if snapshot.width == 0
            || snapshot.height == 0
            || px.len() != snapshot.width * snapshot.height
        {
            return None;
        }
        Some(ColorImage {
            size: [snapshot.width, snapshot.height],
            pixels: px.clone(),
            ..Default::default()
        })
    }

    pub(crate) fn build_scalar_field_2d(&self) -> Option<ScalarField2D> {
        let periodic = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .periodic_boundary;
        if self.is_forked() {
            let values = active_morphogen_slice(&self.local_runner.system).to_vec();
            let mut field = ScalarField2D::continuous(self.width, self.height, values, periodic);
            if self.local_runner.system.diffusion.mask.has_active_cells() {
                let mask: Vec<bool> = self
                    .local_runner
                    .system
                    .diffusion
                    .mask
                    .roles()
                    .iter()
                    .map(|&r| ReactionCellRole::from_u8(r) != ReactionCellRole::ZeroFluxWall)
                    .collect();
                field = field.with_active_mask(mask);
            }
            return Some(field);
        }
        if let Some(ref runner) = self.step_runner {
            return Some(extract_runner_field(runner, periodic));
        }

        let snapshot = self.worker.latest_snapshot()?;
        let expected = snapshot.width.checked_mul(snapshot.height)?;
        if snapshot.width == 0 || snapshot.height == 0 {
            return None;
        }
        let structured = snapshot.structured_data.as_ref()?;
        if let Some(mdata) = structured.downcast_ref::<MorphoSnapshotData>() {
            if mdata.u_data.len() != expected {
                return None;
            }
            let mut field = ScalarField2D::continuous(
                snapshot.width,
                snapshot.height,
                mdata.u_data.clone(),
                periodic,
            );
            if mdata.mask_roles.len() == expected {
                let mask: Vec<bool> = mdata
                    .mask_roles
                    .iter()
                    .map(|&r| ReactionCellRole::from_u8(r) != ReactionCellRole::ZeroFluxWall)
                    .collect();
                field = field.with_active_mask(mask);
            }
            return Some(field);
        }
        let u_vec = structured.downcast_ref::<Vec<f64>>()?;
        (u_vec.len() == expected).then(|| {
            ScalarField2D::continuous(snapshot.width, snapshot.height, u_vec.clone(), periodic)
        })
    }
}
