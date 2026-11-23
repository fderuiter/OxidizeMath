//! Cold-path 2D scalar-field extraction for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_explorer::biology::diffusion::ReactionCellRole;

use super::runner::MorphoSnapshotData;
use super::tool::MorphogenesisTool;
use crate::framework::ScalarField2D;
use crate::presets::MorphogenesisKineticsMode;

impl MorphogenesisTool {
    pub(crate) fn build_scalar_field_2d(&self) -> Option<ScalarField2D> {
        let periodic = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .periodic_boundary;
        if let Some(ref runner) = self.step_runner {
            let mode = runner
                .kinetics_state
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .mode;
            let values = match mode {
                MorphogenesisKineticsMode::Schnakenberg => runner.system.u().to_vec(),
                MorphogenesisKineticsMode::GrayScott => runner.system.v().to_vec(),
            };
            let mut field =
                ScalarField2D::continuous(runner.width, runner.height, values, periodic);
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
            return Some(field);
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
