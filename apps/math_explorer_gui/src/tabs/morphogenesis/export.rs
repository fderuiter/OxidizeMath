#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ExportCell, ExportTable};
use math_commons::math_kernel::types::flatten_2d_index;

use super::counterfactual::{active_morphogen_slice, MorphogenesisSnapshotData};
use super::MorphogenesisTool;

impl MorphogenesisTool {
    fn worker_morphogenesis_table(&self, width: usize, height: usize) -> Option<ExportTable> {
        let num_cells = width * height;
        let snap = self.worker.latest_snapshot()?;
        let data = snap
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<MorphogenesisSnapshotData>())
            .filter(|d| {
                d.u_data.len() == num_cells && d.step_count >= self.local_runner.step_count
            })?;
        Some(ExportTable::from_bounded_grid_2d(
            "Morphogenesis Concentration Grid",
            width,
            height,
            vec![
                "x".to_string(),
                "y".to_string(),
                "active_species".to_string(),
                "cell_role".to_string(),
            ],
            |x, y| {
                let idx = flatten_2d_index(x, y, width);
                vec![
                    ExportCell::Int(x as i64),
                    ExportCell::Int(y as i64),
                    ExportCell::Float(data.u_data.get(idx).copied().unwrap_or(0.0)),
                    ExportCell::Int(i64::from(data.mask_roles.get(idx).copied().unwrap_or(0))),
                ]
            },
        ))
    }

    pub(crate) fn build_morphogenesis_export_tables(&self) -> Vec<ExportTable> {
        if self.is_forked() {
            return Vec::new();
        }
        let (width, height) = (self.width.max(1), self.height.max(1));
        if let Some(table) = self.worker_morphogenesis_table(width, height) {
            return vec![table];
        }
        let u_slice = self.local_runner.system.u();
        let v_slice = self.local_runner.system.v();
        let active_slice = active_morphogen_slice(&self.local_runner.system);
        let roles = self.local_runner.system.diffusion.mask.roles();
        let table = ExportTable::from_bounded_grid_2d(
            "Morphogenesis Concentration Grid",
            width,
            height,
            vec![
                "x".to_string(),
                "y".to_string(),
                "u".to_string(),
                "v".to_string(),
                "active_species".to_string(),
                "cell_role".to_string(),
            ],
            |x, y| {
                let idx = flatten_2d_index(x, y, width);
                vec![
                    ExportCell::Int(x as i64),
                    ExportCell::Int(y as i64),
                    ExportCell::Float(u_slice.get(idx).copied().unwrap_or(0.0)),
                    ExportCell::Float(v_slice.get(idx).copied().unwrap_or(0.0)),
                    ExportCell::Float(active_slice.get(idx).copied().unwrap_or(0.0)),
                    ExportCell::Int(i64::from(roles.get(idx).copied().unwrap_or(0))),
                ]
            },
        );
        vec![table]
    }

    pub(crate) fn build_morphogenesis_compute_shader_wgsl(&self) -> Result<String, String> {
        use crate::framework::shader_foundry::{
            emit_pde_stencil_compute_wgsl, PdeComputeKinetics, PdeStencilComputeSpec,
        };
        use crate::presets::MorphogenesisKineticsMode;
        let ks = *self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        let (kinetics, rate_primary, rate_secondary) = match ks.mode {
            MorphogenesisKineticsMode::Schnakenberg => (PdeComputeKinetics::Schnakenberg, p.a, p.b),
            MorphogenesisKineticsMode::GrayScott => (PdeComputeKinetics::GrayScott, ks.f, ks.k),
        };
        emit_pde_stencil_compute_wgsl(&PdeStencilComputeSpec {
            grid_dims: [self.width.max(4) as u32, self.height.max(4) as u32],
            dt: p.dt,
            d_u: p.d_u,
            d_v: p.d_v,
            rate_primary,
            rate_secondary,
            kinetics,
        })
    }
}
