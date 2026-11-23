//! Typed tabular exports for [`super::DoseCalculationTool`] across point-source PSF and 4-beam IMRT modes.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::imrt::{voxel_coord_cm, IMRT_BEAM_COUNT, IMRT_GRID_DIM, IMRT_PHANTOM_SIZE_CM};
use super::DoseCalculationTool;
use crate::framework::{compute_matrix_stride, ExportCell, ExportTable};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::medical::dose::kernel::{DoseKernel, ExponentialKernel};

impl DoseCalculationTool {
    pub(super) fn build_export_tables(&self) -> Vec<ExportTable> {
        if self.imrt_mode {
            self.build_imrt_export_tables()
        } else {
            vec![self.build_point_source_export_table()]
        }
    }

    fn build_point_source_export_table(&self) -> ExportTable {
        let width = self.width.max(1);
        let height = self.height.max(1);
        let (r_stride, c_stride) = compute_matrix_stride(height, width);
        let kernel = ExponentialKernel::new(self.amplitude, self.beta);
        let max_dose = kernel.value_at(0.1).unwrap_or(10.0);
        let dx = IMRT_PHANTOM_SIZE_CM / width as f64;
        let dy = IMRT_PHANTOM_SIZE_CM / height as f64;
        let mut rows = Vec::new();
        for y in (0..height).step_by(r_stride) {
            let py = (y as f64 - height as f64 / 2.0) * dy;
            for x in (0..width).step_by(c_stride) {
                let px = (x as f64 - width as f64 / 2.0) * dx;
                let r = px.hypot(py);
                let dose = if r < 1e-3 {
                    max_dose
                } else {
                    kernel.value_at(r).unwrap_or(0.0)
                };
                rows.push(vec![
                    ExportCell::UInt(y as u64),
                    ExportCell::UInt(x as u64),
                    ExportCell::Float(px),
                    ExportCell::Float(py),
                    ExportCell::Float(dose),
                ]);
            }
        }
        ExportTable::new(
            "Dose Field",
            vec![
                "row".into(),
                "col".into(),
                "x_cm".into(),
                "y_cm".into(),
                "dose".into(),
            ],
            rows,
        )
    }

    fn build_imrt_export_tables(&self) -> Vec<ExportTable> {
        let (r_stride, c_stride) = compute_matrix_stride(IMRT_GRID_DIM, IMRT_GRID_DIM);
        let mut dose_rows = Vec::new();
        for y in (0..IMRT_GRID_DIM).step_by(r_stride) {
            let py = voxel_coord_cm(y);
            for x in (0..IMRT_GRID_DIM).step_by(c_stride) {
                let px = voxel_coord_cm(x);
                let dose = self.imrt_workspace.dose_vec[flatten_2d_index(x, y, IMRT_GRID_DIM)];
                dose_rows.push(vec![
                    ExportCell::UInt(y as u64),
                    ExportCell::UInt(x as u64),
                    ExportCell::Float(px),
                    ExportCell::Float(py),
                    ExportCell::Float(dose),
                ]);
            }
        }
        let beam_rows: Vec<Vec<ExportCell>> = (0..IMRT_BEAM_COUNT)
            .map(|b| {
                vec![
                    ExportCell::UInt((b + 1) as u64),
                    ExportCell::Float(self.beam_angles_deg[b]),
                    ExportCell::Float(self.beam_weights[b]),
                ]
            })
            .collect();
        vec![
            ExportTable::new(
                "IMRT Dose Field",
                vec![
                    "row".into(),
                    "col".into(),
                    "x_cm".into(),
                    "y_cm".into(),
                    "dose_gy".into(),
                ],
                dose_rows,
            ),
            ExportTable::new(
                "IMRT Beams",
                vec!["beam".into(), "angle_deg".into(), "weight".into()],
                beam_rows,
            ),
        ]
    }
}
