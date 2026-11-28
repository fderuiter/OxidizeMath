#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ExportCell, ExportTable};
use crate::tabs::medical::imrt::{IMRT_BEAM_COUNT, IMRT_GRID_DIM, IMRT_PHANTOM_SIZE_CM};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::medical::dose::kernel::{DoseKernel, ExponentialKernel};

use super::canvas::active_imrt_dose_slice;
use super::DoseCalculationTool;

fn build_imrt_tables(tool: &DoseCalculationTool) -> Vec<ExportTable> {
    let mut plan_rows = Vec::with_capacity(IMRT_BEAM_COUNT);
    for idx in 0..IMRT_BEAM_COUNT {
        plan_rows.push(vec![
            ExportCell::Int(idx as i64),
            ExportCell::Float(tool.beam_angles_deg[idx]),
            ExportCell::Float(tool.beam_weights[idx]),
            ExportCell::Float(tool.prescription_gy),
            ExportCell::Float(tool.oar_limit_gy),
        ]);
    }
    let plan_table = ExportTable::new(
        "IMRT Beam Plan",
        vec![
            "beam_index".to_string(),
            "gantry_angle_deg".to_string(),
            "beam_weight".to_string(),
            "prescription_gy".to_string(),
            "oar_limit_gy".to_string(),
        ],
        plan_rows,
    );
    let dose_slice = active_imrt_dose_slice(tool);
    let step_cm = IMRT_PHANTOM_SIZE_CM / (IMRT_GRID_DIM as f64);
    let grid_table = ExportTable::from_bounded_grid_2d(
        "IMRT Dose Grid",
        IMRT_GRID_DIM,
        IMRT_GRID_DIM,
        vec![
            "x".to_string(),
            "y".to_string(),
            "x_cm".to_string(),
            "y_cm".to_string(),
            "dose_gy".to_string(),
        ],
        |x, y| {
            let x_cm = (x as f64 - IMRT_GRID_DIM as f64 * 0.5) * step_cm;
            let y_cm = (y as f64 - IMRT_GRID_DIM as f64 * 0.5) * step_cm;
            let dose = dose_slice
                .get(flatten_2d_index(x, y, IMRT_GRID_DIM))
                .copied()
                .unwrap_or(0.0);
            vec![
                ExportCell::Int(x as i64),
                ExportCell::Int(y as i64),
                ExportCell::Float(x_cm),
                ExportCell::Float(y_cm),
                ExportCell::Float(dose),
            ]
        },
    );
    vec![plan_table, grid_table]
}

pub(super) fn build_dose_export_tables(tool: &DoseCalculationTool) -> Vec<ExportTable> {
    if tool.imrt_mode {
        return build_imrt_tables(tool);
    }
    let (width, height) = (tool.width.max(1), tool.height.max(1));
    let kernel = ExponentialKernel::new(tool.amplitude, tool.beta);
    let (dx, dy) = (
        IMRT_PHANTOM_SIZE_CM / width as f64,
        IMRT_PHANTOM_SIZE_CM / height as f64,
    );
    let max_dose = kernel.value_at(0.1).unwrap_or(10.0);
    let table = ExportTable::from_bounded_grid_2d(
        "Dose Kernel Grid",
        width,
        height,
        vec![
            "x".to_string(),
            "y".to_string(),
            "x_cm".to_string(),
            "y_cm".to_string(),
            "radius_cm".to_string(),
            "dose_gy".to_string(),
        ],
        |x, y| {
            let x_cm = (x as f64 - width as f64 * 0.5) * dx;
            let y_cm = (y as f64 - height as f64 * 0.5) * dy;
            let r = x_cm.hypot(y_cm);
            let dose = if r < 1e-3 {
                max_dose
            } else {
                kernel.value_at(r).unwrap_or(0.0)
            };
            vec![
                ExportCell::Int(x as i64),
                ExportCell::Int(y as i64),
                ExportCell::Float(x_cm),
                ExportCell::Float(y_cm),
                ExportCell::Float(r),
                ExportCell::Float(dose),
            ]
        },
    );
    vec![table]
}

impl scientific_metadata::theory::TheoryDescribable for DoseCalculationTool {
    fn theory_description(&self) -> String {
        format!(
            "Radiation therapy dose distribution computed via point spread function (PSF) kernel convolution: $D(\\vec{{r}}) = \\int \\Phi(\\vec{{r}}') K(\\vec{{r}} - \\vec{{r}}') d^3\\vec{{r}}'$, using an exponential kernel $K(r) = \\frac{{A}}{{r^2}} e^{{-\\beta r}}$ with amplitude $A = {:.2}$ and attenuation coefficient $\\beta = {:.2}$, coupled with 4D respiratory tissue-shift accumulation and LINAC beam-loading scaling $E(I_b) = E_0 - R_b I_b$.",
            self.amplitude, self.beta
        )
    }
    fn phonetic_description(&self) -> String {
        "Radiation therapy dose calculation, contour sculpting, and four-dimensional respiratory-gated inverse IMRT optimization".into()
    }
    fn theory_citation(&self) -> String {
        "The Physics of Radiation Therapy (Khan, 2014); Convolution/superposition dose calculation (Mackie et al., 1985); AAPM TG-76 Respiratory Motion Management (Keall et al., 2006)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::from([
            (
                "amplitude".into(),
                "Kernel primary energy fluence scaling factor A".into(),
            ),
            (
                "beta".into(),
                "Radial dose attenuation and scattering decay coefficient beta".into(),
            ),
        ])
    }
}
