#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::MorphogenesisTool;
use crate::async_sim::SimulationRunner;
use crate::framework::{compute_matrix_stride, ExportCell, ExportTable};

pub(super) fn build_morphogenesis_tables(tool: &MorphogenesisTool) -> Vec<ExportTable> {
    let width = tool.width.max(1);
    let height = tool.height.max(1);
    let u_grid: Vec<f64> = tool
        .worker
        .latest_snapshot()
        .and_then(|snap| {
            snap.structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<Vec<f64>>())
                .filter(|v| v.len() == width * height)
                .cloned()
        })
        .unwrap_or_else(|| {
            let runner = MorphogenesisTool::create_runner(&tool.params, width, height);
            let snap = runner.get_snapshot();
            snap.structured_data
                .and_then(|d| d.downcast::<Vec<f64>>().ok())
                .map_or_else(Vec::new, |b| *b)
        });

    let (r_stride, c_stride) = compute_matrix_stride(height, width);
    let mut rows = Vec::new();
    if u_grid.len() == width * height {
        for r in (0..height).step_by(r_stride) {
            for c in (0..width).step_by(c_stride) {
                rows.push(vec![
                    ExportCell::UInt(r as u64),
                    ExportCell::UInt(c as u64),
                    ExportCell::Float(u_grid[r * width + c]),
                ]);
            }
        }
    }
    let (a, b, d_u, d_v, dt) = {
        let p = tool.params.read().unwrap_or_else(|e| e.into_inner());
        (p.a, p.b, p.d_u, p.d_v, p.dt)
    };
    vec![
        ExportTable::new(
            "Activator Concentration",
            vec!["row".into(), "col".into(), "u_concentration".into()],
            rows,
        ),
        ExportTable::new(
            "Reaction Parameters",
            vec![
                "a".into(),
                "b".into(),
                "d_u".into(),
                "d_v".into(),
                "dt".into(),
            ],
            vec![vec![
                ExportCell::Float(a),
                ExportCell::Float(b),
                ExportCell::Float(d_u),
                ExportCell::Float(d_v),
                ExportCell::Float(dt),
            ]],
        ),
    ]
}
