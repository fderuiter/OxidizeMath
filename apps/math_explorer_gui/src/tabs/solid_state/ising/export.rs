#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::SimulationRunner;
use crate::framework::{ExportCell, ExportTable};
use math_commons::math_kernel::types::flatten_2d_index;

use super::IsingModelTool;

impl IsingModelTool {
    fn build_observables_table(&self) -> ExportTable {
        let headers = vec![
            "temperature".to_string(),
            "j_coupling".to_string(),
            "h_field".to_string(),
            "magnetization".to_string(),
            "uniform_magnetization".to_string(),
            "staggered_magnetization".to_string(),
            "energy".to_string(),
            "cooperator_fraction".to_string(),
        ];
        let row = vec![
            ExportCell::Float(self.temperature),
            ExportCell::Float(self.j_coupling),
            ExportCell::Float(self.h_field),
            ExportCell::Float(self.magnetization),
            ExportCell::Float(self.uniform_magnetization),
            ExportCell::Float(self.staggered_magnetization),
            ExportCell::Float(self.energy),
            ExportCell::Float(self.cooperator_fraction()),
        ];
        ExportTable::new("Observables", headers, vec![row])
    }

    fn build_graph_spins_table(&self) -> Option<ExportTable> {
        if !self.is_graph_mode() {
            return None;
        }
        let gs = self.graph_system.as_ref()?;
        let spins = gs.spins();
        if spins.is_empty() {
            return None;
        }
        let rows = spins
            .iter()
            .enumerate()
            .map(|(idx, &s)| vec![ExportCell::Int(idx as i64), ExportCell::Int(i64::from(s))])
            .collect();
        Some(ExportTable::new(
            "Graph Spins",
            vec!["node_id".to_string(), "spin".to_string()],
            rows,
        ))
    }

    fn build_lattice_spins_table(&self) -> ExportTable {
        if let Some(snap) = self.worker.latest_snapshot() {
            if let Some(encoded) = snap
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<Vec<i8>>())
                .filter(|v| v.len() == snap.width * snap.height && snap.width > 0)
            {
                return Self::encoded_lattice_to_table(snap.width, snap.height, encoded);
            }
        }
        let fallback = Self::build_runner(&self.params).get_snapshot();
        let empty = Vec::new();
        let encoded = fallback
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<Vec<i8>>())
            .unwrap_or(&empty);
        Self::encoded_lattice_to_table(fallback.width, fallback.height, encoded)
    }

    fn encoded_lattice_to_table(width: usize, height: usize, encoded: &[i8]) -> ExportTable {
        ExportTable::from_bounded_grid_2d(
            "Spin Lattice",
            width,
            height,
            vec![
                "x".to_string(),
                "y".to_string(),
                "spin".to_string(),
                "pinned".to_string(),
            ],
            |x, y| {
                let raw = encoded
                    .get(flatten_2d_index(x, y, width))
                    .copied()
                    .unwrap_or(1);
                let spin = if raw < 0 { -1_i64 } else { 1_i64 };
                let pinned = raw.abs() > 1;
                vec![
                    ExportCell::Int(x as i64),
                    ExportCell::Int(y as i64),
                    ExportCell::Int(spin),
                    ExportCell::Bool(pinned),
                ]
            },
        )
    }

    pub(super) fn build_ising_export_tables(&self) -> Vec<ExportTable> {
        let obs = self.build_observables_table();
        let spins = self
            .build_graph_spins_table()
            .unwrap_or_else(|| self.build_lattice_spins_table());
        vec![obs, spins]
    }
}
