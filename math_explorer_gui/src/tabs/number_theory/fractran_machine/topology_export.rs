//! Minsky counter-machine state graph [`GraphTopologyDocument`] export for [`super::FractranMachineTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::FractranMachineTool;
use math_explorer::pure_math::graph_theory::interchange::{
    GraphTopologyDocument, TopologyEdge, TopologyEdgeStyle, TopologyNode, TopologyNodeShape,
};
use std::collections::HashSet;
use std::f64::consts::TAU;

impl FractranMachineTool {
    pub(super) fn build_fractran_topology(&self) -> Result<GraphTopologyDocument, String> {
        let mut doc = GraphTopologyDocument::new(format!("FRACTRAN_{:?}", self.preset), true);
        doc.attributes
            .insert("step_count".to_string(), self.step_count.to_string());
        doc.attributes
            .insert("initial_value".to_string(), self.initial_value.to_string());
        self.populate_fractran_nodes(&mut doc);
        self.populate_fractran_edges(&mut doc);
        doc.validate().map_err(|e| e.to_string())?;
        Ok(doc)
    }

    fn populate_fractran_nodes(&self, doc: &mut GraphTopologyDocument) {
        let decoded = self.minsky.decode(&self.vm, &self.registers);
        let active_primes: HashSet<u64> = decoded
            .active_state_primes
            .iter()
            .map(|&(p, _)| p)
            .collect();
        let state_primes = &self.minsky.state_primes;
        let total = 1 + state_primes.len();

        for idx in 0..total {
            let (id, label, is_active, prime_attr) = if idx == 0 {
                (
                    "q0".to_string(),
                    "q0 (resting)".to_string(),
                    active_primes.is_empty(),
                    None,
                )
            } else {
                let p = state_primes[idx - 1];
                (
                    format!("q_{p}"),
                    format!("q{p}"),
                    active_primes.contains(&p),
                    Some(p.to_string()),
                )
            };
            let (x, y) = if total == 1 {
                (300.0, 200.0)
            } else {
                let angle = -std::f64::consts::FRAC_PI_2 + (idx as f64) * TAU / (total as f64);
                (300.0 + 130.0 * angle.cos(), 210.0 + 130.0 * angle.sin())
            };
            let mut node = TopologyNode::new(id, label).with_position(x, y);
            if let Some(p_str) = prime_attr {
                node.attributes.insert("prime".to_string(), p_str);
            }
            if is_active {
                node.shape = TopologyNodeShape::DoubleCircle;
                node.fill_hex = Some("#D97706".to_string());
                node.attributes
                    .insert("active".to_string(), "true".to_string());
            } else {
                node.fill_hex = Some("#1D4ED8".to_string());
            }
            doc.nodes.push(node);
        }
    }

    fn populate_fractran_edges(&self, doc: &mut GraphTopologyDocument) {
        let known_primes: HashSet<u64> = self.minsky.state_primes.iter().copied().collect();
        let state_id = |st: Option<u64>| match st {
            Some(p) if known_primes.contains(&p) => format!("q_{p}"),
            _ => "q0".to_string(),
        };
        for tr in &self.transitions {
            let frac_label = self
                .program
                .fractions
                .get(tr.rule_index)
                .map(|f| format!("#{}: {}/{}", tr.rule_index + 1, f.numer(), f.denom()))
                .unwrap_or_else(|| format!("#{}", tr.rule_index + 1));
            let mut edge = TopologyEdge::new(
                state_id(tr.from_state),
                state_id(tr.to_state),
                (tr.rule_index + 1) as f64,
            )
            .with_label(frac_label)
            .with_attribute("rule_index", tr.rule_index.to_string());
            if self.last_fired_rule == Some(tr.rule_index) {
                edge.style = TopologyEdgeStyle::Thick;
                edge.color_hex = Some("#F59E0B".to_string());
            }
            doc.edges.push(edge);
        }
    }
}
