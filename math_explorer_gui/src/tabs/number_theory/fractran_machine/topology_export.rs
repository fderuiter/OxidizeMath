//! Minsky counter-machine state-transition topology export for [`super::FractranMachineTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::FractranMachineTool;
use math_explorer::pure_math::graph_theory::graph_io::{
    GraphTopologyDocument, TopologyEdge, TopologyEdgeStyle, TopologyNode, TopologyNodeShape,
};
use std::f64::consts::TAU;

impl FractranMachineTool {
    pub(super) fn build_fractran_topology(&self) -> GraphTopologyDocument {
        let mut doc = GraphTopologyDocument::new(format!("Fractran_{}", self.preset.label()), true);
        let decoded = self.minsky.decode(&self.vm, &self.registers);
        doc.attributes
            .insert("preset".into(), self.preset.label().to_string());
        doc.attributes
            .insert("initial_value".into(), self.initial_value.to_string());
        doc.attributes
            .insert("step_count".into(), self.step_count.to_string());
        let counters = decoded
            .counters
            .iter()
            .map(|(p, e)| format!("r{p}={e}"))
            .collect::<Vec<_>>()
            .join(", ");
        doc.attributes.insert("counters".into(), counters);

        self.append_minsky_state_nodes(&mut doc, &decoded.active_state_primes);
        self.append_minsky_transition_edges(&mut doc);
        doc
    }

    fn append_minsky_state_nodes(
        &self,
        doc: &mut GraphTopologyDocument,
        active_primes: &[(u64, u32)],
    ) {
        let total = 1 + self.minsky.state_primes.len();
        for idx in 0..total {
            let state_prime = if idx == 0 {
                None
            } else {
                Some(self.minsky.state_primes[idx - 1])
            };
            let (x, y) = if total == 1 {
                (300.0, 220.0)
            } else {
                let angle = -TAU * 0.25 + TAU * (idx as f64) / (total as f64);
                (300.0 + 180.0 * angle.cos(), 220.0 + 130.0 * angle.sin())
            };
            let is_active = state_prime.map_or(active_primes.is_empty(), |p| {
                active_primes.iter().any(|&(ap, _)| ap == p)
            });
            let id = state_prime.map_or_else(|| "q0".to_string(), |p| format!("q{p}"));
            let label = state_prime.map_or_else(|| "q0 (rest)".to_string(), |p| format!("q{p}"));
            let node = TopologyNode::new(id, label)
                .with_position(x, y)
                .with_shape(if is_active {
                    TopologyNodeShape::DoubleCircle
                } else {
                    TopologyNodeShape::Circle
                })
                .with_color(if is_active { "#16A34A" } else { "#4B5563" })
                .with_attribute("active", if is_active { "true" } else { "false" });
            doc.nodes.push(node);
        }
    }

    fn append_minsky_transition_edges(&self, doc: &mut GraphTopologyDocument) {
        for tr in &self.transitions {
            let src = tr
                .from_state
                .map_or_else(|| "q0".to_string(), |p| format!("q{p}"));
            let dst = tr
                .to_state
                .map_or_else(|| "q0".to_string(), |p| format!("q{p}"));
            let fired = self.last_fired_rule == Some(tr.rule_index);
            let label = self.program.fractions.get(tr.rule_index).map_or_else(
                || format!("R{}", tr.rule_index + 1),
                |f| {
                    format!(
                        "R{}: {}/{}",
                        tr.rule_index + 1,
                        f.0.numerator(),
                        f.0.denominator()
                    )
                },
            );
            let mut edge = TopologyEdge::new(src, dst, (tr.rule_index + 1) as f64)
                .with_label(label)
                .with_attribute("rule_index", (tr.rule_index + 1).to_string())
                .with_attribute("fired", if fired { "true" } else { "false" });
            if fired {
                edge = edge
                    .with_style(TopologyEdgeStyle::Thick)
                    .with_color("#2563EB");
            }
            doc.edges.push(edge);
        }
    }
}
