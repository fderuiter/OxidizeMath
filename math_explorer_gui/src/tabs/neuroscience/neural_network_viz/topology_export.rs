//! Directed Hodgkin-Huxley and Kuramoto circuit topology export and theory metadata
//! for [`super::NeuralNetworkVizTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{NeuralDynamicsMode, NeuralNetworkVizTool};

#[cfg(feature = "pure_math")]
use math_explorer::pure_math::graph_theory::graph_io::{
    GraphTopologyDocument, TopologyEdge, TopologyEdgeStyle, TopologyNode, TopologyNodeShape,
};

#[cfg(feature = "pure_math")]
impl NeuralNetworkVizTool {
    pub(super) fn build_neural_topology(&self) -> GraphTopologyDocument {
        let mut doc = GraphTopologyDocument::new("NeuralNetworkCircuit", true);
        let is_kuramoto = self.mode == NeuralDynamicsMode::Kuramoto;
        doc.attributes.insert(
            "model".into(),
            if is_kuramoto {
                "Kuramoto"
            } else {
                "Hodgkin-Huxley"
            }
            .into(),
        );
        doc.attributes
            .insert("dt_ms".into(), format!("{:.4}", self.dt));
        doc.attributes
            .insert("neuron_count".into(), self.neurons.len().to_string());
        if is_kuramoto {
            let (r, psi) = self.kuramoto.order_parameter();
            doc.attributes.insert(
                "coupling_k".into(),
                format!("{:.4}", self.kuramoto.coupling()),
            );
            doc.attributes
                .insert("order_parameter_r".into(), format!("{r:.4}"));
            doc.attributes
                .insert("mean_phase_rad".into(), format!("{psi:.4}"));
        }
        self.populate_neural_nodes(&mut doc, is_kuramoto);
        self.populate_neural_edges(&mut doc);
        doc
    }

    fn populate_neural_nodes(&self, doc: &mut GraphTopologyDocument, is_kuramoto: bool) {
        let phases = self.kuramoto.phases();
        for (i, neuron) in self.neurons.iter().enumerate() {
            let v = neuron.v();
            let pos = self.positions.get(i).copied().unwrap_or([0.0, 0.0]);
            let ext = self.external_input.get(i).copied().unwrap_or(0.0);
            let theta = phases.get(i).copied().unwrap_or(0.0);
            let spiking = !is_kuramoto && v > 0.0;
            let label = if is_kuramoto {
                format!("N{i} ({theta:.2} rad)")
            } else {
                format!("N{i} ({v:.0} mV)")
            };
            let mut node = TopologyNode::new(format!("N{i}"), label)
                .with_position(300.0 + f64::from(pos[0]), 220.0 + f64::from(pos[1]))
                .with_shape(if spiking {
                    TopologyNodeShape::DoubleCircle
                } else {
                    TopologyNodeShape::Circle
                })
                .with_color(if spiking { "#DC2626" } else { "#2563EB" })
                .with_attribute("voltage_mv", format!("{v:.3}"))
                .with_attribute("external_input_ua", format!("{ext:.3}"));
            if is_kuramoto {
                node = node.with_attribute("phase_rad", format!("{theta:.4}"));
            }
            doc.nodes.push(node);
        }
    }

    fn populate_neural_edges(&self, doc: &mut GraphTopologyDocument) {
        let n = self.neurons.len();
        for pre in 0..n {
            for post in 0..n {
                let w = self
                    .weights
                    .get(post)
                    .and_then(|r| r.get(pre))
                    .copied()
                    .unwrap_or(0.0);
                if w.abs() <= 1e-12 {
                    continue;
                }
                let excitatory = w > 0.0;
                let edge = TopologyEdge::new(format!("N{pre}"), format!("N{post}"), w)
                    .with_label(format!("{w:+.2}"))
                    .with_color(if excitatory { "#16A34A" } else { "#DC2626" })
                    .with_style(if excitatory {
                        TopologyEdgeStyle::Solid
                    } else {
                        TopologyEdgeStyle::Dashed
                    })
                    .with_attribute(
                        "synapse_type",
                        if excitatory {
                            "excitatory"
                        } else {
                            "inhibitory"
                        },
                    );
                doc.edges.push(edge);
            }
        }
    }
}

impl scientific_metadata::theory::TheoryDescribable for NeuralNetworkVizTool {
    fn theory_description(&self) -> String {
        match self.mode {
            NeuralDynamicsMode::HodgkinHuxley => format!(
                "Coupled network of {} biophysical Hodgkin-Huxley conductance-based neurons: membrane potential dynamics $C_m \\frac{{dV_i}}{{dt}} = I_{{ext,i}} - I_{{ion}}(V_i) + \\sum_j W_{{ij}} s_j(t)$ with synaptic weight matrix $W$.",
                self.neurons.len()
            ),
            NeuralDynamicsMode::Kuramoto => format!(
                "Coupled network of {} Kuramoto phase oscillators: $\\frac{{d\\theta_i}}{{dt}} = \\omega_i + \\frac{{K}}{{N}} \\sum_j A_{{ji}} \\sin(\\theta_j - \\theta_i)$ with complex synchronization order parameter $r(t) e^{{i\\psi(t)}} = \\frac{{1}}{{N}} \\sum_j e^{{i\\theta_j}}$.",
                self.kuramoto.num_nodes()
            ),
        }
    }
    fn phonetic_description(&self) -> String {
        "Coupled Hodgkin-Huxley neural network and Kuramoto phase synchronization theory".into()
    }
    fn theory_citation(&self) -> String {
        "Hodgkin & Huxley (1952); Chemical Oscillations, Waves, and Turbulence (Kuramoto, 1984); Theoretical Neuroscience (Dayan & Abbott, 2001)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

#[cfg(all(test, feature = "pure_math"))]
mod tests {
    use super::*;
    use crate::app::dataflow_bus::GraphTopologyPayload;
    use crate::framework::InteractiveTool;

    #[test]
    fn test_kuramoto_dataflow_topology_export_includes_synchronization_metadata() {
        let mut tool = NeuralNetworkVizTool::default();
        let payload = GraphTopologyPayload::sanitized(
            7,
            vec![[0.0, 0.0], [40.0, 0.0], [20.0, 35.0]],
            vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)],
            false,
        );
        tool.ingest_graph_topology(&payload);
        tool.mode = NeuralDynamicsMode::Kuramoto;
        tool.step_once();

        let doc = tool
            .export_topology_document()
            .expect("topology capable")
            .expect("valid topology document");
        assert_eq!(doc.nodes.len(), 3);
        assert_eq!(
            doc.attributes.get("model").map(String::as_str),
            Some("Kuramoto")
        );
        assert!(doc.attributes.contains_key("coupling_k"));
        assert!(doc.attributes.contains_key("order_parameter_r"));
        assert!(doc.attributes.contains_key("mean_phase_rad"));
        assert!(doc
            .nodes
            .iter()
            .all(|n| n.attributes.contains_key("phase_rad")));
    }
}
