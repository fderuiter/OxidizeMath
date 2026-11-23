//! Directed Hodgkin-Huxley synaptic circuit topology export for [`super::NeuralNetworkVizTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::NeuralNetworkVizTool;
use math_explorer::pure_math::graph_theory::graph_io::{
    GraphTopologyDocument, TopologyEdge, TopologyEdgeStyle, TopologyNode, TopologyNodeShape,
};

impl NeuralNetworkVizTool {
    pub(super) fn build_neural_topology(&self) -> GraphTopologyDocument {
        let mut doc = GraphTopologyDocument::new("NeuralNetworkCircuit", true);
        doc.attributes
            .insert("model".into(), "Hodgkin-Huxley".into());
        doc.attributes
            .insert("dt_ms".into(), format!("{:.4}", self.dt));
        doc.attributes
            .insert("neuron_count".into(), self.neurons.len().to_string());

        for (i, neuron) in self.neurons.iter().enumerate() {
            let v = neuron.v();
            let pos = self.positions.get(i).copied().unwrap_or([0.0, 0.0]);
            let ext = self.external_input.get(i).copied().unwrap_or(0.0);
            let spiking = v > 0.0;
            let node = TopologyNode::new(format!("N{i}"), format!("N{i} ({v:.0} mV)"))
                .with_position(300.0 + f64::from(pos[0]), 220.0 + f64::from(pos[1]))
                .with_shape(if spiking {
                    TopologyNodeShape::DoubleCircle
                } else {
                    TopologyNodeShape::Circle
                })
                .with_color(if spiking { "#DC2626" } else { "#2563EB" })
                .with_attribute("voltage_mv", format!("{v:.3}"))
                .with_attribute("external_input_ua", format!("{ext:.3}"));
            doc.nodes.push(node);
        }

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
        doc
    }
}
