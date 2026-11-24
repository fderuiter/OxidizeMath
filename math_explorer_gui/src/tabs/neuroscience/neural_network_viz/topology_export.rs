//! Synaptic and Kuramoto topology export for [`super::NeuralNetworkVizTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{NeuralDynamicsMode, NeuralNetworkVizTool};
use math_explorer::pure_math::graph_theory::interchange::{
    GraphTopologyDocument, TopologyEdge, TopologyEdgeStyle, TopologyNode, TopologyNodeShape,
};

impl NeuralNetworkVizTool {
    pub(super) fn build_neural_topology(&self) -> Result<GraphTopologyDocument, String> {
        if self.mode == NeuralDynamicsMode::Kuramoto {
            return self.build_kuramoto_topology();
        }
        let mut doc = GraphTopologyDocument::new("HodgkinHuxleyNetwork", true);
        doc.attributes
            .insert("mode".to_string(), "hodgkin_huxley".to_string());
        doc.attributes
            .insert("dt".to_string(), format!("{:.4}", self.dt));
        if self.upstream_linked {
            doc.attributes
                .insert("dataflow_linked".to_string(), "true".to_string());
        }

        for (i, neuron) in self.neurons.iter().enumerate() {
            let pos = self.positions.get(i).copied().unwrap_or([0.0, 0.0]);
            let v = neuron.v();
            let ext = self.external_input.get(i).copied().unwrap_or(0.0);
            let fill = if v > 0.0 {
                "#F59E0B"
            } else if v > -55.0 {
                "#3B82F6"
            } else {
                "#1E3A8A"
            };
            doc.nodes.push(
                TopologyNode::new(format!("N{i}"), format!("N{i} ({v:.1}mV)"))
                    .with_position(f64::from(pos[0]) + 260.0, f64::from(pos[1]) + 220.0)
                    .with_shape(TopologyNodeShape::Circle)
                    .with_color(fill)
                    .with_attribute("voltage_mv", format!("{v:.4}"))
                    .with_attribute("external_input", format!("{ext:.4}")),
            );
        }

        for pre in 0..self.neurons.len() {
            let start = self.csr_row_ptr.get(pre).copied().unwrap_or(0);
            let end = self.csr_row_ptr.get(pre + 1).copied().unwrap_or(start);
            for idx in start..end {
                let post = self.csr_col_idx[idx];
                let w = self.csr_weights[idx];
                let mut edge = TopologyEdge::new(format!("N{pre}"), format!("N{post}"), w)
                    .with_label(format!("{w:+.2}"));
                if w < 0.0 {
                    edge.style = TopologyEdgeStyle::Dashed;
                    edge.color_hex = Some("#EF4444".to_string());
                    edge.attributes
                        .insert("synapse".to_string(), "inhibitory".to_string());
                } else {
                    edge.color_hex = Some("#10B981".to_string());
                    edge.attributes
                        .insert("synapse".to_string(), "excitatory".to_string());
                }
                doc.edges.push(edge);
            }
        }

        doc.validate().map_err(|e| e.to_string())?;
        Ok(doc)
    }

    fn build_kuramoto_topology(&self) -> Result<GraphTopologyDocument, String> {
        let mut doc = GraphTopologyDocument::new("KuramotoOscillatorNetwork", false);
        let (r, psi) = self.kuramoto.order_parameter();
        doc.attributes
            .insert("mode".to_string(), "kuramoto".to_string());
        doc.attributes.insert(
            "coupling_k".to_string(),
            format!("{:.4}", self.kuramoto.coupling()),
        );
        doc.attributes
            .insert("order_r".to_string(), format!("{r:.4}"));
        doc.attributes
            .insert("mean_phase_psi".to_string(), format!("{psi:.4}"));

        let phases = self.kuramoto.phases();
        let n = phases.len();
        for (i, &theta) in phases.iter().enumerate() {
            let pos = self.positions.get(i).copied().unwrap_or_else(|| {
                let ang = (i as f32) * std::f32::consts::TAU / (n.max(1) as f32);
                [100.0 * ang.cos(), 100.0 * ang.sin()]
            });
            doc.nodes.push(
                TopologyNode::new(format!("O{i}"), format!("O{i} ({theta:.2}rad)"))
                    .with_position(f64::from(pos[0]) + 260.0, f64::from(pos[1]) + 220.0)
                    .with_color("#6366F1")
                    .with_attribute("phase_rad", format!("{theta:.6}")),
            );
        }

        let k_eff = self.kuramoto.coupling() / (n.max(1) as f64);
        if self.csr_col_idx.is_empty() {
            for i in 0..n {
                for j in (i + 1)..n {
                    doc.edges
                        .push(TopologyEdge::new(format!("O{i}"), format!("O{j}"), k_eff));
                }
            }
        } else {
            for pre in 0..n.min(self.csr_row_ptr.len().saturating_sub(1)) {
                for idx in self.csr_row_ptr[pre]..self.csr_row_ptr[pre + 1] {
                    let post = self.csr_col_idx[idx];
                    if pre < post && post < n {
                        doc.edges.push(TopologyEdge::new(
                            format!("O{pre}"),
                            format!("O{post}"),
                            k_eff,
                        ));
                    }
                }
            }
        }

        doc.validate().map_err(|e| e.to_string())?;
        Ok(doc)
    }
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::*;
    use crate::framework::InteractiveTool;
    use math_explorer::pure_math::graph_theory::interchange::TopologyFormat;

    #[test]
    fn test_neural_topology_exports_hodgkin_huxley_and_kuramoto_modes() {
        let mut tool = NeuralNetworkVizTool::default();
        let hh_doc = tool
            .export_topology_document()
            .expect("HH topology document");
        assert!(hh_doc.directed);
        assert_eq!(hh_doc.nodes.len(), 5);
        assert_eq!(hh_doc.edges.len(), 10);
        let dot = hh_doc.serialize(TopologyFormat::Dot).expect("DOT");
        assert!(dot.contains("inhibitory") && dot.contains("excitatory"));

        tool.mode = NeuralDynamicsMode::Kuramoto;
        tool.step_once();
        let kur_doc = tool
            .export_topology_document()
            .expect("Kuramoto topology document");
        assert!(!kur_doc.directed);
        assert!(kur_doc.attributes.contains_key("order_r"));
        let tikz = kur_doc.serialize(TopologyFormat::Tikz).expect("TikZ");
        assert!(tikz.contains("\\begin{tikzpicture}"));
    }
}
