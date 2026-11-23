//! Directed 4-node Wheatstone congestion network topology export for
//! [`super::braess::BraessNetworkParadoxTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::braess::BraessNetworkParadoxTool;
use math_explorer::pure_math::graph_theory::graph_io::{
    GraphTopologyDocument, TopologyEdge, TopologyEdgeStyle, TopologyNode, TopologyNodeShape,
};

impl BraessNetworkParadoxTool {
    pub(super) fn build_braess_topology(&self) -> GraphTopologyDocument {
        let mut doc = GraphTopologyDocument::new("BraessWheatstoneNetwork", true);
        let s = &self.state;
        let m = self.metrics;
        for (k, v) in [
            ("demand", format!("{:.4}", s.demand)),
            ("wardrop_latency", format!("{:.4}", m.wardrop_latency)),
            ("closed_latency", format!("{:.4}", m.closed_latency)),
            ("pigou_latency", format!("{:.4}", m.pigou_latency)),
            ("optimal_toll", format!("{:.4}", m.optimal_toll)),
            ("price_of_anarchy", format!("{:.4}", m.price_of_anarchy)),
            ("nash_fixed_point", m.nash_fixed_point.to_string()),
        ] {
            doc.attributes.insert(k.to_string(), v);
        }
        append_wheatstone_nodes(&mut doc);
        self.append_wheatstone_edges(&mut doc);
        doc
    }

    fn append_wheatstone_edges(&self, doc: &mut GraphTopologyDocument) {
        let s = &self.state;
        let w = self.metrics.wardrop_flow;
        let (f_sa, f_at, f_sb, f_bt) = (w[0] + w[2], w[0], w[1], w[1] + w[2]);
        let f_ab = if s.shortcut_open { w[2] } else { 0.0 };
        let specs = [
            (
                "S",
                "A",
                f_sa,
                s.congestion_slope * f_sa,
                format!("{:.2}*f", s.congestion_slope),
                true,
            ),
            (
                "A",
                "T",
                f_at,
                s.bypass_latency,
                format!("{:.2}", s.bypass_latency),
                true,
            ),
            (
                "S",
                "B",
                f_sb,
                s.bypass_latency,
                format!("{:.2}", s.bypass_latency),
                true,
            ),
            (
                "B",
                "T",
                f_bt,
                s.congestion_slope * f_bt,
                format!("{:.2}*f", s.congestion_slope),
                true,
            ),
            (
                "A",
                "B",
                f_ab,
                s.shortcut_cost + s.shortcut_toll,
                format!("{:.2}+{:.2}", s.shortcut_cost, s.shortcut_toll),
                s.shortcut_open,
            ),
        ];
        for (u, v, flow, lat, formula, open) in specs {
            let mut edge = TopologyEdge::new(u, v, flow)
                .with_label(format!("f={flow:.2}, l={lat:.2}"))
                .with_attribute("wardrop_flow", format!("{flow:.4}"))
                .with_attribute("edge_latency", format!("{lat:.4}"))
                .with_attribute("cost_formula", formula)
                .with_attribute("open", open.to_string());
            if !open {
                edge = edge
                    .with_style(TopologyEdgeStyle::Dashed)
                    .with_color("#6B7280");
            }
            doc.edges.push(edge);
        }
    }
}

fn append_wheatstone_nodes(doc: &mut GraphTopologyDocument) {
    for (id, label, x, y, double) in [
        ("S", "S (Origin)", 90.0, 220.0, true),
        ("A", "A (Upper)", 280.0, 100.0, false),
        ("B", "B (Lower)", 280.0, 340.0, false),
        ("T", "T (Destination)", 470.0, 220.0, true),
    ] {
        let shape = if double {
            TopologyNodeShape::DoubleCircle
        } else {
            TopologyNodeShape::Circle
        };
        doc.nodes.push(
            TopologyNode::new(id, label)
                .with_position(x, y)
                .with_shape(shape),
        );
    }
}
