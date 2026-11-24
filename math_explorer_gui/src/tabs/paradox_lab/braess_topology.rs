//! Wheatstone routing network [`GraphTopologyDocument`] export for [`super::braess::BraessNetworkParadoxTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::braess::BraessNetworkParadoxTool;
use math_explorer::pure_math::graph_theory::interchange::{
    GraphTopologyDocument, TopologyEdge, TopologyEdgeStyle, TopologyNode, TopologyNodeShape,
};

impl BraessNetworkParadoxTool {
    pub(super) fn build_braess_topology(&self) -> Result<GraphTopologyDocument, String> {
        let s = &self.state;
        let m = &self.metrics;
        let mut doc = GraphTopologyDocument::new("BraessWheatstoneNetwork", true);
        for (k, v) in [
            ("demand", s.demand),
            ("wardrop_latency", m.wardrop_latency),
            ("pigou_latency", m.pigou_latency),
            ("price_of_anarchy", m.price_of_anarchy),
            ("optimal_toll", m.optimal_toll),
        ] {
            doc.attributes.insert(k.to_string(), format!("{v:.4}"));
        }
        doc.nodes = braess_wheatstone_nodes();
        self.append_braess_edges(&mut doc);
        doc.validate().map_err(|e| e.to_string())?;
        Ok(doc)
    }

    fn append_braess_edges(&self, doc: &mut GraphTopologyDocument) {
        let s = &self.state;
        let [f_up, f_low, f_zig] = self.metrics.wardrop_flow;
        let (f_sa, f_bt) = (f_up + f_zig, f_low + f_zig);
        let slope_label = |f: f64| format!("f={f:.2}, c={:.2}f", s.congestion_slope);
        let bypass_label = |f: f64| format!("f={f:.2}, c={:.2}", s.bypass_latency);

        push_braess_outer_edge(
            doc,
            "S",
            "A",
            f_sa,
            s.congestion_slope * f_sa,
            slope_label(f_sa),
            true,
        );
        push_braess_outer_edge(
            doc,
            "A",
            "T",
            f_up,
            s.bypass_latency,
            bypass_label(f_up),
            false,
        );
        push_braess_outer_edge(
            doc,
            "S",
            "B",
            f_low,
            s.bypass_latency,
            bypass_label(f_low),
            false,
        );
        push_braess_outer_edge(
            doc,
            "B",
            "T",
            f_bt,
            s.congestion_slope * f_bt,
            slope_label(f_bt),
            true,
        );
        doc.edges.push(self.braess_shortcut_edge(f_zig));
    }

    fn braess_shortcut_edge(&self, f_zig: f64) -> TopologyEdge {
        let s = &self.state;
        let (style, hex, label) = if s.shortcut_open {
            (
                TopologyEdgeStyle::Thick,
                "#F59E0B",
                format!(
                    "f={f_zig:.2}, c={:.2}+tau={:.2}",
                    s.shortcut_cost, s.shortcut_toll
                ),
            )
        } else {
            (TopologyEdgeStyle::Dashed, "#6B7280", "closed".to_string())
        };
        TopologyEdge::new("A", "B", s.shortcut_cost + s.shortcut_toll)
            .with_label(label)
            .with_style(style)
            .with_color(hex)
            .with_attribute("shortcut_open", s.shortcut_open.to_string())
            .with_attribute("wardrop_flow", format!("{f_zig:.4}"))
            .with_attribute("toll", format!("{:.4}", s.shortcut_toll))
    }
}

fn push_braess_outer_edge(
    doc: &mut GraphTopologyDocument,
    u: &str,
    v: &str,
    flow: f64,
    latency: f64,
    label: String,
    congested: bool,
) {
    let (style, hex) = if congested {
        (TopologyEdgeStyle::Thick, "#EF4444")
    } else {
        (TopologyEdgeStyle::Solid, "#3B82F6")
    };
    doc.edges.push(
        TopologyEdge::new(u, v, latency)
            .with_label(label)
            .with_style(style)
            .with_color(hex)
            .with_attribute("wardrop_flow", format!("{flow:.4}"))
            .with_attribute("latency", format!("{latency:.4}")),
    );
}

fn braess_wheatstone_nodes() -> Vec<TopologyNode> {
    [
        (
            "S",
            "Source S",
            100.0,
            200.0,
            TopologyNodeShape::DoubleCircle,
            "#2563EB",
        ),
        (
            "A",
            "Node A",
            280.0,
            90.0,
            TopologyNodeShape::Circle,
            "#0EA5E9",
        ),
        (
            "B",
            "Node B",
            280.0,
            310.0,
            TopologyNodeShape::Circle,
            "#0EA5E9",
        ),
        (
            "T",
            "Sink T",
            460.0,
            200.0,
            TopologyNodeShape::DoubleCircle,
            "#16A34A",
        ),
    ]
    .into_iter()
    .map(|(id, label, x, y, shape, hex)| {
        TopologyNode::new(id, label)
            .with_position(x, y)
            .with_shape(shape)
            .with_color(hex)
    })
    .collect()
}
