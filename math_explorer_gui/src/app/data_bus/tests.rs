#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::types::{
    adjacency_degree_moments, EpidemicTraceSignal, SharedGraph, SharedNode, SESSION_MAX_BUS_NODES,
    URL_MAX_BUS_NODES,
};
use super::ui::{render_follow_workspace_toggle, render_node_budget_banner};
use super::{
    publish_epidemic_trace, publish_shared_graph, read_epidemic_trace, read_shared_graph,
    validate_embedded_url_bus_fields, WorkspaceDataBus,
};
use crate::app::{contrast_ratio, ThemeMode};
use eframe::egui;
use std::sync::Arc;

#[test]
fn test_workspace_data_bus_monotonic_revisions_and_context_roundtrip() {
    let ctx = egui::Context::default();
    assert!(read_shared_graph(&ctx).is_none());
    assert!(read_epidemic_trace(&ctx).is_none());

    let g1 = publish_shared_graph(
        &ctx,
        SharedGraph {
            revision: 1,
            nodes: vec![
                SharedNode {
                    id: 0,
                    norm_pos: [0.2, 0.3],
                },
                SharedNode {
                    id: 1,
                    norm_pos: [0.8, 0.7],
                },
            ],
            edges: vec![(0, 1, 1.5)],
        },
    );
    assert_eq!(g1.revision, 1);
    assert_eq!(read_shared_graph(&ctx).unwrap().revision, 1);

    // Even if caller passes revision 0, bus advances monotonically to 2
    let g2 = publish_shared_graph(
        &ctx,
        SharedGraph {
            revision: 0,
            nodes: g1.nodes.clone(),
            edges: vec![(0, 1, -2.0)],
        },
    );
    assert_eq!(g2.revision, 2);
    assert_eq!(read_shared_graph(&ctx).unwrap().edges[0].2, -2.0);

    let t1 = publish_epidemic_trace(
        &ctx,
        EpidemicTraceSignal {
            revision: 1,
            population: 50,
            beta: 0.3,
            gamma: 0.1,
            heterogeneous_r0: 3.5,
            susceptible: vec![[0.0, 49.0], [1.0, 45.0]],
            infected: vec![[0.0, 1.0], [1.0, 4.0]],
            recovered: vec![[0.0, 0.0], [1.0, 1.0]],
        },
    );
    assert_eq!(t1.revision, 1);
    assert_eq!(read_epidemic_trace(&ctx).unwrap().population, 50);
}

#[test]
fn test_shared_graph_budget_clamping_and_degree_moments() {
    let nodes: Vec<SharedNode> = (0..40)
        .map(|i| SharedNode {
            id: i,
            norm_pos: [0.5, 0.5],
        })
        .collect();
    let edges = vec![(0, 1, 1.0), (0, 2, 1.0), (0, 35, 1.0), (1, 2, -1.0)];
    let graph = Arc::new(SharedGraph {
        revision: 4,
        nodes,
        edges,
    });

    let (clamped, orig) = graph.clamped_to_budget(32);
    assert_eq!(orig, Some(40));
    assert_eq!(clamped.nodes.len(), 32);
    // Edge (0, 35) dropped; (0, 1) and (1, 2) retained
    assert_eq!(clamped.edges.len(), 3);

    // Positive adjacency ignores negative edge (1, 2, -1.0)
    let small = SharedGraph {
        revision: 1,
        nodes: (0..4)
            .map(|id| SharedNode {
                id,
                norm_pos: [0.5, 0.5],
            })
            .collect(),
        edges: vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, -1.0)],
    };
    let adj = small.positive_adjacency();
    let (mean_k, var_k) = adjacency_degree_moments(&adj);
    // Degrees: node 0 has 3, nodes 1,2,3 have 1 -> mean = 6/4 = 1.5, E[k^2] = 12/4 = 3.0, var = 0.75
    assert!((mean_k - 1.5).abs() < 1e-12);
    assert!((var_k - 0.75).abs() < 1e-12);
}

#[test]
fn test_session_and_url_bounds_reject_unknown_fields_and_oversize() {
    let valid = WorkspaceDataBus {
        shared_graph: Some(Arc::new(SharedGraph {
            revision: 3,
            nodes: vec![SharedNode {
                id: 0,
                norm_pos: [0.5, 0.5],
            }],
            edges: Vec::new(),
        })),
        epidemic_trace: None,
    };
    let json = serde_json::to_string(&valid).unwrap();
    assert_eq!(WorkspaceDataBus::from_session_json(&json).unwrap(), valid);
    assert_eq!(WorkspaceDataBus::from_url_json(&json).unwrap(), valid);

    // Unknown field on WorkspaceDataBus rejected
    let bad_bus = r#"{"shared_graph":null,"unknown_extra":123}"#;
    assert!(WorkspaceDataBus::from_session_json(bad_bus).is_err());
    assert!(WorkspaceDataBus::from_url_json(bad_bus).is_err());

    // Unknown field inside SharedGraph rejected in embedded URL validation
    let bad_embedded: serde_json::Value = serde_json::from_str(
        r#"{"config":{"seeded_graph":{"revision":1,"nodes":[],"edges":[],"bad":true}}}"#,
    )
    .unwrap();
    assert!(validate_embedded_url_bus_fields(&bad_embedded).is_err());

    // Oversized node counts rejected
    let too_many_url = SharedGraph {
        revision: 1,
        nodes: (0..=URL_MAX_BUS_NODES)
            .map(|id| SharedNode {
                id,
                norm_pos: [0.5, 0.5],
            })
            .collect(),
        edges: Vec::new(),
    };
    assert!(too_many_url.validate(URL_MAX_BUS_NODES, 2_000).is_err());
    assert!(too_many_url.validate(SESSION_MAX_BUS_NODES, 10_000).is_ok());
}

#[test]
fn test_wcag_contrast_and_live_announcements_for_bus_controls() {
    crate::accessibility::clear_test_announcements();
    for mode in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
        let ctx = egui::Context::default();
        ctx.set_visuals(mode.visuals());
        let mut follow = true;
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let bg = ui.visuals().faint_bg_color;
                let fg = crate::app::contrasting_text_color(bg);
                assert!(
                    contrast_ratio(fg, bg) >= 4.5,
                    "Banner text contrast below WCAG AA 4.5:1 in {mode:?}"
                );
                let _ = render_follow_workspace_toggle(ui, &mut follow, "TestTool", Some(2));
                render_node_budget_banner(ui, Some(64), 32, "neurons");
            });
        });
    }
    let msgs = crate::accessibility::all_announcements();
    assert!(msgs.iter().any(|(m, _)| m.contains("clamped shared graph")));
}
