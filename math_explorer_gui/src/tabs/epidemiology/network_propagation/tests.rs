#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::app::MathExplorerApp;
use crate::framework::InteractiveTool;
use eframe::egui::{self, Event, Key, Modifiers, PointerButton, Pos2, RawInput, Rect};
use math_explorer::epidemiology::networks::{NetworkEpidemicModel, NodeState};

fn screen_rect() -> Rect {
    Rect::from_min_size(Pos2::ZERO, egui::vec2(1280.0, 800.0))
}

#[test]
fn test_network_propagation_step_once_keeps_paused() {
    let mut tool = NetworkPropagationTool::default();
    let _ = tool.model.set_beta(1.0);
    assert!(!tool.is_running);
    tool.step_once();
    assert!(!tool.is_running);
    assert_eq!(tool.sim_steps(), 1);
}

#[test]
fn test_zero_allocation_step_under_verified_allocator_in_gui() {
    let mut tool = NetworkPropagationTool::default();
    let target = (0..tool.model.num_nodes())
        .find(|&n| tool.model.states()[n] == NodeState::Susceptible)
        .unwrap();
    assert_eq!(tool.toggle_node_vaccination(target), Ok(true));

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        let _ = compute_network_analytics(&tool.model);
        tool.model.step_with_rng(&mut tool.rng);
    });
}

#[test]
fn test_budget_enforcement_and_patient_zero_rejection() {
    let mut tool = NetworkPropagationTool {
        budget: 2,
        ..Default::default()
    };
    let p0 = tool.model.patient_zero().expect("patient zero exists");
    assert!(tool.toggle_node_vaccination(p0).is_err());

    let sus: Vec<usize> = (0..tool.model.num_nodes())
        .filter(|&n| tool.model.states()[n] == NodeState::Susceptible)
        .take(3)
        .collect();
    assert_eq!(tool.toggle_node_vaccination(sus[0]), Ok(true));
    assert_eq!(tool.toggle_edge_quarantine(0), Ok(true));
    assert_eq!(tool.remaining_budget(), 0);
    assert!(tool.toggle_node_vaccination(sus[1]).is_err());

    // Refunding one slot allows a new intervention
    assert_eq!(tool.toggle_edge_quarantine(0), Ok(false));
    assert_eq!(tool.remaining_budget(), 1);
    assert_eq!(tool.toggle_node_vaccination(sus[1]), Ok(true));
}

#[test]
fn test_try_load_state_rejects_invalid_payloads_atomically() {
    let mut tool = NetworkPropagationTool::default();
    tool.try_load_state(
        r#"{"num_nodes":40,"beta":0.2,"gamma":0.05,"seed":77,"budget":3,"vaccinated_nodes":[0]}"#,
    )
    .or_else(|_| {
        tool.try_load_state(
            r#"{"num_nodes":40,"beta":0.2,"gamma":0.05,"seed":77,"budget":3,"vaccinated_nodes":[1]}"#,
        )
    })
    .expect("valid baseline state");

    let baseline_json = tool.save_state().unwrap();
    let baseline_rev = tool.state_revision();
    let p0 = tool.model.patient_zero().unwrap();

    let invalid_cases = [
        // Out-of-range node index
        r#"{"num_nodes":40,"beta":0.2,"gamma":0.05,"seed":77,"budget":3,"vaccinated_nodes":[999]}"#
            .to_string(),
        // Out-of-range edge index
        r#"{"num_nodes":40,"beta":0.2,"gamma":0.05,"seed":77,"budget":3,"quarantined_edges":[99999]}"#
            .to_string(),
        // Duplicate vaccinated node
        r#"{"num_nodes":40,"beta":0.2,"gamma":0.05,"seed":77,"budget":3,"vaccinated_nodes":[2,2]}"#
            .to_string(),
        // Duplicate quarantined edge
        r#"{"num_nodes":40,"beta":0.2,"gamma":0.05,"seed":77,"budget":3,"quarantined_edges":[0,0]}"#
            .to_string(),
        // Vaccination of patient zero
        format!(
            r#"{{"num_nodes":40,"beta":0.2,"gamma":0.05,"seed":77,"budget":3,"vaccinated_nodes":[{p0}]}}"#
        ),
        // Interventions above budget
        r#"{"num_nodes":40,"beta":0.2,"gamma":0.05,"seed":77,"budget":1,"vaccinated_nodes":[1,2]}"#
            .to_string(),
    ];

    for bad in &invalid_cases {
        assert!(
            tool.try_load_state(bad).is_err(),
            "Expected try_load_state error for {bad}"
        );
        assert_eq!(tool.save_state().as_deref(), Some(baseline_json.as_str()));
        assert_eq!(tool.state_revision(), baseline_rev);
    }
}

#[test]
fn test_seeded_interventions_url_state_roundtrip() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut source_app = MathExplorerApp::default();
    let epi_idx = source_app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "epidemiology")
        .unwrap();
    source_app.selected_tab = epi_idx;
    assert!(source_app.tabs[epi_idx].select_tool_by_name("NetworkPropagationTool"));

    let custom_state = r#"{"num_nodes":48,"beta":0.25,"gamma":0.06,"seed":20261121,"budget":4,"vaccinated_nodes":[1,3],"quarantined_edges":[0,2]}"#;
    {
        let fw = source_app.tabs[epi_idx].framework_mut().unwrap();
        fw.active_tool
            .as_mut()
            .unwrap()
            .try_load_state(custom_state)
            .expect("valid custom state");
    }

    let share_url = source_app.encode_current_share_url().unwrap();
    let hash_idx = share_url.find('#').unwrap();
    let hash = &share_url[hash_idx..];

    let mut dest_app = MathExplorerApp::default();
    assert!(dest_app.apply_share_location_strings(hash, "", None).unwrap());
    let restored = dest_app.tabs[epi_idx]
        .framework()
        .unwrap()
        .active_tool
        .as_ref()
        .unwrap()
        .save_state()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&restored).unwrap(),
        serde_json::from_str::<serde_json::Value>(custom_state).unwrap()
    );
}

#[test]
fn test_disconnected_and_fully_vaccinated_graphs_guard_final_size() {
    let isolated = NetworkEpidemicModel::builder()
        .num_nodes(4)
        .parameters(0.4, 0.1)
        .states(vec![
            NodeState::Infected,
            NodeState::Susceptible,
            NodeState::Susceptible,
            NodeState::Susceptible,
        ])
        .adjacency(vec![vec![]; 4])
        .build()
        .unwrap();
    let summary_isolated = compute_network_analytics(&isolated);
    assert_eq!(summary_isolated.effective_r0, 0.0);
    assert_eq!(summary_isolated.extinction_probability, 1.0);
    assert_eq!(summary_isolated.expected_additional_infections, 0.0);

    let zero_susceptible = NetworkEpidemicModel::builder()
        .num_nodes(3)
        .parameters(0.8, 0.1)
        .states(vec![
            NodeState::Infected,
            NodeState::Vaccinated,
            NodeState::Vaccinated,
        ])
        .adjacency(vec![vec![1, 2], vec![0, 2], vec![0, 1]])
        .build()
        .unwrap();
    let summary_vax = compute_network_analytics(&zero_susceptible);
    assert_eq!(summary_vax.susceptible, 0);
    assert_eq!(summary_vax.expected_additional_infections, 0.0);
}

fn locate_canvas_rect(ctx: &egui::Context, tool: &mut NetworkPropagationTool) -> Rect {
    let _ = ctx.run(
        RawInput {
            screen_rect: Some(screen_rect()),
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    let out = ctx.run(
        RawInput {
            screen_rect: Some(screen_rect()),
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    let ak = out
        .platform_output
        .accesskit_update
        .expect("AccessKit update emitted");
    ak.nodes
        .iter()
        .find_map(|(_, node)| {
            (node.label() == Some("Network Propagation Visualization"))
                .then(|| node.bounds())
                .flatten()
                .map(|b| {
                    Rect::from_min_max(
                        egui::pos2(b.x0 as f32, b.y0 as f32),
                        egui::pos2(b.x1 as f32, b.y1 as f32),
                    )
                })
        })
        .expect("Network Propagation Visualization rect")
}

fn press_key(key: Key) -> Event {
    Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    }
}

#[test]
fn test_canvas_pointer_click_and_keyboard_containment_actions() {
    let mut tool = NetworkPropagationTool::default();
    let target = (0..tool.model.num_nodes())
        .find(|&n| tool.model.states()[n] == NodeState::Susceptible)
        .unwrap();

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let rect = locate_canvas_rect(&ctx, &mut tool);
    let (center, scale) = NetworkPropagationTool::canvas_scale_and_center(rect);
    let click_pos = tool.node_screen_pos(target, center, scale);

    let _ = ctx.run(
        RawInput {
            screen_rect: Some(screen_rect()),
            events: vec![
                Event::PointerMoved(click_pos),
                Event::PointerButton {
                    pos: click_pos,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: Modifiers::NONE,
                },
                Event::PointerButton {
                    pos: click_pos,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: Modifiers::NONE,
                },
            ],
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    assert_eq!(tool.used_budget(), 1);
    assert!(tool.focused_canvas_id().is_some());

    let _ = ctx.run(
        RawInput {
            screen_rect: Some(screen_rect()),
            events: vec![
                press_key(Key::M),
                press_key(Key::ArrowRight),
                press_key(Key::Space),
            ],
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    assert_eq!(tool.interaction_mode, ContainmentMode::QuarantineEdge);
    assert_eq!(tool.quarantined_edges.len(), 1);
    let live_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert!(
        live_msg
            .as_deref()
            .is_some_and(|m| m.contains("Quarantined Edge")),
        "Expected live announcement after keyboard edge quarantine, got {live_msg:?}"
    );
}

#[test]
fn test_hub_vaccination_and_bridge_cut_presets_collapse_outbreak_metrics() {
    let mut hub_tool = NetworkPropagationTool::default();
    hub_tool
        .import_preset_json(super::presets::HUB_VACCINATION_CHALLENGE_JSON)
        .expect("hub challenge preset loads");
    let before_hub = compute_network_analytics(&hub_tool.model);
    assert!(before_hub.effective_r0 > 1.5);

    let mut nodes_by_deg: Vec<usize> = (0..hub_tool.model.num_nodes())
        .filter(|&n| hub_tool.model.states()[n] == NodeState::Susceptible)
        .collect();
    nodes_by_deg.sort_by_key(|&n| std::cmp::Reverse(hub_tool.model.active_degree(n)));
    for &hub in nodes_by_deg.iter().take(hub_tool.budget) {
        assert_eq!(hub_tool.toggle_node_vaccination(hub), Ok(true));
    }
    let after_hub = compute_network_analytics(&hub_tool.model);
    assert!(after_hub.effective_r0 < before_hub.effective_r0);
    assert!(after_hub.expected_additional_infections < before_hub.expected_additional_infections);

    let mut bridge_tool = NetworkPropagationTool::default();
    bridge_tool
        .import_preset_json(super::presets::BRIDGE_CUT_CHALLENGE_JSON)
        .expect("bridge challenge preset loads");
    let before_bridge = compute_network_analytics(&bridge_tool.model);
    let mut edge_scores: Vec<(usize, usize)> = bridge_tool
        .model
        .edges()
        .iter()
        .enumerate()
        .map(|(idx, &[u, v])| {
            (
                idx,
                bridge_tool.model.active_degree(u) + bridge_tool.model.active_degree(v),
            )
        })
        .collect();
    edge_scores.sort_by_key(|&(_, score)| std::cmp::Reverse(score));
    for &(edge_idx, _) in edge_scores.iter().take(bridge_tool.budget) {
        assert_eq!(bridge_tool.toggle_edge_quarantine(edge_idx), Ok(true));
    }
    let after_bridge = compute_network_analytics(&bridge_tool.model);
    assert!(after_bridge.effective_r0 < before_bridge.effective_r0);
}
