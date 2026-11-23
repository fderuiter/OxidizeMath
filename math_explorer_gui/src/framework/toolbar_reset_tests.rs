#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::framework::SimulationFramework;
use eframe::egui;

fn find_ode_tool_index(framework: &SimulationFramework) -> usize {
    framework
        .available_tools
        .iter()
        .position(|t| t.display_name.contains("ODE") || t.id.contains("Ode"))
        .unwrap_or(0)
}

fn mutate_dt_in_state(state_json: &str, new_dt: f64) -> String {
    let mut val: serde_json::Value =
        serde_json::from_str(state_json).expect("valid JSON tool state");
    if let Some(obj) = val.as_object_mut() {
        obj.insert("dt".to_string(), serde_json::json!(new_dt));
        if obj.contains_key("param_k") {
            obj.insert("param_k".to_string(), serde_json::json!(4.25));
        }
    }
    serde_json::to_string(&val).expect("serialize mutated state")
}

fn find_reset_button_center(output: &egui::FullOutput) -> egui::Pos2 {
    let ak_update = output
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit tree must be emitted");
    let reset_bounds = ak_update
        .nodes
        .iter()
        .find_map(|(_, node)| {
            let label = node.label().unwrap_or("");
            if node.role() == egui::accesskit::Role::Button && label.contains("Reset Parameters") {
                node.bounds()
            } else {
                None
            }
        })
        .expect("Toolbar Reset Parameters button must exist with accessible Button role");

    egui::pos2(
        ((reset_bounds.x0 + reset_bounds.x1) * 0.5) as f32,
        ((reset_bounds.y0 + reset_bounds.y1) * 0.5) as f32,
    )
}

fn click_toolbar_button(
    ctx: &egui::Context,
    framework: &mut SimulationFramework,
    screen_rect: egui::Rect,
    center: egui::Pos2,
    id_source: &str,
) -> egui::FullOutput {
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            events: vec![
                egui::Event::PointerMoved(center),
                egui::Event::PointerButton {
                    pos: center,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            ..Default::default()
        },
        |ctx| framework.show(ctx, id_source),
    );
    ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            events: vec![egui::Event::PointerButton {
                pos: center,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        },
        |ctx| framework.show(ctx, id_source),
    )
}

#[test]
fn test_toolbar_reset_button_click_restores_baseline_and_requests_repaint() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();

    let mut framework = SimulationFramework::new("analysis");
    let ode_idx = find_ode_tool_index(&framework);
    framework.select_tool(ode_idx);

    let baseline = framework
        .baseline_tool_state
        .clone()
        .expect("ODE tool must provide baseline_tool_state");
    let modified = mutate_dt_in_state(&baseline, 0.25);
    framework
        .active_tool
        .as_mut()
        .unwrap()
        .load_state(&modified);
    assert_ne!(
        framework.active_tool.as_ref().unwrap().save_state(),
        Some(baseline.clone())
    );

    let screen_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 800.0));
    let first_out = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            ..Default::default()
        },
        |ctx| framework.show(ctx, "toolbar_reset_click_test"),
    );
    let center = find_reset_button_center(&first_out);

    crate::accessibility::clear_test_announcements();
    let click_out = click_toolbar_button(
        &ctx,
        &mut framework,
        screen_rect,
        center,
        "toolbar_reset_click_test",
    );

    assert_eq!(
        framework.active_tool.as_ref().unwrap().save_state(),
        Some(baseline)
    );
    assert_eq!(
        click_out
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map(|v| v.repaint_delay),
        Some(std::time::Duration::ZERO)
    );
    assert!(crate::accessibility::all_announcements()
        .iter()
        .any(|(msg, _)| msg.contains("Parameters reset to baseline defaults")));
}

#[test]
fn test_reset_parameters_synchronizes_saved_states_across_tool_switches() {
    let mut framework = SimulationFramework::new("analysis");
    assert!(framework.available_tools.len() >= 2);

    let ode_idx = find_ode_tool_index(&framework);
    let other_idx = if ode_idx == 0 { 1 } else { 0 };
    framework.select_tool(ode_idx);

    let tool_name = framework.active_tool.as_ref().unwrap().name().to_string();
    let baseline = framework.baseline_tool_state.clone().unwrap();
    let modified = mutate_dt_in_state(&baseline, 0.35);

    // Case 1: Modify -> Reset -> Switch away -> Switch back preserves baseline
    framework
        .active_tool
        .as_mut()
        .unwrap()
        .load_state(&modified);
    framework
        .saved_tool_states
        .insert(tool_name.clone(), modified.clone());
    framework.cache_valid_snapshot();

    framework.reset_parameters();
    assert_eq!(framework.saved_tool_states.get(&tool_name), Some(&baseline));
    assert_eq!(
        framework
            .last_valid_snapshot()
            .and_then(|s| s.tool_state.as_ref()),
        Some(&baseline)
    );

    framework.select_tool(other_idx);
    framework.select_tool(ode_idx);
    assert_eq!(
        framework.active_tool.as_ref().unwrap().save_state(),
        Some(baseline.clone())
    );

    // Case 2: Modify -> Switch away -> Switch back -> Reset still restores factory baseline
    framework
        .active_tool
        .as_mut()
        .unwrap()
        .load_state(&modified);
    let modified_saved = framework.active_tool.as_ref().unwrap().save_state();
    framework.select_tool(other_idx);
    framework.select_tool(ode_idx);
    assert_eq!(
        framework.active_tool.as_ref().unwrap().save_state(),
        modified_saved
    );

    framework.reset_parameters();
    assert_eq!(
        framework.active_tool.as_ref().unwrap().save_state(),
        Some(baseline)
    );
}

#[test]
fn test_reset_parameters_rebuilds_from_factory_when_baseline_missing() {
    let mut framework = SimulationFramework::new("analysis");
    let ode_idx = find_ode_tool_index(&framework);
    framework.select_tool(ode_idx);

    let factory_baseline = framework
        .baseline_tool_state
        .clone()
        .expect("factory baseline expected");
    let modified = mutate_dt_in_state(&factory_baseline, 0.4);
    framework
        .active_tool
        .as_mut()
        .unwrap()
        .load_state(&modified);

    // Simulate missing baseline state and active plot buffers
    framework.baseline_tool_state = None;
    egui_plot::append_last_plot_datasets(&[(
        "Stale Series".to_string(),
        vec![[0.0, 1.0], [1.0, 2.0]],
    )]);
    framework.current_valid_plot_datasets =
        vec![("Stale Series".to_string(), vec![[0.0, 1.0], [1.0, 2.0]])];

    framework.reset_parameters();

    assert_eq!(
        framework.active_tool.as_ref().unwrap().save_state(),
        Some(factory_baseline.clone()),
        "Missing baseline must rebuild active tool from ToolMetadata factory"
    );
    assert_eq!(framework.baseline_tool_state, Some(factory_baseline));
    assert!(egui_plot::get_last_plot_datasets().is_empty());
    assert!(framework.current_valid_plot_datasets.is_empty());
}

fn find_button_center_by_label(output: &egui::FullOutput, needle: &str) -> egui::Pos2 {
    let ak = output
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit tree must be emitted");
    let bounds = ak
        .nodes
        .iter()
        .find_map(|(_, node)| {
            let label = node.label().unwrap_or("");
            if node.role() == egui::accesskit::Role::Button && label.contains(needle) {
                node.bounds()
            } else {
                None
            }
        })
        .unwrap_or_else(|| panic!("Toolbar button containing '{needle}' must exist"));
    egui::pos2(
        ((bounds.x0 + bounds.x1) * 0.5) as f32,
        ((bounds.y0 + bounds.y1) * 0.5) as f32,
    )
}

fn assert_persisted_dual_baselines(
    fw: &SimulationFramework,
    preset_state: &str,
    factory_state: &str,
) {
    let persisted = fw.save_state().expect("framework save_state");
    let mut restored_fw = SimulationFramework::new("analysis");
    restored_fw.load_state(&persisted);
    assert!(restored_fw.has_preset_baseline());
    assert!(restored_fw.reset_to_preset());
    assert_eq!(
        restored_fw.active_tool.as_ref().unwrap().save_state(),
        Some(preset_state.to_string())
    );
    restored_fw.reset_to_factory_defaults();
    assert_eq!(
        restored_fw.active_tool.as_ref().unwrap().save_state(),
        Some(factory_state.to_string())
    );
}

#[test]
fn test_dual_baselines_revert_to_preset_and_factory_defaults_via_toolbar_and_persistence() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();

    let mut fw = SimulationFramework::new("analysis");
    let ode_idx = find_ode_tool_index(&fw);
    fw.select_tool(ode_idx);
    let factory_state = fw.factory_tool_state.clone().expect("factory baseline");
    assert!(!fw.has_preset_baseline());

    let preset_cfg = mutate_dt_in_state(&factory_state, 0.08);
    let envelope = format!(
        r#"{{"metadata":{{"schema_version":"1.0","title":"Test ODE Preset","domain":"analysis","timestamp":"2026-11-22T00:00:00Z"}},"config":{preset_cfg}}}"#
    );
    fw.apply_preset_json(&envelope)
        .expect("preset JSON should apply");
    assert!(fw.has_preset_baseline());
    let preset_state = fw.preset_tool_state.clone().expect("preset baseline");
    assert_ne!(preset_state, factory_state);
    assert_eq!(fw.factory_tool_state.as_ref(), Some(&factory_state));

    let user_modified = mutate_dt_in_state(&preset_state, 0.22);
    fw.active_tool.as_mut().unwrap().load_state(&user_modified);

    let screen_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 800.0));
    let out0 = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            ..Default::default()
        },
        |ctx| fw.show(ctx, "dual_baseline_toolbar"),
    );

    let preset_btn = find_button_center_by_label(&out0, "Revert to Preset");
    let out1 = click_toolbar_button(
        &ctx,
        &mut fw,
        screen_rect,
        preset_btn,
        "dual_baseline_toolbar",
    );
    assert_eq!(
        fw.active_tool.as_ref().unwrap().save_state(),
        Some(preset_state.clone())
    );
    assert_eq!(fw.factory_tool_state.as_ref(), Some(&factory_state));

    let factory_btn = find_button_center_by_label(&out1, "Factory Defaults");
    let _ = click_toolbar_button(
        &ctx,
        &mut fw,
        screen_rect,
        factory_btn,
        "dual_baseline_toolbar",
    );
    assert_eq!(
        fw.active_tool.as_ref().unwrap().save_state(),
        Some(factory_state.clone())
    );
    assert!(
        fw.has_preset_baseline(),
        "Factory reset must retain active preset baseline so user can revert back to preset"
    );
    assert_persisted_dual_baselines(&fw, &preset_state, &factory_state);
}

#[test]
fn test_all_registered_tools_preserve_factory_and_preset_baselines() {
    let mut domains: Vec<&'static str> = inventory::iter::<crate::framework::ToolMetadata>
        .into_iter()
        .map(|m| m.domain)
        .collect();
    domains.sort_unstable();
    domains.dedup();
    assert!(!domains.is_empty());

    for domain in domains {
        let mut fw = SimulationFramework::new(domain);
        for idx in 0..fw.available_tools.len() {
            fw.select_tool(idx);
            let expected_factory = (fw.available_tools[idx].build)().save_state();
            assert_eq!(
                fw.factory_tool_state, expected_factory,
                "Tool {} in domain {domain} must capture factory_tool_state on selection",
                fw.available_tools[idx].display_name
            );
        }

        let cards =
            crate::starter_gallery::get_starter_cards_for_domain(domain, &fw.available_tools);
        for card in cards.iter().filter(|c| c.preset_json.is_some()) {
            assert!(fw.launch_starter_card(card).is_ok());
            assert!(
                fw.factory_tool_state.is_some(),
                "Starter card '{}' in {domain} must retain factory_tool_state",
                card.title
            );
            assert!(
                fw.preset_tool_state.is_some(),
                "Starter card '{}' in {domain} must record preset_tool_state",
                card.title
            );
            let preset_snap = fw.preset_tool_state.clone();
            let factory_snap = fw.factory_tool_state.clone();
            fw.reset_to_factory_defaults();
            assert_eq!(fw.active_tool.as_ref().unwrap().save_state(), factory_snap);
            assert!(fw.reset_to_preset());
            assert_eq!(fw.active_tool.as_ref().unwrap().save_state(), preset_snap);
        }
    }
}
