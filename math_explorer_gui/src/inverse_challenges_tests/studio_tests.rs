//! Studio challenge solving and tampered locked-field clamping tests across Quantum, MRI,
//! Medical, Epidemiology, Astrophysics, Clinical Trials, and Financial Math studios.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::app::MathExplorerApp;
use crate::framework::SimulationFramework;
use crate::url_state::{encode_base64url, parse_share_location};
use eframe::egui;

fn solve_new_studio_challenge(fw: &mut SimulationFramework, id: &str) {
    let tool = fw.active_tool.as_mut().expect("active tool");
    let mut cfg: serde_json::Value = serde_json::from_str(&tool.save_state().unwrap()).unwrap();
    match id {
        "quantum_barrier_50_split" => {
            cfg["v0"] = serde_json::json!(11.35);
            cfg["width"] = serde_json::json!(0.50);
            tool.try_load_state(&cfg.to_string()).unwrap();
            for _ in 0..12 {
                tool.step_simulation();
            }
        }
        "mri_kspace_radial_cs" => {
            cfg["mask_preset"] = serde_json::json!("radial");
            cfg["radial_spokes"] = serde_json::json!(16);
            tool.try_load_state(&cfg.to_string()).unwrap();
            for _ in 0..6 {
                tool.step_simulation();
                std::thread::sleep(std::time::Duration::from_millis(8));
            }
        }
        "radar_cough_zero_miss" => {
            cfg["clutter_window"] = serde_json::json!(56);
            cfg["kalman_process_var"] = serde_json::json!(5.0e-3);
            cfg["kalman_meas_var"] = serde_json::json!(1.0e-7);
            cfg["gating_threshold_mm"] = serde_json::json!(-4.2);
            cfg["gating_hysteresis_mm"] = serde_json::json!(0.30);
            tool.try_load_state(&cfg.to_string()).unwrap();
        }
        "epidemiology_hub_vaccination" => solve_epi_challenge_on_tool(tool, true),
        "epidemiology_bridge_quarantine" => solve_epi_challenge_on_tool(tool, false),
        "astrophysics_mystery_halo_fit" => {
            cfg["halo_kind"] = serde_json::json!("BurkertCore");
            cfg["halo_log10_mass"] = serde_json::json!(10.15);
            cfg["halo_scale_kpc"] = serde_json::json!(2.40);
            cfg["halo_concentration"] = serde_json::json!(13.0);
            tool.try_load_state(&cfg.to_string()).unwrap();
        }
        other => panic!("Unexpected challenge id {other}"),
    }
}

fn solve_epi_challenge_on_tool(tool: &mut Box<dyn crate::framework::InteractiveTool>, vax: bool) {
    use math_explorer::epidemiology::networks::{NetworkEpidemicModel, NodeState};
    use oxidize_core::rng::OxidizeRng;
    let mut cfg: serde_json::Value = serde_json::from_str(&tool.save_state().unwrap()).unwrap();
    let (n, beta, gamma, seed) = (
        cfg["num_nodes"].as_u64().unwrap() as usize,
        cfg["beta"].as_f64().unwrap(),
        cfg["gamma"].as_f64().unwrap(),
        cfg["seed"].as_u64().unwrap(),
    );
    let mut rng = OxidizeRng::new(seed);
    let probe = NetworkEpidemicModel::new_with_rng(n, beta, gamma, &mut rng);
    let p0 = probe.patient_zero().unwrap();
    if vax {
        let mut nodes: Vec<usize> = (0..n)
            .filter(|&i| probe.states()[i] == NodeState::Susceptible)
            .collect();
        nodes.sort_by_key(|&i| {
            std::cmp::Reverse((
                usize::from(probe.adjacency()[p0].contains(&i)),
                probe.active_degree(i),
            ))
        });
        nodes.truncate(5);
        cfg["vaccinated_nodes"] = serde_json::json!(nodes);
    } else {
        let mut edges: Vec<usize> = (0..probe.edges().len()).collect();
        edges.sort_by_key(|&e| {
            let [u, v] = probe.edges()[e];
            std::cmp::Reverse((
                usize::from(u == p0 || v == p0),
                probe.active_degree(u) + probe.active_degree(v),
            ))
        });
        edges.truncate(5);
        cfg["quarantined_edges"] = serde_json::json!(edges);
    }
    tool.try_load_state(&cfg.to_string()).unwrap();
    for _ in 0..90 {
        tool.step_simulation();
    }
}

#[test]
fn test_six_new_studio_challenges_start_unsolved_and_announce_aria_on_solve() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    for (domain, id) in [
        ("quantum", "quantum_barrier_50_split"),
        ("mri", "mri_kspace_radial_cs"),
        ("medical", "radar_cough_zero_miss"),
        ("epidemiology", "epidemiology_hub_vaccination"),
        ("epidemiology", "epidemiology_bridge_quarantine"),
        ("astrophysics", "astrophysics_mystery_halo_fit"),
    ] {
        let ctx = egui::Context::default();
        let mut fw = SimulationFramework::new(domain);
        fw.launch_challenge_by_id(id, Some(&ctx)).unwrap();
        let base_eval = fw
            .active_tool
            .as_ref()
            .and_then(|t| t.evaluate_challenge())
            .unwrap();
        assert!(!base_eval.passed, "{id} started solved: {base_eval:?}");

        crate::accessibility::clear_test_announcements();
        solve_new_studio_challenge(&mut fw, id);
        let _ = ctx.run(egui::RawInput::default(), |ctx| fw.show(ctx, domain));
        let solved_eval = fw
            .active_tool
            .as_ref()
            .and_then(|t| t.evaluate_challenge())
            .unwrap();
        assert!(solved_eval.passed, "{id} did not pass: {solved_eval:?}");
        assert!(crate::accessibility::all_announcements()
            .iter()
            .any(|(msg, prio)| msg.contains("Challenge passed") && prio == "polite"));
    }
}

fn assert_locked_field_clamped(
    slug: &str,
    id: &str,
    key: &str,
    bad_val: serde_json::Value,
    expected_val: serde_json::Value,
) {
    let mut app = MathExplorerApp::default();
    app.apply_share_location_strings(&format!("#tab={slug}&challenge={id}"), "", None)
        .unwrap();
    let url = app.encode_current_share_url().unwrap();
    let params = parse_share_location(&url[url.find('#').unwrap()..], "")
        .unwrap()
        .unwrap();
    let mut env: serde_json::Value =
        serde_json::from_str(params.state_json.as_ref().unwrap()).unwrap();
    env["config"][key] = bad_val;
    let b64 = encode_base64url(env.to_string().as_bytes());
    let mut restored = MathExplorerApp::default();
    assert!(restored
        .apply_share_location_strings(&format!("#tab={slug}&challenge={id}&state={b64}"), "", None)
        .unwrap());
    let fw = restored.tabs[restored.selected_tab].framework().unwrap();
    let cfg: serde_json::Value =
        serde_json::from_str(&fw.active_tool.as_ref().unwrap().save_state().unwrap()).unwrap();
    assert_eq!(
        cfg.get(key).cloned().unwrap_or(serde_json::Value::Null),
        expected_val,
        "{id}:{key}"
    );
}

#[test]
fn test_tampered_locked_fields_clamped_across_five_studios() {
    use serde_json::{json, Value};
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    for (slug, id, key, bad, exp) in [
        (
            "quantum",
            "quantum_barrier_50_split",
            "k0",
            json!(9.5),
            json!(5.0),
        ),
        (
            "mri",
            "mri_kspace_radial_cs",
            "phantom",
            json!("turing_spots"),
            json!("shepp_logan"),
        ),
        (
            "medical",
            "radar_cough_zero_miss",
            "system_latency_ms",
            json!(0.0),
            json!(60.0),
        ),
        (
            "epidemiology",
            "epidemiology_hub_vaccination",
            "seed",
            json!(999),
            json!(20261121),
        ),
        (
            "epidemiology",
            "epidemiology_bridge_quarantine",
            "budget",
            json!(25),
            Value::Null,
        ),
        (
            "astrophysics",
            "astrophysics_mystery_halo_fit",
            "distance_mpc",
            json!(99.0),
            json!(14.5),
        ),
    ] {
        assert_locked_field_clamped(slug, id, key, bad, exp);
    }
}
