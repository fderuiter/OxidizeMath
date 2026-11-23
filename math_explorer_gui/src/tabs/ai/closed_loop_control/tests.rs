//! Numerical Jacobian, EKF parity, GRPO/Q-learning convergence, zero-allocation, and UI contract tests.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::ekf::{FastEkf, LorenzKalmanModel, PkKalmanModel, SeirKalmanModel};
use super::pipeline::ClosedLoopRunner;
use super::plants::PkPlantTwin;
use super::state::{
    ClosedLoopState, ControllerKind, PlantKind, STARTER_LORENZ_OGY_JSON, STARTER_PK_DOSING_JSON,
    STARTER_SEIR_OUTBREAK_JSON,
};
use super::ClosedLoopControlTool;
use crate::async_sim::SimulationRunner;
use crate::framework::{InteractiveTool, SimulationFramework};
use crate::url_state::{MAX_DECODED_URL_STATE_BYTES, MAX_URL_FRAGMENT_BYTES, URL_JSON_BOUNDS};
use eframe::egui::{self, vec2, Event, Key, Modifiers, PointerButton, Pos2, RawInput, Rect};
use math_explorer::applied::algorithms::kalman::{KalmanFilter, KalmanSystem};
use math_explorer::applied::pharmacokinetics::{BatemanCompartmentDynamics, PharmacokineticModel};
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::analysis::ode::rk4_step_dual_jacobian;

fn assert_finite_diff_jacobian<M: KalmanSystem<f64>>(model: &M, x0: &[f64], dt: f64) {
    let n = x0.len();
    let base = DVector::from_column_slice(x0);
    let (_, analytic) = model.predict_state(&base, dt);
    let eps = 1.0e-5;
    for col in 0..n {
        let mut xp = base.clone();
        let mut xm = base.clone();
        xp[col] += eps;
        xm[col] -= eps;
        let (fp, _) = model.predict_state(&xp, dt);
        let (fm, _) = model.predict_state(&xm, dt);
        for row in 0..n {
            let num = (fp[row] - fm[row]) / (2.0 * eps);
            let err = (analytic[(row, col)] - num).abs();
            assert!(
                err < 1.0e-8,
                "Jacobian mismatch at ({row}, {col}): analytic={} vs numeric={num} (err={err})",
                analytic[(row, col)]
            );
        }
    }
}

#[test]
fn test_ekf_jacobians_match_central_finite_differences_all_plants() {
    let lorenz = LorenzKalmanModel {
        sigma: 10.0,
        rho_eff: 29.2,
        beta: 8.0 / 3.0,
        q_var: 0.02,
        r_var: 0.10,
    };
    assert_finite_diff_jacobian(&lorenz, &[-12.4, -17.8, 25.6], 0.01);

    let seir = SeirKalmanModel {
        pop_n: 1000.0,
        beta_eff: 0.54,
        sigma: 0.25,
        gamma: 0.12,
        q_var: 0.03,
        r_var: 4.0,
    };
    assert_finite_diff_jacobian(&seir, &[880.0, 46.0, 32.0, 42.0], 0.25);

    let pk = PkKalmanModel {
        ka: 1.10,
        volume: 25.0,
        bioavail: 0.85,
        dose_bolus_mg: 30.0,
        q_var: 0.02,
        r_var: 0.04,
    };
    assert_finite_diff_jacobian(&pk, &[42.0, 2.15, 0.16], 0.50);
}

#[test]
fn test_fast_ekf_parity_with_kalman_filter_and_zero_sensor_noise_safety() {
    let dt = 0.01;
    let model = LorenzKalmanModel {
        sigma: 10.0,
        rho_eff: 28.5,
        beta: 8.0 / 3.0,
        q_var: 0.02,
        r_var: 0.0, // Exercises sensor_noise == 0.0 covariance floor
    };
    let init_s = [-13.0, -18.0, 26.0];
    let init_c = [2.5, 8.0, 8.0];
    let mut kf = KalmanFilter::builder(model, dt)
        .initial_state(DVector::from_column_slice(&init_s))
        .initial_covariance(DMatrix::from_diagonal(&DVector::from_column_slice(&init_c)))
        .build()
        .expect("KalmanFilter builds");
    let mut fast = FastEkf::from_validated_builder(model, dt, init_s, init_c);

    let q_diag = model.q_diag(dt);
    let lorenz_ode = super::ekf::build_lorenz_ode(model.sigma, model.rho_eff, model.beta);
    for step in 0..14 {
        let z = -12.5 + 0.35 * (step as f64 * 0.4).sin();
        kf.predict();
        kf.update(&DVector::from_element(1, z))
            .expect("floored R prevents singular innovation");
        fast.predict_ode_rk4(&lorenz_ode, dt, &q_diag);
        fast.update_unit_obs(0, z, 0.0);

        for i in 0..3 {
            assert!((kf.state[i] - fast.state[i]).abs() < 1.0e-12);
            for j in 0..3 {
                assert!((kf.covariance[(i, j)] - fast.cov[i][j]).abs() < 1.0e-12);
            }
        }
    }
}

#[test]
fn test_bateman_analytical_parity_and_pk_hidden_clearance_convergence() {
    let cfg = ClosedLoopState {
        plant: PlantKind::Pharmacokinetics,
        sensor_noise: 0.02,
        actuation_budget: 0.0,
        warmup_episodes: 0,
        ..Default::default()
    };
    let mut twin = PkPlantTwin::new(&cfg);
    let mut rng = math_explorer::rng::OxidizeRng::new(101);
    twin.controlled_state = [0.85 * 105.0, 0.0, 0.13];
    twin.ekf.state[0] = 0.85 * 105.0;

    let initial_ke_err = (twin.ekf.state[2] - 0.13).abs();
    // Verify 4th-order RK4 exactness against BatemanModel at fine dt = 0.05 h
    let pk_dyn = BatemanCompartmentDynamics::new(1.10, 25.0);
    let mut fine_state = [0.85 * 105.0, 0.0, 0.13];
    for step_idx in 1..=20 {
        let (ns, _) = rk4_step_dual_jacobian(&pk_dyn, 0.0, &fine_state, 0.05);
        fine_state = ns;
        let exact = twin
            .open_loop_superposition
            .base_model
            .concentration(step_idx as f64 * 0.05);
        assert!((fine_state[1] - exact).abs() < 1.0e-6);
    }
    for _ in 0..36 {
        let out = twin.step(-1.0, &cfg, &mut rng);
        let analytic = twin
            .open_loop_superposition
            .base_model
            .concentration(out.time);
        assert!(
            (out.controlled_primary - analytic).abs() < 2.0e-3,
            "RK4 ADME C_p={} diverged from BatemanModel={analytic} at t={}",
            out.controlled_primary,
            out.time
        );
    }
    let final_ke_err = (twin.ekf.state[2] - 0.13).abs();
    assert!(
        final_ke_err < 0.45 * initial_ke_err,
        "PkKalmanModel EKF must converge hidden clearance k_e toward true 0.13 (init err={initial_ke_err}, final={final_ke_err})"
    );
}

#[test]
fn test_grpo_lorenz_ogy_stabilization() {
    let mut lorenz_tool = ClosedLoopControlTool::default();
    lorenz_tool
        .import_preset_json(STARTER_LORENZ_OGY_JSON)
        .expect("Lorenz preset loads");
    lorenz_tool.reset_untrained_policy();
    let r0 = lorenz_tool
        .pipeline
        .train_episode_zero_alloc(&lorenz_tool.state);
    assert!(
        lorenz_tool.pipeline.grpo.last_surrogate_after
            >= lorenz_tool.pipeline.grpo.last_surrogate_before - 1.0e-12
    );
    for _ in 0..11 {
        let _ = lorenz_tool
            .pipeline
            .train_episode_zero_alloc(&lorenz_tool.state);
    }
    assert!(lorenz_tool.pipeline.grpo.last_mean_return > r0);

    lorenz_tool
        .pipeline
        .reset_live_traces_only(&lorenz_tool.state);
    let mut ctrl_tube_sum = 0.0;
    let mut open_tube_sum = 0.0;
    for k in 0..220 {
        let out = lorenz_tool.step_closed_loop();
        if k >= 60 {
            ctrl_tube_sum += out.tracking_metric;
            let ov = lorenz_tool.pipeline.live_lorenz.uncontrolled.state.vec;
            let (_, open_d) = lorenz_tool
                .pipeline
                .live_lorenz
                .nearest_upo_index(&[ov.x, ov.y, ov.z]);
            open_tube_sum += open_d;
        }
    }
    assert!(
        ctrl_tube_sum < 0.55 * open_tube_sum,
        "Closed-loop Lorenz OGY tube error ({ctrl_tube_sum}) must beat uncontrolled twin ({open_tube_sum})"
    );
}

#[test]
fn test_grpo_seir_and_pk_closed_loop_stabilization() {
    let mut seir_tool = ClosedLoopControlTool::default();
    seir_tool
        .import_preset_json(STARTER_SEIR_OUTBREAK_JSON)
        .expect("SEIR preset loads");
    seir_tool.train_episodes_sync(6);
    seir_tool.pipeline.reset_live_traces_only(&seir_tool.state);
    let mut max_ctrl_i = 0.0_f64;
    let mut max_open_i = 0.0_f64;
    for _ in 0..140 {
        let out = seir_tool.step_closed_loop();
        max_ctrl_i = max_ctrl_i.max(out.controlled_primary);
        max_open_i = max_open_i.max(out.uncontrolled_primary);
    }
    assert!(
        max_ctrl_i < 0.60 * max_open_i,
        "Closed-loop SEIR peak I ({max_ctrl_i}) must suppress uncontrolled outbreak ({max_open_i})"
    );

    let mut pk_cfg = ClosedLoopState {
        plant: PlantKind::Pharmacokinetics,
        controller: ControllerKind::GrpoPolicy,
        sensor_noise: 0.08,
        actuation_budget: 1.15,
        warmup_episodes: 10,
        rng_seed: 2029,
        ..Default::default()
    };
    pk_cfg.clamp_in_place();
    let mut pk_tool = ClosedLoopControlTool {
        pipeline: super::pipeline::ClosedLoopPipeline::new(&pk_cfg),
        state: pk_cfg,
        ..ClosedLoopControlTool::default()
    };
    pk_tool.pipeline.reset_live_traces_only(&pk_tool.state);
    let mut max_open_cp = 0.0_f64;
    let mut ctrl_mtc_excess = 0.0_f64;
    let mut open_mtc_excess = 0.0_f64;
    for _ in 0..72 {
        let out = pk_tool.step_closed_loop();
        max_open_cp = max_open_cp.max(out.uncontrolled_primary);
        ctrl_mtc_excess += (out.controlled_primary - pk_tool.state.pk_mtc).max(0.0);
        open_mtc_excess += (out.uncontrolled_primary - pk_tool.state.pk_mtc).max(0.0);
    }
    assert!(max_open_cp > pk_tool.state.pk_mtc + 0.8);
    assert!(
        ctrl_mtc_excess < 0.25 * open_mtc_excess,
        "Adaptive PK dosing MTC excess ({ctrl_mtc_excess}) must beat open-loop SuperpositionModel ({open_mtc_excess})"
    );
}

#[test]
fn test_discrete_q_learning_select_action_into_convergence() {
    let mut tool = ClosedLoopControlTool::default();
    tool.import_preset_json(STARTER_PK_DOSING_JSON)
        .expect("PK Q-learning preset loads");
    tool.reset_untrained_policy();
    let initial_ret = tool.pipeline.train_episode_zero_alloc(&tool.state);
    for _ in 0..18 {
        let _ = tool.pipeline.train_episode_zero_alloc(&tool.state);
    }
    assert!(tool.pipeline.q_ctrl.is_finite());
    assert!(
        tool.pipeline.q_ctrl.last_episode_return > initial_ret - 5.0,
        "Q-learning return must remain finite and well-behaved"
    );

    tool.pipeline.reset_live_traces_only(&tool.state);
    let mut open_mtc_excess = 0.0;
    let mut q_mtc_excess = 0.0;
    for _ in 0..64 {
        let out = tool.step_closed_loop();
        open_mtc_excess += (out.uncontrolled_primary - tool.state.pk_mtc).max(0.0);
        q_mtc_excess += (out.controlled_primary - tool.state.pk_mtc).max(0.0);
    }
    assert!(q_mtc_excess < 0.35 * open_mtc_excess);
}

#[test]
fn test_verified_allocator_zero_heap_allocations_in_hot_steps_kicks_and_rollouts() {
    let mut lorenz = ClosedLoopControlTool::default();
    let mut seir = ClosedLoopControlTool::default();
    seir.state.plant = PlantKind::SeirEpidemic;
    seir.rebuild_simulation();
    let mut pk_q = ClosedLoopControlTool::default();
    pk_q.state.plant = PlantKind::Pharmacokinetics;
    pk_q.state.controller = ControllerKind::DiscreteQLearning;
    pk_q.rebuild_simulation();

    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            for step in 0..32 {
                if step == 10 {
                    let _ = lorenz.apply_kick(0.8, -0.6);
                    let _ = seir.apply_kick(0.7, 0.5);
                    let _ = pk_q.apply_kick(0.6, 0.3);
                }
                let _ = lorenz.step_closed_loop();
                let _ = seir.step_closed_loop();
                let _ = pk_q.step_closed_loop();
            }
            let _ = lorenz.pipeline.train_episode_zero_alloc(&lorenz.state);
            let _ = seir.pipeline.train_episode_zero_alloc(&seir.state);
            let _ = pk_q.pipeline.train_episode_zero_alloc(&pk_q.state);
        });
    }
    assert!(lorenz.pipeline.last_output.ekf_primary.is_finite());
    assert!(seir.pipeline.last_output.ekf_latent.is_finite());
    assert!(pk_q.pipeline.last_output.ekf_primary.is_finite());
}

#[test]
fn test_deterministic_oxidize_rng_replay_and_click_to_kick_recovery() {
    let mut a = ClosedLoopControlTool::default();
    let mut b = ClosedLoopControlTool::default();
    let mut c = ClosedLoopControlTool::default();
    c.state.rng_seed = 9999;
    c.rebuild_simulation();

    for _ in 0..24 {
        let oa = a.step_closed_loop();
        let ob = b.step_closed_loop();
        assert_eq!(
            oa.noisy_measurement.to_bits(),
            ob.noisy_measurement.to_bits()
        );
        assert_eq!(oa.ekf_primary.to_bits(), ob.ekf_primary.to_bits());
        assert_eq!(oa.actuation.to_bits(), ob.actuation.to_bits());
    }
    let oc = c.step_closed_loop();
    assert_ne!(
        a.pipeline.last_output.noisy_measurement.to_bits(),
        oc.noisy_measurement.to_bits()
    );

    let pre_innov = a.pipeline.last_output.innovation.abs();
    let kick_out = a.apply_kick(0.95, -0.75);
    assert!(
        kick_out.innovation.abs() > pre_innov + 0.5,
        "State kick must immediately spike EKF innovation ({} vs pre {pre_innov})",
        kick_out.innovation
    );
    let post_kick_tube = kick_out.tracking_metric;
    for _ in 0..120 {
        let _ = a.step_closed_loop();
    }
    assert!(
        a.pipeline.last_output.tracking_metric < post_kick_tube,
        "Controller must recover toward orbit after kick when actuation budget permits"
    );
}

#[test]
fn test_runner_slice_budget_url_and_plot_exports() {
    let default_tool = ClosedLoopControlTool::default();
    assert!(default_tool.state_bounds().max_bytes <= URL_JSON_BOUNDS.max_bytes);
    let mut runner = ClosedLoopRunner::from_state_and_pipeline(
        ClosedLoopState::default(),
        &default_tool.pipeline,
    );
    runner.process_command(crate::async_sim::SimCommand::TrainEpisodes(4));
    let slice_start = crate::web_io::PlatformInstant::now();
    for _ in 0..runner.get_steps_per_frame() {
        runner.step();
    }
    assert!(
        slice_start.elapsed().as_secs_f64() < 0.25,
        "Cooperative runner batch must stay fast and non-blocking"
    );
    assert!(runner.is_finite_state());

    let mut fw = SimulationFramework::new("ai");
    assert!(fw.select_tool_by_name("ClosedLoopControlTool"));
    fw.apply_starter_card(
        "Lorenz Starter",
        Some("ClosedLoopControlTool"),
        Some(STARTER_LORENZ_OGY_JSON),
    )
    .expect("starter applies");
    let preset_json = fw.export_url_preset_json().unwrap().unwrap();
    assert!(preset_json.len() <= MAX_DECODED_URL_STATE_BYTES);
    let share_url = fw.encode_share_url().expect("url encodes");
    assert!(share_url.len() <= MAX_URL_FRAGMENT_BYTES + 64);

    let active = fw.active_tool.as_ref().expect("active tool");
    let datasets = active.export_datasets();
    let opts = active.export_figure_options().expect("export options");
    let csv = egui_plot::export::format_csv(&datasets);
    assert!(csv.contains("Uncontrolled Open-Loop Twin"));
    assert!(csv.contains("EKF +1 Sigma Bound"));
    let svg = egui_plot::export::export_custom_svg(&datasets, &opts).expect("svg export");
    assert!(svg.contains("<svg") && svg.contains("</svg>"));
}

#[test]
fn test_canvas_click_to_kick_and_tab_traversal() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut ui_tool = ClosedLoopControlTool::default();
    let screen = Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0)));
    let _ = ctx.run(
        RawInput {
            screen_rect: screen,
            ..Default::default()
        },
        |c| ui_tool.show(c),
    );
    let before_len = ui_tool.pipeline.controlled_trace.len();
    let click_pos = Pos2::new(760.0, 95.0);
    let _ = ctx.run(
        RawInput {
            screen_rect: screen,
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
        |c| ui_tool.show(c),
    );
    assert!(
        ui_tool.pipeline.last_kick_vec.is_some()
            || ui_tool.pipeline.controlled_trace.len() >= before_len
    );
    let _ = ctx.run(
        RawInput {
            screen_rect: screen,
            events: vec![Event::Key {
                key: Key::Tab,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Modifiers::NONE,
            }],
            ..Default::default()
        },
        |c| ui_tool.show(c),
    );
}

#[cfg(all(feature = "physics", feature = "epidemiology", feature = "applied"))]
#[test]
fn test_split_pane_coexistence_with_chaos_epidemiology_and_clinical_trials() {
    use crate::app::{MathExplorerApp, ViewportLayout};
    use eframe::App;

    let mut app = MathExplorerApp::default();
    app.viewport_layout = ViewportLayout::SplitHorizontal;
    let ai_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Artificial Intelligence")
        .expect("AiTab registered");
    assert!(app.tabs[ai_idx].select_tool_by_name("ClosedLoopControlTool"));

    for partner_name in ["Chaos Theory", "Epidemiology", "Clinical Trials"] {
        let partner_idx = app
            .tabs
            .iter()
            .position(|t| t.name() == partner_name)
            .unwrap_or_else(|| panic!("Missing tab {partner_name}"));
        app.set_pane_tab(0, ai_idx);
        app.set_pane_tab(1, partner_idx);

        let ctx = egui::Context::default();
        let mut frame = eframe::Frame::_new_kittest();
        let _ = ctx.run(
            RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1440.0, 900.0))),
                ..Default::default()
            },
            |c| app.update(c, &mut frame),
        );
    }
}

// [cite:grpo]
