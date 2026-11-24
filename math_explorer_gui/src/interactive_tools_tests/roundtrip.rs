#[cfg(all(
    feature = "pure_math",
    feature = "applied",
    feature = "physics",
    any(feature = "climate", feature = "epidemiology")
))]
fn assert_tool_roundtrip(tool: &mut dyn crate::framework::InteractiveTool, custom_json: &str) {
    let default_saved = tool.save_state().expect("default save_state");
    tool.load_state(custom_json);
    let updated_saved = tool.save_state().expect("updated save_state");
    assert_ne!(
        updated_saved,
        default_saved,
        "State did not change for {}",
        tool.name()
    );
    let expected: serde_json::Value = serde_json::from_str(custom_json).unwrap();
    let actual: serde_json::Value = serde_json::from_str(&updated_saved).unwrap();
    assert_eq!(
        actual,
        expected,
        "Roundtrip JSON mismatch for {}",
        tool.name()
    );
}

#[cfg(any(feature = "applied", not(target_arch = "wasm32")))]
fn assert_tab_roundtrip(tab: &mut dyn crate::tabs::ExplorerTab, custom_json: &str) {
    let default_saved = tab.save_state().expect("default save_state");
    tab.load_state(custom_json);
    let updated_saved = tab.save_state().expect("updated save_state");
    assert_ne!(
        updated_saved,
        default_saved,
        "State did not change for {}",
        tab.name()
    );
    let expected: serde_json::Value = serde_json::from_str(custom_json).unwrap();
    let actual: serde_json::Value = serde_json::from_str(&updated_saved).unwrap();
    assert_eq!(
        actual,
        expected,
        "Roundtrip JSON mismatch for {}",
        tab.name()
    );
}

#[test]
#[cfg(all(
    feature = "pure_math",
    feature = "applied",
    feature = "climate",
    feature = "physics"
))]
fn test_hoisted_domain_tools_state_roundtrip_part_1() {
    assert_tool_roundtrip(
        &mut crate::tabs::analysis::complex_mapping::ComplexMappingTool::default(),
        r#"{"function":"Sine","grid_min":-3.0,"grid_max":3.0,"grid_density":14,"inspection_point":[0.5,-0.5]}"#,
    );
    assert_tool_roundtrip(
        &mut crate::tabs::analysis::riemann::RiemannIntegrationTool::default(),
        r#"{"preset":"XSquared","min_x":0.0,"max_x":3.0,"steps":25}"#,
    );
    assert_tool_roundtrip(
        &mut crate::tabs::battery_degradation::capacity_fade::CapacityFadeTool::default(),
        r#"{"dod":60.0,"temperature":35.0,"cycles_to_simulate":1500.0}"#,
    );
    assert_tool_roundtrip(
        &mut crate::tabs::battery_degradation::lifetime_estimator::LifetimeEstimatorTool::default(),
        r#"{"target_capacity":75.0,"dod":40.0,"cycles_per_day":2.0}"#,
    );
    assert_tool_roundtrip(
        &mut crate::tabs::chaos::bifurcation::BifurcationDiagram::default(),
        r#"{"r_min":2.8,"r_max":3.9,"steps":250}"#,
    );
    assert_tool_roundtrip(
        &mut crate::tabs::chaos::fractals::FractalViewer::default(),
        r#"{"center_re":-0.5,"center_im":0.2,"zoom":2.5,"max_iter":150,"mode":"Julia","julia_re":-0.8,"julia_im":0.156}"#,
    );

    let mut cera = crate::tabs::climate::cera::CeraTool {
        diverged: true,
        prediction_result: Some(Ok(vec![1.0])),
        ..Default::default()
    };
    assert_tool_roundtrip(
        &mut cera,
        r#"{"in_channels":3,"latent_channels":6,"aligned_channels":3,"num_levels":12,"output_size":4}"#,
    );
    assert!(!cera.diverged);
    assert!(cera.prediction_result.is_none());
    assert!(cera.diagnostic_report.is_none());

    assert_tool_roundtrip(
        &mut crate::tabs::climate::co2_projections::Co2ProjectionsTool::default(),
        r#"{"reduction_scenario":0.65}"#,
    );
    assert_tool_roundtrip(
        &mut crate::tabs::climate::temperature::TemperatureAnomaliesTool::default(),
        r#"{"time_series":[[2000.0,0.42],[2010.0,0.71]]}"#,
    );
}

#[test]
#[cfg(all(
    feature = "pure_math",
    feature = "applied",
    feature = "epidemiology",
    feature = "physics"
))]
fn test_hoisted_domain_tools_state_roundtrip_part_2() {
    assert_tool_roundtrip(
        &mut <crate::tabs::clinical_trials::randomization::RandomizationTool>::default(),
        r#"{"randomization_type":"Block","n_subjects":24,"block_size":6}"#,
    );
    assert_tool_roundtrip(
        &mut crate::tabs::clinical_trials::sample_size::SampleSizeCalculatorTool::default(),
        r#"{"mode":"Proportions","alpha":0.01,"power":0.9,"delta":0.4,"sigma":1.2,"p1":0.3,"p2":0.5}"#,
    );
    assert_tool_roundtrip(
        &mut crate::tabs::clinical_trials::survival::SurvivalAnalysisTool::default(),
        r#"{"input_text":"5.0, 1\n12.0, 0"}"#,
    );
    assert_tool_roundtrip(
        &mut crate::tabs::clinical_trials::win_ratio::WinRatioStressTesterTool::default(),
        r#"{"lambda1":0.16,"lambda2":0.36,"alpha":0.7,"theta":0.6,"censor_horizon":4.5,"dropout_rate":0.06,"sig_alpha":0.05,"n_min":30,"n_max":150,"n_steps":5,"replicates":80,"seed":9001,"chart_mode":"TierWaterfall"}"#,
    );
    assert_tool_roundtrip(
        &mut crate::tabs::clinical_trials::win_ratio_workbench::WinRatioWorkbenchTool::default(),
        r#"{"cohort_text":"Treatment,120,0\nControl,80,2","pairing_mode":"Unmatched","active_view":"CohortWaterfall","tier_specs":[{"label":"Survival","kind":"HigherIsBetter","threshold":0.0},{"label":"Hosp","kind":"LowerIsBetter","threshold":0.0}],"weibull_cfg":{"lambda1":0.08,"lambda2":0.18,"alpha":1.25,"theta":0.65,"horizon_c":8.0}}"#,
    );
    assert_tool_roundtrip(
        &mut crate::tabs::epidemiology::network_propagation::NetworkPropagationTool::default(),
        r#"{"num_nodes":30,"beta":0.08,"gamma":0.03}"#,
    );
    assert_tool_roundtrip(
        &mut crate::tabs::financial_math::bankroll_growth::BankrollGrowthTool::default(),
        r#"{"initial_bankroll":2500.0,"probability":0.6,"odds":2.5,"num_bets":50}"#,
    );
    assert_tool_roundtrip(
        &mut crate::tabs::financial_math::bet_size::BetSizeCalculatorTool::default(),
        r#"{"probability_input":60.0,"odds_format":"American","decimal_odds":2.2,"american_odds":120.0,"fractional_numerator":6.0,"fractional_denominator":5.0,"bankroll":2000.0}"#,
    );
    assert_tool_roundtrip(
        &mut crate::tabs::financial_math::market_devig::MarketDeviggingTool::default(),
        r#"{"bankroll":2500.0,"quote_format":"Decimal","method":"Additive","decimal_quotes":[1.9,3.25,6.5,10.0,21.0,101.0],"target_outcome":5,"offered_decimal_odds":110.0}"#,
    );
    assert_tool_roundtrip(
        &mut crate::tabs::financial_math::market_execution::MarketExecutionTool::default(),
        r#"{"budget":500.0,"mode":"Hedge","arb_odds":[2.1,1.98],"dutch_odds":[2.2,3.5,4.2],"dutch_selected":[false,true,true],"dutch_target_profit_mode":false,"dutch_target_profit":50.0,"middle_odds_a":1.91,"middle_odds_b":1.91,"middle_hit_prob_pct":9.0,"hedge_orig_stake":100.0,"hedge_orig_odds":9.0,"hedge_live_odds":1.67,"hedge_use_target_floor":true,"hedge_target_floor":0.0}"#,
    );
    assert_tool_roundtrip(
        &mut crate::tabs::solid_state::band_structure::BandStructureTool::default(),
        r#"{"e0":0.5,"t":1.5,"a":1.2,"hbar":1.0,"m":0.8,"show_free_electron":true}"#,
    );
    let mut crystal = crate::tabs::solid_state::crystal_viewer::CrystalViewer::default();
    assert!(crate::framework::InteractiveTool::try_load_state(
        &mut crystal,
        r#"{"system":"SimpleCubic","lattice_constant":3.5,"yaw":0.75,"pitch":0.25,"zoom":60.0}"#
    )
    .is_ok());
    assert_tool_roundtrip(
        &mut crate::tabs::solid_state::crystal_viewer::CrystalViewer::default(),
        r#"{"system":"SimpleCubic","lattice_constant":3.5,"camera":{"pitch":0.25,"yaw":0.75,"zoom":1.2}}"#,
    );
}

#[cfg(feature = "applied")]
#[test]
fn test_favoritism_tab_state_roundtrip() {
    let mut favoritism_tab = crate::tabs::favoritism::FavoritismTab::default();
    crate::tabs::ExplorerTab::load_state(
        &mut favoritism_tab,
        r#"{"selected_child_index":1,"new_child_name":"Sibling"}"#,
    );
    let migrated_favoritism_json =
        crate::tabs::ExplorerTab::save_state(&favoritism_tab).expect("favoritism save_state");
    assert_tab_roundtrip(
        &mut crate::tabs::favoritism::FavoritismTab::default(),
        &migrated_favoritism_json,
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_traceability_tab_state_roundtrip() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let manifest_dir_raw =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set");
    let valid_repo_dir = oxidize_core::path_utils::normalize_path(
        std::path::Path::new(&manifest_dir_raw)
            .canonicalize()
            .expect("canonical manifest dir"),
    );
    assert_tab_roundtrip(
        &mut crate::tabs::traceability::TraceabilityTab::default(),
        &serde_json::json!({ "repo_path": valid_repo_dir }).to_string(),
    );
}
