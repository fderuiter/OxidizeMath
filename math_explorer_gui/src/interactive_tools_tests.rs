#[cfg(test)]
mod tests {
    fn exercise_interactive_tool<T: crate::framework::InteractiveTool + Default>(
        ctx: &eframe::egui::Context,
    ) {
        let mut tool = T::default();
        let _ = tool.name();
        let _ = tool.theory();
        let _ = tool.export_datasets();
        let _ = tool.export_figure_options();
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            tool.show(ctx);
        });
    }

    #[test]
    fn test_additional_interactive_tools() {
        let ctx = eframe::egui::Context::default();

        exercise_interactive_tool::<crate::tabs::ai::q_table_inspector::QTableInspectorTool>(&ctx);
        exercise_interactive_tool::<crate::tabs::ai::attention_maps::AttentionMapsTool>(&ctx);
        exercise_interactive_tool::<crate::tabs::ai::tokenization::TokenizationTool>(&ctx);
        exercise_interactive_tool::<crate::tabs::financial_math::bet_size::BetSizeCalculatorTool>(
            &ctx,
        );
        exercise_interactive_tool::<crate::tabs::financial_math::bankroll_growth::BankrollGrowthTool>(
            &ctx,
        );
        exercise_interactive_tool::<
            crate::tabs::financial_math::correlated_risk::CorrelatedRiskScenarioTool,
        >(&ctx);
        exercise_interactive_tool::<crate::tabs::financial_math::market_devig::MarketDeviggingTool>(
            &ctx,
        );
        exercise_interactive_tool::<
            crate::tabs::financial_math::market_execution::MarketExecutionTool,
        >(&ctx);
        exercise_interactive_tool::<crate::tabs::game_theory::auction::MyersonAuctionTool>(&ctx);
        exercise_interactive_tool::<crate::tabs::game_theory::glicko2::Glicko2TournamentTool>(&ctx);
        exercise_interactive_tool::<crate::tabs::game_theory::markov::MarkovSequenceWorkbenchTool>(
            &ctx,
        );
        exercise_interactive_tool::<
            crate::tabs::battery_degradation::lifetime_estimator::LifetimeEstimatorTool,
        >(&ctx);
        exercise_interactive_tool::<
            crate::tabs::battery_degradation::capacity_fade::CapacityFadeTool,
        >(&ctx);
        exercise_interactive_tool::<crate::tabs::solid_state::band_structure::BandStructureTool>(
            &ctx,
        );
        exercise_interactive_tool::<crate::tabs::solid_state::ising::IsingModelTool>(&ctx);
        exercise_interactive_tool::<crate::tabs::quantum::clebsch::ClebschGordanTool>(&ctx);
        exercise_interactive_tool::<crate::tabs::graph_theory::network_metrics::NetworkMetricsTool>(
            &ctx,
        );
        exercise_interactive_tools_part_2(&ctx);
    }

    fn exercise_interactive_tools_part_2(ctx: &eframe::egui::Context) {
        exercise_interactive_tool::<
            crate::tabs::clinical_trials::pharmacokinetics::PharmacokineticsTool,
        >(ctx);
        exercise_interactive_tool::<crate::tabs::clinical_trials::survival::SurvivalAnalysisTool>(
            ctx,
        );
        exercise_interactive_tool::<
            crate::tabs::clinical_trials::sample_size::SampleSizeCalculatorTool,
        >(ctx);
        exercise_interactive_tool::<crate::tabs::clinical_trials::randomization::RandomizationTool>(
            ctx,
        );
        exercise_interactive_tool::<
            crate::tabs::clinical_trials::win_ratio::WinRatioStressTesterTool,
        >(ctx);
        exercise_interactive_tool::<
            crate::tabs::clinical_trials::win_ratio_workbench::WinRatioWorkbenchTool,
        >(ctx);
        exercise_interactive_tool::<crate::tabs::geometry_topology::vietoris_rips::VietorisRipsTool>(
            ctx,
        );
        exercise_interactive_tool::<
            crate::tabs::geometry_topology::hyperbolic_geometry::HyperbolicGeometryTool,
        >(ctx);
        exercise_interactive_tool::<crate::tabs::climate::cera::CeraTool>(ctx);
        exercise_interactive_tool::<crate::tabs::climate::co2_projections::Co2ProjectionsTool>(ctx);
        exercise_interactive_tool::<crate::tabs::climate::temperature::TemperatureAnomaliesTool>(
            ctx,
        );
        exercise_interactive_tool::<crate::tabs::fluid_dynamics::potential_flow::PotentialFlowTool>(
            ctx,
        );
        exercise_interactive_tool::<crate::tabs::fluid_dynamics::turbulence::TurbulenceTool>(ctx);
        exercise_interactive_tool::<crate::tabs::ai::activation_functions::ActivationFunctionsTool>(
            ctx,
        );
        exercise_interactive_tool::<crate::tabs::ai::gaussian_splatting::GaussianSplattingTool>(
            ctx,
        );
        exercise_interactive_tool::<crate::tabs::ai::training_monitor::TrainingMonitorTool>(ctx);
        exercise_interactive_tool::<crate::tabs::ai::loss_landscape::LossLandscapeTool>(ctx);
        exercise_interactive_tool::<crate::tabs::ai::grid_world::GridWorldTool>(ctx);
        exercise_interactive_tool::<crate::tabs::ai::closed_loop_control::ClosedLoopControlTool>(
            ctx,
        );
        exercise_interactive_tools_part_3(ctx);
    }

    fn exercise_interactive_tools_part_3(ctx: &eframe::egui::Context) {
        exercise_interactive_tool::<crate::tabs::medical::beam_profiling::BeamProfilingTool>(ctx);
        exercise_interactive_tool::<crate::tabs::medical::dose::DoseCalculationTool>(ctx);
        exercise_interactive_tool::<
            crate::tabs::medical::pharmacokinetic_regimen::PharmacokineticRegimenTool,
        >(ctx);
        exercise_interactive_tool::<crate::tabs::medical::radar_gating::RadarGatingTool>(ctx);
        exercise_interactive_tool::<
            crate::tabs::epidemiology::network_propagation::NetworkPropagationTool,
        >(ctx);
        exercise_interactive_tool::<crate::tabs::chaos::hamiltonian::HamiltonianChaosTool>(ctx);
        exercise_interactive_tool::<crate::tabs::chaos::billiards::DynamicalBilliardsTool>(ctx);
        exercise_interactive_tool::<crate::tabs::algorithmic_information::state::FractranMachineTool>(
            ctx,
        );
        exercise_interactive_tool::<
            crate::tabs::number_theory::fractran_machine::FractranMachineTool,
        >(ctx);
        exercise_interactive_tool::<crate::tabs::number_theory::ualbf_widget::UalbfWidget>(ctx);
        exercise_interactive_tool::<crate::tabs::number_theory::partitions_widget::PartitionsWidget>(
            ctx,
        );
        exercise_interactive_tool::<crate::tabs::number_theory::prime_spiral::PrimeSpiralWidget>(
            ctx,
        );
        exercise_interactive_tool::<crate::tabs::morphogenesis::lenia::LeniaTool>(ctx);
        exercise_interactive_tool::<crate::tabs::neuroscience::spike_analysis::SpikeAnalysisTool>(
            ctx,
        );
        exercise_interactive_tool::<
            crate::tabs::neuroscience::neural_network_viz::NeuralNetworkVizTool,
        >(ctx);
        exercise_interactive_tool::<
            crate::tabs::neuroscience::cortical_morphometry::CorticalMorphometryTool,
        >(ctx);
        #[cfg(all(feature = "pure_math", feature = "biology"))]
        exercise_interactive_tool::<
            crate::tabs::geometry_topology::surface_morphogenesis::SurfaceMorphogenesisTool,
        >(ctx);
    }

    #[test]
    fn test_all_registered_tools_theory_metadata() {
        let mut count = 0;
        for meta in inventory::iter::<crate::framework::ToolMetadata> {
            let tool = (meta.build)();
            let theory = tool.theory();
            let desc = theory.theory_description();
            let phonetic = theory.phonetic_description();
            let citation = theory.theory_citation();

            assert!(
                !desc.trim().is_empty(),
                "Tool {} has empty theory_description",
                meta.id
            );
            assert!(
                !desc.contains("Theoretical context not available"),
                "Tool {} still has placeholder theory_description: {}",
                meta.id,
                desc
            );

            assert!(
                !phonetic.trim().is_empty(),
                "Tool {} has empty phonetic_description",
                meta.id
            );
            assert!(
                !phonetic.contains("Theoretical context not available"),
                "Tool {} still has placeholder phonetic_description: {}",
                meta.id,
                phonetic
            );

            assert!(
                !citation.trim().is_empty(),
                "Tool {} has empty theory_citation",
                meta.id
            );
            assert_ne!(
                citation.trim(),
                "Uncited",
                "Tool {} has placeholder citation 'Uncited'",
                meta.id
            );

            count += 1;
        }

        assert!(
            count >= 40,
            "Expected at least 40 registered tools, got {}",
            count
        );
    }

    #[test]
    fn test_zero_registered_tools_render_placeholder_labels() {
        let screen_rect = Some(eframe::egui::Rect::from_min_size(
            eframe::egui::Pos2::ZERO,
            eframe::egui::vec2(1280.0, 800.0),
        ));
        for meta in inventory::iter::<crate::framework::ToolMetadata> {
            let mut tool = (meta.build)();
            let ctx = eframe::egui::Context::default();
            ctx.enable_accesskit();
            let out = ctx.run(
                eframe::egui::RawInput {
                    screen_rect,
                    ..Default::default()
                },
                |ctx| tool.show(ctx),
            );
            if let Some(ak) = out.platform_output.accesskit_update {
                for (_, node) in &ak.nodes {
                    let label = node.label().unwrap_or("");
                    assert!(
                        !label.contains("Placeholder"),
                        "Tool '{}' rendered placeholder label: '{}'",
                        meta.id,
                        label
                    );
                }
            }
            let mut stack: Vec<&eframe::egui::epaint::Shape> =
                out.shapes.iter().map(|c| &c.shape).collect();
            while let Some(shape) = stack.pop() {
                match shape {
                    eframe::egui::epaint::Shape::Text(t) => assert!(
                        !t.galley.text().contains("Placeholder"),
                        "Tool '{}' painted placeholder text: '{}'",
                        meta.id,
                        t.galley.text()
                    ),
                    eframe::egui::epaint::Shape::Vec(v) => stack.extend(v.iter()),
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn test_theory_context_portal_citation_registration() {
        use crate::framework::SimulationFramework;
        let ctx = eframe::egui::Context::default();

        let mut framework = SimulationFramework::new("medical");
        framework.show_theory_portal = true;

        // Select Dose Calculation tool if present
        let dose_idx = framework
            .available_tools
            .iter()
            .position(|m| m.display_name.contains("Dose") || m.id.contains("Dose"))
            .unwrap_or(0);
        framework.select_tool(dose_idx);

        let active_name = framework.active_tool.as_ref().unwrap().name().to_string();
        let expected_citation = framework
            .active_tool
            .as_ref()
            .unwrap()
            .theory()
            .theory_citation();

        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            framework.show(ctx, "test_portal_registration");
        });

        let registered =
            scientific_metadata::citation_registry::CitationRegistry::get(&active_name);
        assert_eq!(
            registered,
            Some(expected_citation),
            "Expected citation to be registered for {}",
            active_name
        );
    }

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
            &mut crate::tabs::clinical_trials::win_ratio_workbench::WinRatioWorkbenchTool::default(
            ),
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
}
