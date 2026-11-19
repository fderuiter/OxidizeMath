#[cfg(test)]
mod tests {
    #[cfg(any(
        feature = "ai",
        feature = "pure_math",
        feature = "applied",
        feature = "physics",
        feature = "climate",
        feature = "epidemiology",
        feature = "biology"
    ))]
    fn check_tool_instance<T: Default + crate::framework::InteractiveTool>(
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
    #[cfg(feature = "ai")]
    fn test_additional_interactive_tools_ai() {
        let ctx = eframe::egui::Context::default();
        check_tool_instance::<crate::tabs::ai::q_table_inspector::QTableInspectorTool>(&ctx);
        check_tool_instance::<crate::tabs::ai::attention_maps::AttentionMapsTool>(&ctx);
        check_tool_instance::<crate::tabs::ai::tokenization::TokenizationTool>(&ctx);
        check_tool_instance::<crate::tabs::ai::activation_functions::ActivationFunctionsTool>(&ctx);
        check_tool_instance::<crate::tabs::ai::training_monitor::TrainingMonitorTool>(&ctx);
        check_tool_instance::<crate::tabs::ai::loss_landscape::LossLandscapeTool>(&ctx);
        check_tool_instance::<crate::tabs::ai::grid_world::GridWorldTool>(&ctx);
    }

    #[test]
    #[cfg(feature = "pure_math")]
    fn test_additional_interactive_tools_pure_math() {
        let ctx = eframe::egui::Context::default();
        check_tool_instance::<crate::tabs::financial_math::bet_size::BetSizeCalculatorTool>(&ctx);
        check_tool_instance::<crate::tabs::financial_math::bankroll_growth::BankrollGrowthTool>(
            &ctx,
        );
        check_tool_instance::<crate::tabs::graph_theory::network_metrics::NetworkMetricsTool>(&ctx);
        check_tool_instance::<crate::tabs::geometry_topology::vietoris_rips::VietorisRipsTool>(
            &ctx,
        );
        check_tool_instance::<crate::tabs::number_theory::ualbf_widget::UalbfWidget>(&ctx);
        check_tool_instance::<crate::tabs::number_theory::partitions_widget::PartitionsWidget>(
            &ctx,
        );
        check_tool_instance::<crate::tabs::number_theory::prime_spiral::PrimeSpiralWidget>(&ctx);
    }

    #[test]
    #[cfg(feature = "applied")]
    fn test_additional_interactive_tools_applied() {
        let ctx = eframe::egui::Context::default();
        check_tool_instance::<
            crate::tabs::battery_degradation::lifetime_estimator::LifetimeEstimatorTool,
        >(&ctx);
        check_tool_instance::<crate::tabs::battery_degradation::capacity_fade::CapacityFadeTool>(
            &ctx,
        );
        check_tool_instance::<crate::tabs::clinical_trials::survival::SurvivalAnalysisTool>(&ctx);
        check_tool_instance::<crate::tabs::clinical_trials::sample_size::SampleSizeCalculatorTool>(
            &ctx,
        );
        check_tool_instance::<crate::tabs::clinical_trials::randomization::RandomizationTool>(&ctx);
    }

    #[test]
    #[cfg(feature = "physics")]
    fn test_additional_interactive_tools_physics() {
        let ctx = eframe::egui::Context::default();
        check_tool_instance::<crate::tabs::solid_state::band_structure::BandStructureTool>(&ctx);
        check_tool_instance::<crate::tabs::solid_state::ising::IsingModelTool>(&ctx);
        check_tool_instance::<crate::tabs::quantum::clebsch::ClebschGordanTool>(&ctx);
        check_tool_instance::<crate::tabs::fluid_dynamics::potential_flow::PotentialFlowTool>(&ctx);
        check_tool_instance::<crate::tabs::fluid_dynamics::turbulence::TurbulenceTool>(&ctx);
        check_tool_instance::<crate::tabs::medical::beam_profiling::BeamProfilingTool>(&ctx);
        check_tool_instance::<crate::tabs::medical::dose::DoseCalculationTool>(&ctx);
    }

    #[test]
    #[cfg(feature = "climate")]
    fn test_additional_interactive_tools_climate() {
        let ctx = eframe::egui::Context::default();
        check_tool_instance::<crate::tabs::climate::cera::CeraTool>(&ctx);
        check_tool_instance::<crate::tabs::climate::co2_projections::Co2ProjectionsTool>(&ctx);
        check_tool_instance::<crate::tabs::climate::temperature::TemperatureAnomaliesTool>(&ctx);
    }

    #[test]
    #[cfg(feature = "epidemiology")]
    fn test_additional_interactive_tools_epidemiology() {
        let ctx = eframe::egui::Context::default();
        check_tool_instance::<crate::tabs::epidemiology::network_propagation::NetworkPropagationTool>(
            &ctx,
        );
    }

    #[test]
    #[cfg(feature = "biology")]
    fn test_additional_interactive_tools_biology() {
        let ctx = eframe::egui::Context::default();
        check_tool_instance::<crate::tabs::neuroscience::spike_analysis::SpikeAnalysisTool>(&ctx);
        check_tool_instance::<crate::tabs::neuroscience::neural_network_viz::NeuralNetworkVizTool>(
            &ctx,
        );
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
        let _ = count;

        #[cfg(all(
            feature = "pure_math",
            feature = "applied",
            feature = "ai",
            feature = "biology",
            feature = "climate",
            feature = "epidemiology",
            feature = "physics"
        ))]
        assert!(
            count >= 40,
            "Expected at least 40 registered tools, got {}",
            count
        );
    }

    #[test]
    #[cfg(feature = "physics")]
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
    fn test_traceability_tab_state_roundtrip() {
        assert_tab_roundtrip(
            &mut crate::tabs::traceability::TraceabilityTab::default(),
            r#"{"repo_path":"/tmp/mock_repo"}"#,
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
            &mut crate::tabs::solid_state::band_structure::BandStructureTool::default(),
            r#"{"e0":0.5,"t":1.5,"a":1.2,"hbar":1.0,"m":0.8,"show_free_electron":true}"#,
        );
        assert_tool_roundtrip(
            &mut crate::tabs::solid_state::crystal_viewer::CrystalViewer::default(),
            r#"{"system":"SimpleCubic","lattice_constant":3.5,"yaw":0.75,"pitch":0.25,"zoom":60.0}"#,
        );
        assert_tab_roundtrip(
            &mut crate::tabs::favoritism::FavoritismTab::default(),
            r#"{"selected_child_index":1,"new_child_name":"Sibling"}"#,
        );
        assert_tab_roundtrip(
            &mut crate::tabs::traceability::TraceabilityTab::default(),
            r#"{"repo_path":"/tmp/mock_repo"}"#,
        );
    }
}
