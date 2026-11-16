#[cfg(test)]
mod tests {
    #[test]
    fn test_additional_interactive_tools() {
        use crate::framework::InteractiveTool;
        let ctx = eframe::egui::Context::default();

        macro_rules! check_tool {
            ($tool_ty:path) => {
                let mut tool = <$tool_ty>::default();
                let _ = tool.name();
                let _ = tool.theory();
                let _ = tool.export_datasets();
                let _ = tool.export_figure_options();
                let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
                    tool.show(ctx);
                });
            };
        }

        check_tool!(crate::tabs::ai::q_table_inspector::QTableInspectorTool);
        check_tool!(crate::tabs::ai::attention_maps::AttentionMapsTool);
        check_tool!(crate::tabs::ai::tokenization::TokenizationTool);
        check_tool!(crate::tabs::financial_math::bet_size::BetSizeCalculatorTool);
        check_tool!(crate::tabs::financial_math::bankroll_growth::BankrollGrowthTool);
        check_tool!(crate::tabs::battery_degradation::lifetime_estimator::LifetimeEstimatorTool);
        check_tool!(crate::tabs::battery_degradation::capacity_fade::CapacityFadeTool);
        check_tool!(crate::tabs::solid_state::band_structure::BandStructureTool);
        check_tool!(crate::tabs::solid_state::ising::IsingModelTool);
        check_tool!(crate::tabs::quantum::clebsch::ClebschGordanTool);
        check_tool!(crate::tabs::graph_theory::network_metrics::NetworkMetricsTool);
        check_tool!(crate::tabs::clinical_trials::survival::SurvivalAnalysisTool);
        check_tool!(crate::tabs::clinical_trials::sample_size::SampleSizeCalculatorTool);
        check_tool!(crate::tabs::clinical_trials::randomization::RandomizationTool);
        check_tool!(crate::tabs::geometry_topology::vietoris_rips::VietorisRipsTool);
        check_tool!(crate::tabs::climate::cera::CeraTool);
        check_tool!(crate::tabs::climate::co2_projections::Co2ProjectionsTool);
        check_tool!(crate::tabs::climate::temperature::TemperatureAnomaliesTool);
        check_tool!(crate::tabs::fluid_dynamics::potential_flow::PotentialFlowTool);
        check_tool!(crate::tabs::fluid_dynamics::turbulence::TurbulenceTool);
        check_tool!(crate::tabs::ai::activation_functions::ActivationFunctionsTool);
        check_tool!(crate::tabs::ai::training_monitor::TrainingMonitorTool);
        check_tool!(crate::tabs::ai::loss_landscape::LossLandscapeTool);
        check_tool!(crate::tabs::ai::grid_world::GridWorldTool);
        check_tool!(crate::tabs::medical::beam_profiling::BeamProfilingTool);
        check_tool!(crate::tabs::medical::dose::DoseCalculationTool);
        check_tool!(crate::tabs::epidemiology::network_propagation::NetworkPropagationTool);
        check_tool!(crate::tabs::number_theory::ualbf_widget::UalbfWidget);
        check_tool!(crate::tabs::number_theory::partitions_widget::PartitionsWidget);
        check_tool!(crate::tabs::number_theory::prime_spiral::PrimeSpiralWidget);
        check_tool!(crate::tabs::neuroscience::spike_analysis::SpikeAnalysisTool);
        check_tool!(crate::tabs::neuroscience::neural_network_viz::NeuralNetworkVizTool);
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
                meta.name
            );
            assert!(
                !desc.contains("Theoretical context not available"),
                "Tool {} still has placeholder theory_description: {}",
                meta.name,
                desc
            );

            assert!(
                !phonetic.trim().is_empty(),
                "Tool {} has empty phonetic_description",
                meta.name
            );
            assert!(
                !phonetic.contains("Theoretical context not available"),
                "Tool {} still has placeholder phonetic_description: {}",
                meta.name,
                phonetic
            );

            assert!(
                !citation.trim().is_empty(),
                "Tool {} has empty theory_citation",
                meta.name
            );
            assert_ne!(
                citation.trim(),
                "Uncited",
                "Tool {} has placeholder citation 'Uncited'",
                meta.name
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
    fn test_theory_context_portal_citation_registration() {
        use crate::framework::SimulationFramework;
        let ctx = eframe::egui::Context::default();

        let mut framework = SimulationFramework::new("medical");
        framework.show_theory_portal = true;

        // Select Dose Calculation tool if present
        let dose_idx = framework
            .available_tools
            .iter()
            .position(|m| m.name.contains("Dose"))
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
}
