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
}
