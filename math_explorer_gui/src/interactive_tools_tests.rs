#[cfg(test)]
mod roundtrip;

#[cfg(test)]
mod tests {
    #[cfg(any(
        feature = "ai",
        feature = "applied",
        feature = "biology",
        feature = "climate",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
    fn check_one<T: crate::framework::InteractiveTool + Default>(ctx: &eframe::egui::Context) {
        let mut tool = T::default();
        let _ = tool.name();
        let _ = tool.theory();
        let _ = tool.export_datasets();
        let _ = tool.export_figure_options();
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            tool.show(ctx);
        });
    }

    #[cfg(any(feature = "applied", feature = "pure_math"))]
    fn check_math_and_applied_tools(ctx: &eframe::egui::Context) {
        #[cfg(feature = "pure_math")]
        {
            check_one::<crate::tabs::financial_math::bet_size::BetSizeCalculatorTool>(ctx);
            check_one::<crate::tabs::financial_math::bankroll_growth::BankrollGrowthTool>(ctx);
            check_one::<crate::tabs::financial_math::correlated_risk::CorrelatedRiskScenarioTool>(
                ctx,
            );
            check_one::<crate::tabs::financial_math::market_devig::MarketDeviggingTool>(ctx);
            check_one::<crate::tabs::financial_math::market_execution::MarketExecutionTool>(ctx);
            check_one::<crate::tabs::financial_math::monte_carlo_workbench::MonteCarloWorkbenchTool>(
                ctx,
            );
            check_one::<crate::tabs::graph_theory::network_metrics::NetworkMetricsTool>(ctx);
            check_one::<crate::tabs::geometry_topology::vietoris_rips::VietorisRipsTool>(ctx);
            check_one::<crate::tabs::geometry_topology::hyperbolic_geometry::HyperbolicGeometryTool>(
                ctx,
            );
            check_one::<crate::tabs::algorithmic_information::state::FractranMachineTool>(ctx);
            check_one::<crate::tabs::number_theory::fractran_machine::FractranMachineTool>(ctx);
            check_one::<crate::tabs::number_theory::ualbf_widget::UalbfWidget>(ctx);
            check_one::<crate::tabs::number_theory::partitions_widget::PartitionsWidget>(ctx);
            check_one::<crate::tabs::number_theory::prime_spiral::PrimeSpiralWidget>(ctx);
        }
        #[cfg(feature = "applied")]
        {
            check_one::<crate::tabs::game_theory::auction::MyersonAuctionTool>(ctx);
            check_one::<crate::tabs::game_theory::glicko2::Glicko2TournamentTool>(ctx);
            check_one::<crate::tabs::game_theory::markov::MarkovSequenceWorkbenchTool>(ctx);
            check_one::<crate::tabs::battery_degradation::lifetime_estimator::LifetimeEstimatorTool>(
                ctx,
            );
            check_one::<crate::tabs::battery_degradation::capacity_fade::CapacityFadeTool>(ctx);
            check_one::<crate::tabs::clinical_trials::pharmacokinetics::PharmacokineticsTool>(ctx);
            check_one::<crate::tabs::clinical_trials::survival::SurvivalAnalysisTool>(ctx);
            check_one::<crate::tabs::clinical_trials::sample_size::SampleSizeCalculatorTool>(ctx);
            check_one::<crate::tabs::clinical_trials::randomization::RandomizationTool>(ctx);
            check_one::<crate::tabs::clinical_trials::win_ratio::WinRatioStressTesterTool>(ctx);
            check_one::<crate::tabs::clinical_trials::win_ratio_workbench::WinRatioWorkbenchTool>(
                ctx,
            );
        }
    }

    #[cfg(any(
        feature = "ai",
        feature = "biology",
        feature = "climate",
        feature = "epidemiology",
        feature = "physics"
    ))]
    fn check_science_and_ai_tools(ctx: &eframe::egui::Context) {
        #[cfg(feature = "ai")]
        {
            check_one::<crate::tabs::ai::q_table_inspector::QTableInspectorTool>(ctx);
            check_one::<crate::tabs::ai::attention_maps::AttentionMapsTool>(ctx);
            check_one::<crate::tabs::ai::tokenization::TokenizationTool>(ctx);
            check_one::<crate::tabs::ai::activation_functions::ActivationFunctionsTool>(ctx);
            check_one::<crate::tabs::ai::gaussian_splatting::GaussianSplattingTool>(ctx);
            check_one::<crate::tabs::ai::training_monitor::TrainingMonitorTool>(ctx);
            check_one::<crate::tabs::ai::loss_landscape::LossLandscapeTool>(ctx);
            check_one::<crate::tabs::ai::grid_world::GridWorldTool>(ctx);
            check_one::<crate::tabs::ai::closed_loop_control::ClosedLoopControlTool>(ctx);
            check_one::<crate::tabs::ai::grpo_alignment_lab::GrpoAlignmentLabTool>(ctx);
            check_one::<crate::tabs::ai::lorahub_simplex::LoraHubSimplexTool>(ctx);
            check_one::<crate::tabs::ai::sds_nerf_distillation::SdsNeRfDistillationTool>(ctx);
        }
        #[cfg(feature = "physics")]
        {
            check_one::<crate::tabs::solid_state::band_structure::BandStructureTool>(ctx);
            check_one::<crate::tabs::solid_state::ising::IsingModelTool>(ctx);
            check_one::<crate::tabs::quantum::clebsch::ClebschGordanTool>(ctx);
            check_one::<crate::tabs::fluid_dynamics::potential_flow::PotentialFlowTool>(ctx);
            check_one::<crate::tabs::fluid_dynamics::turbulence::TurbulenceTool>(ctx);
            check_one::<crate::tabs::medical::beam_profiling::BeamProfilingTool>(ctx);
            check_one::<crate::tabs::medical::dose::DoseCalculationTool>(ctx);
            check_one::<crate::tabs::medical::pharmacokinetic_regimen::PharmacokineticRegimenTool>(
                ctx,
            );
            check_one::<crate::tabs::medical::radar_gating::RadarGatingTool>(ctx);
            check_one::<crate::tabs::chaos::hamiltonian::HamiltonianChaosLab>(ctx);
        }
        #[cfg(feature = "climate")]
        {
            check_one::<crate::tabs::climate::cera::CeraTool>(ctx);
            check_one::<crate::tabs::climate::co2_projections::Co2ProjectionsTool>(ctx);
            check_one::<crate::tabs::climate::temperature::TemperatureAnomaliesTool>(ctx);
        }
        #[cfg(feature = "epidemiology")]
        check_one::<crate::tabs::epidemiology::network_propagation::NetworkPropagationTool>(ctx);
        #[cfg(feature = "biology")]
        {
            check_one::<crate::tabs::morphogenesis::lenia::LeniaTool>(ctx);
            check_one::<crate::tabs::neuroscience::spike_analysis::SpikeAnalysisTool>(ctx);
            check_one::<crate::tabs::neuroscience::neural_network_viz::NeuralNetworkVizTool>(ctx);
            check_one::<crate::tabs::neuroscience::cortical_morphometry::CorticalMorphometryTool>(
                ctx,
            );
            #[cfg(feature = "pure_math")]
            check_one::<
                crate::tabs::geometry_topology::surface_morphogenesis::SurfaceMorphogenesisTool,
            >(ctx);
        }
    }

    #[cfg(any(
        feature = "ai",
        feature = "applied",
        feature = "biology",
        feature = "climate",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
    #[test]
    fn test_additional_interactive_tools() {
        let ctx = eframe::egui::Context::default();
        #[cfg(any(feature = "applied", feature = "pure_math"))]
        check_math_and_applied_tools(&ctx);
        #[cfg(any(
            feature = "ai",
            feature = "biology",
            feature = "climate",
            feature = "epidemiology",
            feature = "physics"
        ))]
        check_science_and_ai_tools(&ctx);
    }

    #[cfg(all(
        feature = "ai",
        feature = "applied",
        feature = "biology",
        feature = "climate",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
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

    #[cfg(feature = "physics")]
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
}
