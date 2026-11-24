//! Cross-domain zero-allocation and `< 8 ms` frame-budget verification for fused 2D FitzHugh-Nagumo PDE and 3-lead dipole integration.

#[cfg(all(feature = "biology", feature = "physics"))]
mod coupled_verification {
    use math_explorer::biology::neuroscience::{DEFAULT_SHEET_DIM, ExcitableSheet2D};
    use math_explorer::physics::medical::bioelectric::{DipoleLeadArray, LeadReadoutMode};
    use math_explorer::verified_engine::allocator::VerifiedAllocator;
    use math_explorer::verified_engine::engine::VerifiedEngine;
    #[cfg(not(target_arch = "wasm32"))]
    use std::time::Instant;

    #[global_allocator]
    static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

    #[test]
    fn test_fused_fhn_96x96_and_3_lead_dipole_zero_alloc_under_8ms_budget() {
        let mut sheet = ExcitableSheet2D::new_default();
        sheet.trigger_s1_s2_spiral();
        let leads = DipoleLeadArray::new_default(DEFAULT_SHEET_DIM, DEFAULT_SHEET_DIM);

        // Warm up caches prior to budget check.
        sheet.step(0.1);
        let _ = leads.evaluate_sheet(
            sheet.voltage(),
            sheet.scar_mask(),
            LeadReadoutMode::BipolarPseudoEcg,
        );

        #[cfg(not(target_arch = "wasm32"))]
        #[allow(clippy::disallowed_methods)]
        let start = Instant::now();
        let (ecg, metrics) = VerifiedEngine::run_verified(|| {
            sheet.step(0.1);
            leads.evaluate_sheet(
                sheet.voltage(),
                sheet.scar_mask(),
                LeadReadoutMode::BipolarPseudoEcg,
            )
        });

        assert!(ecg.iter().all(|v| v.is_finite()));
        assert!(metrics.function_calls > 0);
        assert!(metrics.arithmetic_ops > 0);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let elapsed = start.elapsed();
            let budget_secs = if cfg!(debug_assertions) { 0.250 } else { 0.008 };
            assert!(
                elapsed.as_secs_f64() < budget_secs,
                "96x96 FHN + 3-lead dipole step exceeded {:.0} ms budget: {:?}",
                budget_secs * 1000.0,
                elapsed
            );
        }
    }
}
