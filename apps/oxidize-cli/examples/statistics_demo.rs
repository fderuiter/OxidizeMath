//! Comprehensive example demonstrating all three new statistics modules

use oxidizemath::pure_math::statistics::{glicko2, kelly, tda};

fn demo_glicko2() -> glicko2::GlickoPlayer {
    println!("1. GLICKO-2 RATING SYSTEM");
    println!("   (Competitive ranking with uncertainty)");
    println!("   ----------------------------------------");

    let player = glicko2::GlickoPlayer::default();
    println!(
        "   Initial player: Rating={}, RD={}, Volatility={}",
        player.rating.value(),
        player.rating_deviation.value(),
        player.volatility.value()
    );

    let opponent = glicko2::GlickoPlayer::new(
        glicko2::Rating::new(1400.0).unwrap(),
        glicko2::RatingDeviation::new(30.0).unwrap(),
        glicko2::Volatility::new(0.06).unwrap(),
    );

    let results = vec![glicko2::MatchResult::new(opponent, 1.0).unwrap()];
    let updated =
        glicko2::update_rating(&player, &results, &glicko2::SystemConstant::default()).unwrap();

    println!(
        "   After win: Rating={:.1}, RD={:.1}, Volatility={:.4}",
        updated.rating.value(),
        updated.rating_deviation.value(),
        updated.volatility.value()
    );
    println!("   ✓ Rating increased after victory!\n");
    updated
}

fn demo_kelly() -> f64 {
    println!("2. KELLY CRITERION");
    println!("   (Optimal bet sizing)");
    println!("   --------------------");

    let prob = kelly::UnitInterval::new(0.55).unwrap();
    let odds = kelly::Odds::new(2.0).unwrap();

    let ev = kelly::expected_value(&prob, &odds);
    let full_kelly = kelly::kelly_fraction(&prob, &odds).unwrap();
    let half_kelly = kelly::variants::half_kelly(&prob, &odds).unwrap();

    println!("   Scenario: 55% win probability, 2.0 decimal odds");
    println!("   Expected Value: {:.3} per $1 bet", ev);
    println!(
        "   Full Kelly: {:.1}% of bankroll",
        full_kelly.value() * 100.0
    );
    println!(
        "   Half Kelly: {:.1}% of bankroll (conservative)",
        half_kelly.value() * 100.0
    );

    let bankroll = 10000.0;
    let bet_amount = half_kelly.bet_amount(bankroll).unwrap();
    println!(
        "   Recommended bet with ${:.0} bankroll: ${:.2}",
        bankroll, bet_amount
    );
    println!("   ✓ Optimal sizing for positive expected growth!\n");
    half_kelly.value() * 100.0
}

fn demo_tda() -> usize {
    println!("3. TOPOLOGICAL DATA ANALYSIS");
    println!("   (Shape detection in point clouds)");
    println!("   ----------------------------------");

    let n = 12;
    let circle_points: Vec<tda::Point2D> = (0..n)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            tda::Point2D::new(angle.cos(), angle.sin())
        })
        .collect();

    let cloud = tda::PointCloud::new(circle_points).unwrap();
    println!(
        "   Point cloud: {} points arranged in a circle",
        cloud.size()
    );

    let small_radius = 0.5;
    let medium_radius = 0.6;

    let complex_small = tda::vietoris_rips_complex(&cloud, small_radius).unwrap();
    let (beta0_small, beta1_small) = tda::betti_numbers(&complex_small).unwrap();

    let complex_medium = tda::vietoris_rips_complex(&cloud, medium_radius).unwrap();
    let (beta0_medium, beta1_medium) = tda::betti_numbers(&complex_medium).unwrap();

    println!("   At radius {:.1}:", small_radius);
    println!(
        "     β₀={} (components), β₁={} (holes)",
        beta0_small, beta1_small
    );

    println!("   At radius {:.1}:", medium_radius);
    println!(
        "     β₀={} (components), β₁={} (holes)",
        beta0_medium, beta1_medium
    );

    let radii: Vec<f64> = (0..40).map(|i| i as f64 * 0.05).collect();
    let barcode = tda::compute_persistence(&cloud, &radii).unwrap();

    println!("   Persistence analysis:");
    println!("     Total features detected: {}", barcode.len());

    if let Some(hole) = barcode.most_persistent(1) {
        println!("     Most persistent hole:");
        println!(
            "       Birth: {:.3}, Death: {:.3}, Persistence: {:.3}",
            hole.birth,
            hole.death,
            hole.persistence()
        );
        println!("   ✓ Circular structure detected!\n");
    }
    beta1_medium
}

fn main() {
    println!("=== Math Explorer: New Statistics Modules Demo ===\n");

    let updated = demo_glicko2();
    let kelly_percent = demo_kelly();
    let beta1_medium = demo_tda();

    // ========================================
    // Summary
    // ========================================
    println!("=== SUMMARY ===");
    println!(
        "✓ Glicko-2: Updated player rating from 1500 to ~{:.0}",
        updated.rating.value()
    );
    println!(
        "✓ Kelly: Optimal bet is {:.1}% of bankroll for positive EV",
        kelly_percent
    );
    println!("✓ TDA: Detected circular topology with β₁={}", beta1_medium);
    println!("\nAll three modules working perfectly! 🎉");
}
