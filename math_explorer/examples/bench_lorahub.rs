//! Example target for bench lorahub.
#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]
use math_explorer::applied::lorahub::{LoraEnsemble, LoraStateDict};
use math_explorer::nalgebra::DMatrix;
use std::time::Instant;

fn main() {
    let smoke = std::env::var("MATH_EXPLORER_SMOKE")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let (size, num_modules) = if smoke { (64, 4) } else { (1000, 10) };

    println!(
        "Generating {} LoRA modules with {}x{} matrices...",
        num_modules, size, size
    );

    let mut modules = Vec::new();
    for _ in 0..num_modules {
        let mut dict = LoraStateDict::new();
        // Create a few layers
        dict.insert(
            "layer1.weight".to_string(),
            DMatrix::from_element(size, size, 1.0),
        );
        dict.insert(
            "layer2.weight".to_string(),
            DMatrix::from_element(size, size, 2.0),
        );
        dict.insert(
            "layer3.weight".to_string(),
            DMatrix::from_element(size, size, 3.0),
        );
        modules.push(dict);
    }

    let ensemble = LoraEnsemble::new(modules);
    let weights = vec![1.0 / num_modules as f64; num_modules];

    println!("Benchmarking combine...");
    let start = Instant::now();
    let _combined = ensemble.combine(&weights).expect("Combination failed");
    let duration = start.elapsed();

    println!("Time taken: {:.2?}", duration);
}
