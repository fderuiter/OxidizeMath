#![allow(missing_docs)]
use math_explorer::applied::lorahub::{LoraEnsemble, LoraStateDict};
use math_explorer::nalgebra::DMatrix;
use std::time::Instant;

fn main() {
    let size = 1000;
    let num_modules = 10;

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
