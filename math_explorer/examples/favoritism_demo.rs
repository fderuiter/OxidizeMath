//! Example target for favoritism demo.
use math_explorer::applied::favoritism::{FavoritismInputs, calculate_favoritism_score};

fn main() {
    let mut inputs = FavoritismInputs::default();
    inputs.personality.wealth = 10.0; // High wealth factor
    inputs.social.helped_during_crisis = true; // High social utility

    let score = calculate_favoritism_score(&inputs);
    println!("Favoritism Score: {}", score); // Higher is better
}
