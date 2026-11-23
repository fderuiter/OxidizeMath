//! Example target for glicko2 demo.
use math_explorer::pure_math::statistics::glicko2::{
    GlickoPlayer, MatchResult, Rating, RatingDeviation, SystemConstant, Volatility, update_rating,
};

fn main() {
    // Player wins against a 1700-rated opponent
    let player = GlickoPlayer::default(); // 1500 rating
    let opponent = GlickoPlayer::new(
        Rating::new(1700.0).unwrap(),
        RatingDeviation::new(300.0).unwrap(),
        Volatility::default(),
    );

    let result = MatchResult::new(opponent, 1.0).unwrap(); // Win
    let tau = SystemConstant::new(0.5).unwrap();

    let new_player = update_rating(&player, &[result], &tau).unwrap();
    println!("New Rating: {:.0}", new_player.rating.value());
}
