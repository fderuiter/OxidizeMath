#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::StarterPresetMetadata;

pub const PRESET_SHIN_THREE_WAY_DEVIG: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Market Devigging",
    "domain": "sports_analytics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "three_way_market": true,
    "odds_home": 1.85,
    "odds_draw": 3.60,
    "odds_away": 4.20,
    "offered_bet_odds": 2.05,
    "bankroll": 1500.0,
    "method": "Shin"
  }
}"#;

pub const PRESET_DIXON_COLES_DERBY: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Match Score Heatmap",
    "domain": "sports_analytics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "variant": "DixonColes",
    "lambda_home": 1.72,
    "lambda_away": 1.15,
    "dixon_coles_rho": -0.13,
    "shared_lambda3": 0.14,
    "goal_line": 2.5,
    "handicap_line": -0.5,
    "offered_home_odds": 2.10,
    "bankroll": 1500.0
  }
}"#;

pub const PRESET_QB_WR_STACK_SGP: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Same-Game Parlay Copula",
    "domain": "sports_analytics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "three_legs": true,
    "leg1_prob": 0.58,
    "leg2_prob": 0.55,
    "leg3_prob": 0.52,
    "rho_12": 0.42,
    "rho_13": 0.28,
    "rho_23": 0.24,
    "offered_sgp_odds": 4.75,
    "bankroll": 1000.0
  }
}"#;

pub const PRESET_THREE_POINT_SHOOTER_ZIP: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "ZIP Count Props",
    "domain": "sports_analytics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "zero_inflation_pi": 0.25,
    "active_rate_lambda": 2.85,
    "prop_line_half": 2,
    "offered_over_odds": 2.20,
    "bankroll": 1000.0
  }
}"#;

pub const PRESET_FOURTH_QUARTER_COMEBACK_OU: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Live Momentum (OU)",
    "domain": "sports_analytics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "current_margin": -6.0,
    "pregame_spread_mu": 4.5,
    "reversion_theta": 1.85,
    "volatility_sigma": 8.0,
    "remaining_time": 0.40,
    "offered_live_odds": 2.55,
    "rng_seed": 2026,
    "bankroll": 1000.0
  }
}"#;

pub const PRESET_CONTENDER_UPSET_RATINGS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Elo & Glicko-2 Ratings",
    "domain": "sports_analytics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "home_rating": 1640.0,
    "away_rating": 1530.0,
    "home_rd": 60.0,
    "away_rd": 125.0,
    "volatility": 0.06,
    "home_advantage": 55.0,
    "elo_k": 24.0,
    "margin_of_victory": 2,
    "offered_home_odds": 1.72,
    "bankroll": 1000.0
  }
}"#;

pub const PRESET_HALF_KELLY_CLV_SWEET_SPOT: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "CLV & Kelly Risk of Ruin",
    "domain": "sports_analytics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "bet_odds": 2.16,
    "close_home_odds": 1.87,
    "close_away_odds": 2.03,
    "kelly_multiplier": 0.50,
    "drawdown_ratio": 0.50,
    "bankroll_units": 50.0,
    "devig_method": "Shin"
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "sports_shin_three_way_devig",
        title: "Shin 1X2 Insider Devigging",
        description: "Strips favorite-longshot bias from 3-way football odds via Shin's insider parameter z*.",
        category: "Sports Analytics",
        icon: "📊",
        domain: "sports_analytics",
        tool_name: Some("MarketDeviggingTool"),
        preset_generator: || PRESET_SHIN_THREE_WAY_DEVIG,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "sports_dixon_coles_derby",
        title: "Dixon-Coles Low-Score Derby Grid",
        description: "8x8 bivariate score heatmap with negative rho elevating 0-0 and 1-1 draw probabilities.",
        category: "Sports Analytics",
        icon: "⚽",
        domain: "sports_analytics",
        tool_name: Some("MatchScoreHeatmapTool"),
        preset_generator: || PRESET_DIXON_COLES_DERBY,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "sports_qb_wr_stack_sgp",
        title: "Correlated 3-Leg Passing Stack SGP",
        description: "Gaussian copula Cholesky 3x3 elliptope pricing positively correlated QB/WR parlay legs.",
        category: "Sports Analytics",
        icon: "🔗",
        domain: "sports_analytics",
        tool_name: Some("SgpCopulaTool"),
        preset_generator: || PRESET_QB_WR_STACK_SGP,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "sports_three_point_zip",
        title: "Bench-Risk 3PT Shooter ZIP Prop",
        description: "Zero-Inflated Poisson Over 2.5 prop contrasting structural zeros against equi-mean Poisson.",
        category: "Sports Analytics",
        icon: "🏀",
        domain: "sports_analytics",
        tool_name: Some("ZipCountPropsTool"),
        preset_generator: || PRESET_THREE_POINT_SHOOTER_ZIP,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "sports_fourth_quarter_ou",
        title: "Favorite Trailing In-Play OU Comeback",
        description: "Ornstein-Uhlenbeck mean-reverting score margin SDE sizing live comeback moneylines.",
        category: "Sports Analytics",
        icon: "⏱️",
        domain: "sports_analytics",
        tool_name: Some("LiveMomentumTool"),
        preset_generator: || PRESET_FOURTH_QUARTER_COMEBACK_OU,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "sports_contender_upset_ratings",
        title: "High-RD Challenger vs Established Favorite",
        description: "Compares Elo Margin-of-Victory updates against Glicko-2 RD-attenuated win probabilities.",
        category: "Sports Analytics",
        icon: "🏆",
        domain: "sports_analytics",
        tool_name: Some("SportsRatingsTool"),
        preset_generator: || PRESET_CONTENDER_UPSET_RATINGS,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "sports_half_kelly_clv",
        title: "Closing Line Value & Half-Kelly Shield",
        description: "Captures 75% of maximum log-growth at 50% stake while squaring continuous drawdown survival.",
        category: "Sports Analytics",
        icon: "🛡️",
        domain: "sports_analytics",
        tool_name: Some("ClvRuinTool"),
        preset_generator: || PRESET_HALF_KELLY_CLV_SWEET_SPOT,
    }
}

// [cite:sports_quantitative_modeling]
