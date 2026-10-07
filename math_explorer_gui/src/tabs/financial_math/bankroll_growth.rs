use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::pure_math::statistics::kelly::{
    kelly_fraction, variants, BankrollFraction, UnitInterval, Odds,
};
use rand::Rng;

pub struct BankrollGrowthTool {
    initial_bankroll: f64,
    probability: f64,
    odds: f64,
    num_bets: usize,

    // Cached plot data
    full_kelly_points: Vec<[f64; 2]>,
    half_kelly_points: Vec<[f64; 2]>,
    quarter_kelly_points: Vec<[f64; 2]>,

    error_msg: Option<String>,
}

impl Default for BankrollGrowthTool {
    fn default() -> Self {
        let mut tool = Self {
            initial_bankroll: 1000.0,
            probability: 0.55,
            odds: 2.0, // Decimal odds (net profit multiplier + 1, wait Kelly odds b is net profit multiplier. The Odds type in math_explorer: Odds::new(b). Let's check.)
            num_bets: 100,
            full_kelly_points: vec![],
            half_kelly_points: vec![],
            quarter_kelly_points: vec![],
            error_msg: None,
        };
        tool.recalculate();
        tool
    }
}

impl BankrollGrowthTool {
    fn recalculate(&mut self) {
        self.full_kelly_points.clear();
        self.half_kelly_points.clear();
        self.quarter_kelly_points.clear();
        self.error_msg = None;

        let prob_result = UnitInterval::new(self.probability);
        let odds_result = Odds::new(self.odds);

        match (prob_result, odds_result) {
            (Ok(p), Ok(o)) => {
                let full_kelly_res = kelly_fraction(&p, &o);
                let half_kelly_res = variants::half_kelly(&p, &o);
                let quarter_kelly_res = variants::quarter_kelly(&p, &o);

                match (full_kelly_res, half_kelly_res, quarter_kelly_res) {
                    (Ok(fk), Ok(hk), Ok(qk)) => {
                        let mut rng = oxidize_core::rng::OxidizeRng::default();

                        let mut fk_bankroll = self.initial_bankroll;
                        let mut hk_bankroll = self.initial_bankroll;
                        let mut qk_bankroll = self.initial_bankroll;

                        self.full_kelly_points.push([0.0, fk_bankroll]);
                        self.half_kelly_points.push([0.0, hk_bankroll]);
                        self.quarter_kelly_points.push([0.0, qk_bankroll]);

                        for i in 1..=self.num_bets {
                            let win = rng.r#gen::<f64>() < p.value();

                            // Let's create a helper closure for bankroll update
                            let update_bankroll =
                                |bankroll: &mut f64, fraction: &BankrollFraction| {
                                    let bet_amount = *bankroll * fraction.value();
                                    if win {
                                        *bankroll += bet_amount * o.value();
                                    } else {
                                        *bankroll -= bet_amount;
                                    }
                                };

                            update_bankroll(&mut fk_bankroll, &fk);
                            update_bankroll(&mut hk_bankroll, &hk);
                            update_bankroll(&mut qk_bankroll, &qk);

                            self.full_kelly_points.push([i as f64, fk_bankroll]);
                            self.half_kelly_points.push([i as f64, hk_bankroll]);
                            self.quarter_kelly_points.push([i as f64, qk_bankroll]);
                        }
                    }
                    (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
                        self.error_msg = Some(format!("Error: {:?}", e)); // e is an enum
                    }
                }
            }
            (Err(e), _) => {
                self.error_msg = Some(format!("Error: {:?}", e));
            }
            (_, Err(e)) => {
                self.error_msg = Some(format!("Error: {:?}", e));
            }
        }
    }
}

impl InteractiveTool for BankrollGrowthTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Bankroll Growth"
    }

    fn export_data(&self) -> Option<crate::framework::ExportableData> {
        let datasets = vec![
            ("Full Kelly".to_string(), self.full_kelly_points.clone()),
            ("Half Kelly".to_string(), self.half_kelly_points.clone()),
            ("Quarter Kelly".to_string(), self.quarter_kelly_points.clone()),
        ];
        Some(crate::framework::ExportableData::series(
            "Bankroll Growth",
            vec!["Bet Number".into(), "Bankroll ($)".into()],
            datasets,
        ))
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            self.show_ui(ui);
        });
    }
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show_ui(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.heading("Parameters");
            let mut changed = false;

            changed |= ui
                .add(
                    egui::Slider::new(&mut self.initial_bankroll, 100.0..=10000.0)
                        .text("Initial Bankroll"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.probability, 0.01..=0.99)
                        .text("Win Probability (p)"),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.odds, 0.1..=10.0).text("Net Odds (b)"))
                .accessible_hover_text(
                    "Net profit multiplier. E.g., for +100 / even money, b = 1.0.",
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.num_bets, 10..=1000).text("Number of Bets"))
                .changed();

            if ui.button("Rerun Simulation").clicked() {
                changed = true;
            }

            if changed {
                self.recalculate();
            }

            ui.separator();

            if let Some(ref err) = self.error_msg {
                ui.colored_label(egui::Color32::RED, err);
            } else {
                ui.heading("Bankroll Simulation");

                let plot = Plot::new("bankroll_growth_plot")
                    .view_aspect(2.0)
                    .legend(egui_plot::Legend::default());

                plot.show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new(
                            "Full Kelly",
                            PlotPoints::new(self.full_kelly_points.clone()),
                        )
                        .color(egui::Color32::RED),
                    );
                    plot_ui.line(
                        Line::new(
                            "Half Kelly",
                            PlotPoints::new(self.half_kelly_points.clone()),
                        )
                        .color(egui::Color32::YELLOW),
                    );
                    plot_ui.line(
                        Line::new(
                            "Quarter Kelly",
                            PlotPoints::new(self.quarter_kelly_points.clone()),
                        )
                        .color(egui::Color32::GREEN),
                    );
                });
            }
        });
    }
}

// [cite:clinical_trials_statistics]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "BankrollGrowthTool",
        domain: "financial_math",
        tags: &[],
        build: || Box::new(BankrollGrowthTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for BankrollGrowthTool {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}
