use eframe::egui;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum ThemeMode {
    #[default]
    Dark,
    Light,
    HighContrast,
}

impl ThemeMode {
    pub fn name(self) -> &'static str {
        match self {
            ThemeMode::Dark => "Dark",
            ThemeMode::Light => "Light",
            ThemeMode::HighContrast => "High Contrast",
        }
    }

    pub fn visuals(self) -> egui::Visuals {
        match self {
            ThemeMode::Dark => egui::Visuals::dark(),
            ThemeMode::Light => egui::Visuals::light(),
            ThemeMode::HighContrast => {
                let mut visuals = egui::Visuals::dark();
                visuals.extreme_bg_color = egui::Color32::BLACK;
                visuals.panel_fill = egui::Color32::BLACK;
                visuals.window_fill = egui::Color32::BLACK;
                visuals.widgets.noninteractive.bg_fill = egui::Color32::BLACK;
                visuals.widgets.noninteractive.fg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::WHITE);
                visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
                visuals.widgets.hovered.fg_stroke = egui::Stroke::new(2.0, egui::Color32::YELLOW);
                visuals.widgets.active.fg_stroke = egui::Stroke::new(2.0, egui::Color32::WHITE);
                visuals.selection.bg_fill = egui::Color32::YELLOW;
                visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::BLACK);
                visuals
            }
        }
    }
}

impl std::fmt::Display for ThemeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
