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
                    egui::Stroke::new(1.0_f32, egui::Color32::WHITE);
                visuals.widgets.inactive.fg_stroke =
                    egui::Stroke::new(1.0_f32, egui::Color32::WHITE);
                visuals.widgets.hovered.fg_stroke =
                    egui::Stroke::new(2.0_f32, egui::Color32::YELLOW);
                visuals.widgets.active.fg_stroke = egui::Stroke::new(2.0_f32, egui::Color32::WHITE);
                visuals.selection.bg_fill = egui::Color32::from_rgb(0, 90, 158);
                visuals.selection.stroke = egui::Stroke::new(1.0_f32, egui::Color32::BLACK);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn channel_to_linear(channel: u8) -> f64 {
        let c = channel as f64 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    fn relative_luminance(color: egui::Color32) -> f64 {
        0.2126 * channel_to_linear(color.r())
            + 0.7152 * channel_to_linear(color.g())
            + 0.0722 * channel_to_linear(color.b())
    }

    fn contrast_ratio(c1: egui::Color32, c2: egui::Color32) -> f64 {
        let l1 = relative_luminance(c1);
        let l2 = relative_luminance(c2);
        let lighter = l1.max(l2);
        let darker = l1.min(l2);
        (lighter + 0.05) / (darker + 0.05)
    }

    #[test]
    fn test_high_contrast_selection_contrast_ratio() {
        let visuals = ThemeMode::HighContrast.visuals();
        let selection_bg = visuals.selection.bg_fill;
        assert_eq!(selection_bg, egui::Color32::from_rgb(0, 90, 158));

        // WCAG 2.2 AA requires >= 4.5:1 for normal text and strokes against background
        let active_fg = visuals.widgets.active.fg_stroke.color;
        let active_contrast = contrast_ratio(selection_bg, active_fg);
        assert!(
            active_contrast >= 4.5,
            "Selection fill contrast against active fg ({active_contrast:.2}:1) must exceed 4.5:1"
        );

        let noninteractive_fg = visuals.widgets.noninteractive.fg_stroke.color;
        let noninteractive_contrast = contrast_ratio(selection_bg, noninteractive_fg);
        assert!(
            noninteractive_contrast >= 4.5,
            "Selection fill contrast against noninteractive fg ({noninteractive_contrast:.2}:1) must exceed 4.5:1"
        );

        let inactive_fg = visuals.widgets.inactive.fg_stroke.color;
        let inactive_contrast = contrast_ratio(selection_bg, inactive_fg);
        assert!(
            inactive_contrast >= 4.5,
            "Selection fill contrast against inactive fg ({inactive_contrast:.2}:1) must exceed 4.5:1"
        );
    }

    #[test]
    fn test_legacy_yellow_selection_fails_contrast_ratio() {
        let yellow = egui::Color32::YELLOW;
        let white = egui::Color32::WHITE;
        let ratio = contrast_ratio(yellow, white);
        // Demonstrates why legacy yellow fill violated WCAG contrast
        assert!(
            ratio < 4.5,
            "Legacy yellow on white ({ratio:.2}:1) should fail WCAG 4.5:1 requirement"
        );
    }
}
