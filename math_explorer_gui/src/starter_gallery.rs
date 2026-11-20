//! Starter Template Gallery component providing guided empty-state cards
//! for simulation tabs and instant preset model execution.

use crate::framework::ToolMetadata;
use eframe::egui;

/// Pre-configured starter simulation template card.
#[derive(Clone, Debug, PartialEq)]
pub struct StarterCard {
    /// Unique identifier for the starter card.
    pub id: &'static str,
    /// User-facing display title.
    pub title: &'static str,
    /// Concise mathematical description or governing equation.
    pub description: &'static str,
    /// Category or domain tag.
    pub category: &'static str,
    /// Visual icon or emoji.
    pub icon: &'static str,
    /// Target domain string matching [`crate::framework::InteractiveTool::domain`].
    pub domain: &'static str,
    /// Tool metadata name in the domain to select/activate.
    pub tool_name: Option<&'static str>,
    /// Serialized JSON `PresetWrapper` configuration to load.
    pub preset_json: Option<&'static str>,
}

/// Federated starter preset registration metadata submitted across tools.
#[derive(Clone, Copy, Debug)]
pub struct StarterPresetMetadata {
    /// Unique identifier for the preset template.
    pub id: &'static str,
    /// User-facing display title.
    pub title: &'static str,
    /// Concise mathematical description or governing equation.
    pub description: &'static str,
    /// Category or domain tag.
    pub category: &'static str,
    /// Visual icon or emoji.
    pub icon: &'static str,
    /// Target domain string matching the simulation tool domain.
    pub domain: &'static str,
    /// Target tool metadata name (if associated with a registered ToolMetadata).
    pub tool_name: Option<&'static str>,
    /// Function pointer returning the serialized JSON preset configuration envelope.
    pub preset_generator: fn() -> &'static str,
}

impl PartialEq for StarterPresetMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.title == other.title && self.domain == other.domain
    }
}

impl StarterPresetMetadata {
    /// Generates the JSON preset payload.
    #[inline]
    pub fn generate_preset(&self) -> &'static str {
        (self.preset_generator)()
    }
}

inventory::collect!(StarterPresetMetadata);

/// JSON preset envelope for Harmonic Oscillator ODE.
pub const PRESET_HARMONIC_OSCILLATOR: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "ODE Solver - y'' = -k*y (Harmonic Oscillator)",
    "domain": "analysis",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "preset": "HarmonicOscillator",
    "dt": 0.05,
    "total_time": 20.0,
    "param_k": 2.0,
    "param_r": 1.0,
    "param_cap_k": 10.0,
    "ic_y0": 1.5,
    "ic_v0": 0.0
  }
}"#;

/// JSON preset envelope for Logistic Growth ODE.
pub const PRESET_LOGISTIC_GROWTH: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "ODE Solver - y' = r*y*(1 - y/K) (Logistic Growth)",
    "domain": "analysis",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "preset": "LogisticGrowth",
    "dt": 0.05,
    "total_time": 20.0,
    "param_k": 1.0,
    "param_r": 0.8,
    "param_cap_k": 10.0,
    "ic_y0": 0.5,
    "ic_v0": 0.0
  }
}"#;

/// JSON preset envelope for Exponential Growth/Decay ODE.
pub const PRESET_EXPONENTIAL_DECAY: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "ODE Solver - y' = k*y (Exponential)",
    "domain": "analysis",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "preset": "Exponential",
    "dt": 0.05,
    "total_time": 10.0,
    "param_k": -0.5,
    "param_r": 1.0,
    "param_cap_k": 10.0,
    "ic_y0": 5.0,
    "ic_v0": 0.0
  }
}"#;

/// JSON preset envelope for Turing Spots reaction-diffusion.
pub const PRESET_TURING_SPOTS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Turing Morphogenesis Simulation - Spots",
    "domain": "biology",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "a": 0.1,
    "b": 0.9,
    "d_u": 1.0,
    "d_v": 100.0,
    "dt": 0.05,
    "width": 100,
    "height": 100
  }
}"#;

/// JSON preset envelope for Turing Stripes reaction-diffusion.
pub const PRESET_TURING_STRIPES: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Turing Morphogenesis Simulation - Stripes",
    "domain": "biology",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "a": 0.14,
    "b": 0.86,
    "d_u": 1.0,
    "d_v": 50.0,
    "dt": 0.02,
    "width": 100,
    "height": 100
  }
}"#;

/// JSON preset envelope for Fluid Dynamics Cylinder Vector Field.
pub const PRESET_VECTOR_FIELD: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Potential Flow - Cylinder Vector Field",
    "domain": "fluid_dynamics",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "elements": [
      {
        "element_type": "Uniform",
        "strength": 10.0,
        "x": 0.0,
        "y": 0.0,
        "angle": 0.0
      },
      {
        "element_type": "Doublet",
        "strength": 251.32741228718345,
        "x": 0.0,
        "y": 0.0,
        "angle": 0.0
      }
    ],
    "grid_size": 20,
    "view_range": 10.0
  }
}"#;

/// JSON preset envelope for Fluid Dynamics Source-Sink Pair.
pub const PRESET_SOURCE_SINK: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Potential Flow - Source-Sink Singularity Pair",
    "domain": "fluid_dynamics",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "elements": [
      {
        "element_type": "Source",
        "strength": 10.0,
        "x": -3.0,
        "y": 0.0,
        "angle": 0.0
      },
      {
        "element_type": "Sink",
        "strength": 10.0,
        "x": 3.0,
        "y": 0.0,
        "angle": 0.0
      }
    ],
    "grid_size": 25,
    "view_range": 12.0
  }
}"#;

/// JSON preset envelope for 3D Chaotic Attractor.
pub const PRESET_3D_ATTRACTOR: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Lorenz 3D Attractor - Butterfly",
    "domain": "chaos",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "sigma": 10.0,
    "rho": 28.0,
    "beta": 2.6666666666666665,
    "dt": 0.01,
    "simulation_speed": 5,
    "execution_state": "Running",
    "show_annotations": true,
    "camera": {
      "pitch": 0.35,
      "yaw": -0.45,
      "zoom": 1.0
    }
  }
}"#;

/// JSON preset envelope for Replicator Hawk-Dove Dynamics.
pub const PRESET_HAWK_DOVE: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Replicator Dynamics - Hawk-Dove",
    "domain": "game_theory",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "payoff_matrix": [[-1.0, 2.0], [0.0, 1.0]],
    "initial_population": [0.5, 0.5],
    "time_horizon": 20.0,
    "dt": 0.05,
    "strategy_names": ["Hawk", "Dove"]
  }
}"#;

/// Returns curated or dynamically generated starter cards for a simulation domain.
pub fn get_starter_cards_for_domain(
    domain: &str,
    available_tools: &[&'static ToolMetadata],
) -> Vec<StarterCard> {
    let mut cards: Vec<StarterCard> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .filter(|p| {
            p.domain == domain
                || (domain == "biology" && p.domain == "morphogenesis")
                || (domain == "morphogenesis" && p.domain == "biology")
        })
        .map(|p| StarterCard {
            id: p.id,
            title: p.title,
            description: p.description,
            category: p.category,
            icon: p.icon,
            domain: p.domain,
            tool_name: p.tool_name,
            preset_json: Some((p.preset_generator)()),
        })
        .collect();

    // Dynamic fallback cards based on available tools in domain if no presets submitted
    if cards.is_empty() {
        for meta in available_tools {
            if meta.domain == domain {
                cards.push(StarterCard {
                    id: meta.id,
                    title: meta.display_name,
                    description:
                        "Interactive simulation tool for mathematical exploration and analysis.",
                    category: meta.domain,
                    icon: "🔬",
                    domain: meta.domain,
                    tool_name: Some(meta.id),
                    preset_json: None,
                });
            }
        }
    }

    cards
}

fn render_starter_card(ui: &mut egui::Ui, card: &StarterCard, card_width: f32) -> bool {
    let mut launched = false;
    egui::Frame::group(ui.style())
        .inner_margin(14.0)
        .corner_radius(8.0)
        .show(ui, |ui| {
            ui.set_width(card_width);
            ui.set_min_height(150.0);
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(card.icon).size(26.0));
                    ui.vertical(|ui| {
                        ui.label(egui::RichText::new(card.title).strong().size(15.0));
                        ui.label(
                            egui::RichText::new(format!("[{}]", card.category))
                                .small()
                                .color(ui.visuals().weak_text_color()),
                        );
                    });
                });
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new(card.description)
                        .size(12.0)
                        .color(ui.visuals().text_color()),
                );
                ui.add_space(10.0);
                if ui.button(format!("▶ Launch {}", card.title)).clicked() {
                    launched = true;
                }
            });
        });
    launched
}

struct GalleryOverlayMetrics {
    max_w: f32,
    max_h: f32,
    inner_w: f32,
    scroll_h: f32,
    card_w: f32,
}

fn compute_gallery_metrics(ctx: &egui::Context) -> GalleryOverlayMetrics {
    let raw_screen = ctx.content_rect();
    let avail_w = if raw_screen.width().is_finite() && raw_screen.width() > 64.0 {
        raw_screen.width()
    } else {
        1024.0
    };
    let avail_h = if raw_screen.height().is_finite() && raw_screen.height() > 64.0 {
        raw_screen.height()
    } else {
        768.0
    };
    let max_w = (avail_w - 32.0).clamp(180.0, 740.0);
    let max_h = (avail_h - 32.0).clamp(160.0, 520.0);
    let inner_w = (max_w - 36.0).max(144.0);
    let scroll_h = (max_h - 124.0).max(64.0);
    let card_w = if inner_w < 460.0 {
        (inner_w - 16.0).clamp(128.0, 215.0)
    } else {
        215.0
    };
    GalleryOverlayMetrics {
        max_w,
        max_h,
        inner_w,
        scroll_h,
        card_w,
    }
}

/// Renders the central canvas starter gallery card overlay clamped to root window bounds.
pub fn show_starter_gallery_overlay(
    ctx: &egui::Context,
    id_source: &str,
    cards: &[StarterCard],
    is_open: &mut bool,
) -> Option<StarterCard> {
    if cards.is_empty() {
        return None;
    }
    let m = compute_gallery_metrics(ctx);
    let mut selected_card = None;
    let modal_id = egui::Id::new(format!("{id_source}_starter_gallery_overlay"));
    let frame = egui::Frame::window(&ctx.style())
        .inner_margin(16)
        .corner_radius(12.0)
        .shadow(ctx.style().visuals.window_shadow);

    crate::ModalLifecycle::new(modal_id)
        .frame(frame)
        .backdrop_color(egui::Color32::from_black_alpha(120))
        .max_width(m.max_w)
        .max_height(m.max_h)
        .show(ctx, is_open, |ui, frame_ctx| {
            ui.set_max_width(m.inner_w);
            ui.set_width(m.inner_w);
            ui.horizontal(|ui| {
                ui.set_width(m.inner_w);
                ui.add(
                    egui::Label::new(
                        egui::RichText::new("🚀 Starter Template Gallery").heading(),
                    )
                    .truncate(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("✖ Close Gallery").clicked() {
                        frame_ctx.close();
                    }
                });
            });

            ui.add(
                egui::Label::new(
                    egui::RichText::new(
                        "Choose a guided starter model to launch pre-configured simulation parameters:",
                    )
                    .color(ui.visuals().weak_text_color()),
                )
                .wrap(),
            );
            ui.add_space(6.0);
            ui.separator();
            ui.add_space(8.0);

            egui::ScrollArea::vertical()
                .max_height(m.scroll_h)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
                        for card in cards {
                            if render_starter_card(ui, card, m.card_w) {
                                selected_card = Some(card.clone());
                                frame_ctx.close();
                            }
                        }
                    });
                });
        });

    selected_card
}

#[path = "starter_gallery/root_host.rs"]
mod root_host;
pub(crate) use root_host::{
    begin_root_starter_gallery_scope, flush_deferred_starter_galleries, try_defer_starter_gallery,
};

#[cfg(test)]
#[path = "starter_gallery/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "starter_gallery/contract_tests.rs"]
mod contract_tests;
