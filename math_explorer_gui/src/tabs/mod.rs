#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;

#[cfg(feature = "ai")]
pub mod ai;
#[cfg(feature = "pure_math")]
pub mod algorithmic_information;
#[cfg(feature = "pure_math")]
pub mod analysis;
#[cfg(feature = "physics")]
pub mod astrophysics;
#[cfg(feature = "applied")]
pub mod battery_degradation;
#[cfg(feature = "applied")]
pub mod cannibalism;
#[cfg(feature = "physics")]
pub mod chaos;
#[cfg(feature = "climate")]
pub mod climate;
#[cfg(feature = "applied")]
pub mod clinical_trials;
#[cfg(feature = "pure_math")]
pub mod elliptic_curves;
#[cfg(feature = "epidemiology")]
pub mod epidemiology;
#[cfg(feature = "applied")]
pub mod favoritism;
#[cfg(feature = "pure_math")]
pub mod financial_math;
#[cfg(feature = "physics")]
pub mod fluid_dynamics;
#[cfg(feature = "applied")]
pub mod game_theory;
#[cfg(feature = "pure_math")]
pub mod geometry_topology;
#[cfg(feature = "pure_math")]
pub mod graph_theory;
#[cfg(feature = "physics")]
pub mod medical;
#[cfg(feature = "biology")]
pub mod morphogenesis;
#[cfg(feature = "physics")]
pub mod mri;
#[cfg(feature = "biology")]
pub mod neuroscience;
#[cfg(feature = "pure_math")]
pub mod number_theory;
#[cfg(feature = "physics")]
pub mod optics;
#[cfg(feature = "physics")]
pub mod optics_waves;
#[cfg(feature = "applied")]
pub mod paradox_lab;
#[cfg(feature = "physics")]
pub mod quantum;
#[cfg(feature = "physics")]
pub mod relativity;
#[cfg(feature = "physics")]
pub mod solid_state;
pub mod traceability;

/// Static registration descriptor for a GUI domain tab collected via [`inventory`].
#[derive(Clone, Copy)]
pub struct TabDescriptor {
    /// Canonical module and URL slug used as a deterministic secondary sort key.
    pub id: &'static str,
    /// Primary navigation bar sort priority (lower numbers appear first; default `100`).
    pub order: i32,
    /// Constructor returning a boxed default instance of the tab.
    pub build: fn() -> Box<dyn ExplorerTab>,
}

inventory::collect!(TabDescriptor);

fn sorted_tab_descriptors() -> Vec<&'static TabDescriptor> {
    let mut descriptors: Vec<&'static TabDescriptor> =
        inventory::iter::<TabDescriptor>.into_iter().collect();
    descriptors.sort_by(|a, b| a.order.cmp(&b.order).then_with(|| a.id.cmp(b.id)));
    descriptors
}

/// Instantiates all enabled GUI tabs in deterministic `(order, id)` navigation order.
pub fn instantiate_tabs() -> Vec<Box<dyn ExplorerTab>> {
    sorted_tab_descriptors()
        .into_iter()
        .map(|d| (d.build)())
        .collect()
}

/// Instantiates a single enabled GUI tab by its deterministic `(order, id)` index.
pub fn instantiate_tab(index: usize) -> Option<Box<dyn ExplorerTab>> {
    sorted_tab_descriptors().get(index).map(|d| (d.build)())
}

/// A trait for defining a tab in the Math Explorer application.
///
/// This trait serves as the primary extension point for the GUI. Each mathematical domain
/// (e.g., MRI Physics, Game Theory) implements this trait as a standalone struct.
///
/// # Static Inventory Registration
///
/// To add a new tab to `math_explorer_gui`:
/// 1. Declare a feature-gated `pub mod your_tab;` item in `src/tabs/mod.rs`.
/// 2. Implement [`ExplorerTab`] and [`Default`] on your tab struct.
/// 3. Register a [`TabDescriptor`] via `inventory::submit!`.
///
/// # Example
///
/// ```rust,ignore
/// use crate::framework::SimulationFramework;
/// use crate::tabs::{ExplorerTab, TabDescriptor};
/// use eframe::egui;
///
/// pub struct MyDomainTab {
///     framework: SimulationFramework,
/// }
///
/// impl Default for MyDomainTab {
///     fn default() -> Self {
///         Self {
///             framework: SimulationFramework::new("physics"),
///         }
///     }
/// }
///
/// impl ExplorerTab for MyDomainTab {
///     fn name(&self) -> &'static str {
///         "Physics"
///     }
///
///     fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
///         self.framework.show_inside(ui, "physics");
///     }
///
///     fn framework(&self) -> Option<&SimulationFramework> {
///         Some(&self.framework)
///     }
///
///     fn framework_mut(&mut self) -> Option<&mut SimulationFramework> {
///         Some(&mut self.framework)
///     }
/// }
///
/// inventory::submit! {
///     TabDescriptor {
///         id: "my_domain",
///         order: 100,
///         build: || Box::new(MyDomainTab::default()),
///     }
/// }
/// ```
pub trait ExplorerTab {
    /// Returns the name of the tab, displayed in the navigation bar.
    fn name(&self) -> &'static str;

    /// Renders the content of the tab inside a scoped parent `egui::Ui` (such as a multi-pane viewport region).
    fn show_inside(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame);

    /// Monotonic or deterministic revision token for persisted tab state (`0` if untracked).
    fn state_revision(&self) -> u64 {
        0
    }

    /// Deterministic revision token excluding transient framework chrome (`show_theory_portal`, `show_starter_gallery`).
    fn history_revision(&self) -> u64 {
        self.framework()
            .map_or_else(|| self.state_revision(), |fw| fw.history_revision())
    }

    /// Serializes the tab's parameter and panel state into a String.
    fn save_state(&self) -> Option<String> {
        None
    }

    /// Deserializes and restores the tab's parameter and panel state from a String.
    fn load_state(&mut self, _state: &str) {}

    /// Returns the active tool's interactive canvas `Id` when focus is owned by a non-text canvas.
    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.framework().and_then(|f| f.focused_canvas_id())
    }

    /// Academic domain category of the tab for top navigation bar grouping.
    fn academic_domain(&self) -> &'static str {
        match self.name() {
            "Algorithmic Information"
            | "Analysis & Calculus"
            | "Elliptic Curves"
            | "Financial Math"
            | "Geometry & Topology"
            | "Graph Theory"
            | "Number Theory" => "Pure Math",

            "Astrophysics"
            | "Chaos Theory"
            | "Fluid Dynamics"
            | "General Relativity"
            | "Medical Physics"
            | "MRI Physics"
            | "Optics & Caustics"
            | "Optics & Waves"
            | "Quantum Mechanics"
            | "Solid State Physics" => "Physics",

            "Battery Degradation"
            | "Clinical Trials"
            | "Favoritism"
            | "Game Theory"
            | "Paradox Lab"
            | "Population Structure"
            | "Traceability Portal" => "Applied Math",

            "Morphogenesis" | "Neuroscience" | "Epidemiology" => "Biology",

            "Climate" => "Climate",

            "Artificial Intelligence" => "AI",

            _ => "Applied Math",
        }
    }

    /// Activates a tool within this tab by name.
    /// Returns true if a matching tool was found and activated.
    fn select_tool_by_name(&mut self, _name: &str) -> bool {
        false
    }

    /// Returns a reference to the underlying [`crate::framework::SimulationFramework`] when backed by one.
    fn framework(&self) -> Option<&crate::framework::SimulationFramework> {
        None
    }

    /// Returns a mutable reference to the underlying [`crate::framework::SimulationFramework`] when backed by one.
    fn framework_mut(&mut self) -> Option<&mut crate::framework::SimulationFramework> {
        None
    }

    /// Applies the root-hoisted `StarterGallery` modal outcome to this tab instance.
    fn apply_starter_gallery_outcome(
        &mut self,
        ctx: &egui::Context,
        id_source: &str,
        is_open: bool,
        selected: Option<crate::starter_gallery::StarterCard>,
    ) {
        if let Some(fw) = self.framework_mut() {
            fw.apply_starter_gallery_outcome(ctx, id_source, is_open, selected);
        }
    }

    /// Applies either a starter card or inverse challenge selection from `StarterGallery`.
    fn apply_starter_gallery_selection(
        &mut self,
        ctx: &egui::Context,
        id_source: &str,
        is_open: bool,
        selected: Option<crate::starter_gallery::GallerySelection>,
    ) {
        match selected {
            Some(crate::starter_gallery::GallerySelection::Starter(card)) => {
                self.apply_starter_gallery_outcome(ctx, id_source, is_open, Some(card));
            }
            other => {
                if let Some(fw) = self.framework_mut() {
                    fw.apply_starter_gallery_selection(ctx, id_source, is_open, other);
                }
            }
        }
    }

    /// Canonical URL slug identifying this tab in `#tab=<slug>` share links.
    fn url_slug(&self) -> &'static str {
        match self.name() {
            "Artificial Intelligence" => "ai",
            "Algorithmic Information" => "algorithmic_information",
            "Analysis & Calculus" => "analysis",
            "Astrophysics" => "astrophysics",
            "Battery Degradation" => "battery_degradation",
            "Chaos Theory" => "chaos",
            "Climate" => "climate",
            "Clinical Trials" => "clinical_trials",
            "Elliptic Curves" => "elliptic_curves",
            "Epidemiology" => "epidemiology",
            "Favoritism" => "favoritism",
            "Financial Math" => "financial_math",
            "Fluid Dynamics" => "fluid_dynamics",
            "Game Theory" => "game_theory",
            "General Relativity" => "relativity",
            "Geometry & Topology" => "geometry_topology",
            "Graph Theory" => "graph_theory",
            "Medical Physics" => "medical",
            "Morphogenesis" => "morphogenesis",
            "MRI Physics" => "mri",
            "Neuroscience" => "neuroscience",
            "Number Theory" => "number_theory",
            "Optics & Caustics" => "optics",
            "Optics & Waves" => "optics_waves",
            "Paradox Lab" => "paradox_lab",
            "Population Structure" => "cannibalism",
            "Quantum Mechanics" => "quantum",
            "Solid State Physics" => "solid_state",
            "Traceability Portal" => "traceability",
            _ => "simulation",
        }
    }

    /// Active `InteractiveTool` identifier for `#tool=<id>` share links (`None` for single-view tabs).
    fn active_tool_id(&self) -> Option<&'static str> {
        self.framework().and_then(|fw| fw.active_tool_id())
    }

    /// Active inverse `ChallengeMetadata` identifier for `#challenge=<id>` share links.
    fn active_challenge_id(&self) -> Option<&'static str> {
        self.framework().and_then(|fw| fw.active_challenge_id())
    }

    /// Deterministic parameter-only revision token for 300 ms debounced URL sync.
    fn url_state_revision(&self) -> u64 {
        self.framework()
            .map_or_else(|| self.state_revision(), |fw| fw.url_state_revision())
    }

    /// Exports active simulation parameters as a compact `PresetWrapper` JSON envelope for URL sharing (`<= 4 KiB`).
    fn export_url_preset_json(&self) -> Result<Option<String>, String> {
        match self.framework() {
            Some(fw) => fw.export_url_preset_json(),
            None => Ok(None),
        }
    }

    /// Exports active simulation parameters as a compact `PresetWrapper` JSON envelope for `#ws=...` workspace URL sharing (`<= 32 KiB`).
    fn export_workspace_url_preset_json(&self) -> Result<Option<String>, String> {
        match self.framework() {
            Some(fw) => fw.export_workspace_url_preset_json(),
            None => self.export_url_preset_json(),
        }
    }

    /// Exports structured `ShareUrlParams` (including active story/challenge HUD state and up to 32 KiB per pane) for workspace URL sharing.
    fn export_share_url_params(&self) -> Result<crate::url_state::ShareUrlParams, String> {
        if let Some(fw) = self.framework() {
            return fw.export_workspace_share_url_params(self.url_slug());
        }
        Ok(crate::url_state::ShareUrlParams {
            tab: self.url_slug().to_string(),
            tool: self.active_tool_id().map(str::to_string),
            state_json: self.export_workspace_url_preset_json()?,
            preset: None,
            challenge: self.active_challenge_id().map(str::to_string),
            step: None,
            pipeline: None,
        })
    }

    /// Validates and applies `ShareUrlParams` (supporting `&preset=`, `&challenge=`, `&step=`, and `&state=`).
    fn import_url_share_params(
        &mut self,
        params: &crate::url_state::ShareUrlParams,
        ctx: Option<&egui::Context>,
    ) -> Result<(), String> {
        if params.preset.is_none() && params.challenge.is_none() {
            return self
                .import_url_preset_json(params.tool.as_deref(), params.state_json.as_deref());
        }
        if let Some(fw) = self.framework_mut() {
            return fw.import_url_share_params(params, ctx);
        }
        Err(format!(
            "Tab '{}' does not support story presets or challenges",
            self.name()
        ))
    }

    /// Validates and applies a shared URL tool identifier and `PresetWrapper` JSON payload.
    fn import_url_preset_json(
        &mut self,
        tool_id: Option<&str>,
        preset_json: Option<&str>,
    ) -> Result<(), String> {
        if let Some(fw) = self.framework_mut() {
            return fw.import_url_preset_json(tool_id, preset_json);
        }
        if let Some(unexpected_tool) = tool_id {
            return Err(format!(
                "Tab '{}' does not support sub-tool '{unexpected_tool}'",
                self.name()
            ));
        }
        if preset_json.is_some() {
            return Err(format!(
                "URL parameter state import is not supported for '{}'",
                self.name()
            ));
        }
        Ok(())
    }

    /// Validates and applies a `#ws=...` workspace URL pane payload (allowing up to 32,768 bytes).
    fn import_workspace_url_preset_json(
        &mut self,
        tool_id: Option<&str>,
        preset_json: Option<&str>,
    ) -> Result<(), String> {
        if let Some(fw) = self.framework_mut() {
            return fw.import_workspace_url_preset_json(tool_id, preset_json);
        }
        self.import_url_preset_json(tool_id, preset_json)
    }

    /// Validates and applies `#ws=...` workspace `ShareUrlParams` (supporting stories, challenges, and up to 32,768 bytes).
    fn import_workspace_url_share_params(
        &mut self,
        params: &crate::url_state::ShareUrlParams,
        ctx: Option<&egui::Context>,
    ) -> Result<(), String> {
        if params.preset.is_none() && params.challenge.is_none() {
            return self.import_workspace_url_preset_json(
                params.tool.as_deref(),
                params.state_json.as_deref(),
            );
        }
        if let Some(fw) = self.framework_mut() {
            return fw.import_workspace_url_share_params(params, ctx);
        }
        Err(format!(
            "Tab '{}' does not support story presets or challenges",
            self.name()
        ))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
