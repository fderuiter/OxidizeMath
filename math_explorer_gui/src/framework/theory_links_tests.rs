#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::theory_links::{resolve_citation_reference_links, resolve_portal_paper_links};
use super::*;
use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::StateSnapshot;
use crate::framework::recovery::FiniteStateGuard;
use eframe::egui;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::{BTreeSet, HashMap};

struct DummyUnifiedEquationModel {
    val: f64,
}

impl TheoryDescribable for DummyUnifiedEquationModel {
    fn theory_description(&self) -> String {
        r"Diffusion rate $\frac{\partial u}{\partial t}$ in $\mathbb{R}^2$.".to_string()
    }
    fn theory_latex_equation(&self) -> Option<String> {
        Some(r"\frac{\partial u}{\partial t} = D \nabla^2 u".to_string())
    }
    fn phonetic_description(&self) -> String {
        "d u by d t equals D del squared u".to_string()
    }
    fn theory_citation(&self) -> String {
        "Attention Is All You Need (Vaswani et al., 2017)".to_string()
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

impl FiniteStateGuard for DummyUnifiedEquationModel {
    fn is_finite_state(&self) -> bool {
        self.val.is_finite()
    }
}

impl UnifiedModel for DummyUnifiedEquationModel {
    fn new(params: &HashMap<String, f64>) -> Self {
        Self {
            val: params.get("d").copied().unwrap_or(1.0),
        }
    }
    fn step(&mut self, _params: &HashMap<String, f64>) {}
    fn get_snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            width: 1,
            height: 1,
            pixels: std::sync::Arc::new(std::sync::RwLock::new(vec![eframe::egui::Color32::BLACK])),
            custom_data: Vec::new(),
            structured_data: None,
            sonification: crate::audio::SonificationFrame::default(),
        }
    }
    fn parameters() -> HashMap<String, ParameterConstraint> {
        let mut map = HashMap::new();
        map.insert(
            "d".to_string(),
            ParameterConstraint {
                min: 0.1,
                max: 2.0,
                step: 0.1,
            },
        );
        map
    }
    fn name() -> &'static str {
        "DummyUnifiedEquationModel"
    }
    fn domain() -> &'static str {
        "analysis"
    }
}

#[test]
fn test_unified_sim_tool_forwards_theory_latex_equation() {
    let tool = UnifiedSimTool::<DummyUnifiedEquationModel>::new();
    assert_eq!(
        tool.theory().theory_latex_equation().as_deref(),
        Some(r"\frac{\partial u}{\partial t} = D \nabla^2 u")
    );
}

#[test]
fn test_resolve_portal_paper_and_bibtex_links_filters_specs() {
    let spin_links = resolve_portal_paper_links("SpinVisualizer", "quantum");
    assert!(!spin_links.is_empty());
    assert_eq!(spin_links[0].paper_name, "quantum_mechanics.tex");
    assert!(spin_links[0]
        .pdf_url
        .as_deref()
        .is_some_and(|u| u.ends_with("/quantum_mechanics.pdf")));
    assert!(spin_links[0]
        .tex_url
        .as_deref()
        .is_some_and(|u| u.ends_with("/papers/quantum_mechanics.tex")));

    let ai_links = resolve_portal_paper_links("TrainingMonitorTool", "ai");
    assert!(!ai_links.is_empty());
    assert!(ai_links
        .iter()
        .all(|l| !l.paper_name.starts_with("spec:") && !l.paper_name.starts_with("registry:")));

    let fractran_links = resolve_portal_paper_links("FractranMachineTool", "number_theory");
    assert!(!fractran_links.is_empty());
    assert_eq!(
        fractran_links[0].paper_name,
        "algorithmic_information_rust.tex"
    );

    let glicko_links = resolve_portal_paper_links("Glicko2TournamentTool", "game_theory");
    assert!(!glicko_links.is_empty());
    assert_eq!(glicko_links[0].paper_name, "graph_parameters_rust.tex");

    let astro_links = resolve_portal_paper_links("GalacticOrbitSimulator", "astrophysics");
    assert!(!astro_links.is_empty());
    assert_eq!(
        astro_links[0].paper_name,
        "dwarf_galaxy_empirical_dependencies.tex"
    );

    let cera_ext = resolve_citation_reference_links(
        "CERA: A Framework for Improved Generalization of Machine Learning Models to Changed Climates (Liu et al., 2025)",
    );
    assert!(cera_ext
        .iter()
        .any(|e| e.url == "https://arxiv.org/abs/2509.00010"));

    let attn_ext =
        resolve_citation_reference_links("Attention Is All You Need (Vaswani et al., 2017)");
    assert!(attn_ext
        .iter()
        .any(|e| e.url == "https://arxiv.org/abs/1706.03762"));

    let part_ext = resolve_citation_reference_links(
        "On Eisenstein series, color partition and divisor function (Pushpa & Vasuki, 2022)",
    );
    assert!(part_ext
        .iter()
        .any(|e| e.url == "https://doi.org/10.1007/s40065-022-00360-6"));
}

fn assert_clean_portal_output(tool_id: &str, out: &egui::FullOutput) -> usize {
    const FORBIDDEN_RAW_TEX: &[&str] = &[
        r"\hbar", r"\langle", r"\rangle", r"\vec", r"\hat", r"\bar", r"\dot", r"\ddot", r"\tilde",
        r"\mathbb", r"\left", r"\right", r"\frac", r"\text", r"\mathrm", r"\mathbf", r"\AA",
        r"\deg", r"\dots", r"\pmod", "──", "ddot",
    ];
    let ak = out
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit tree must exist");
    let mut link_count = 0;
    for (_, node) in &ak.nodes {
        for text in node.label().into_iter().chain(node.value()) {
            for raw in FORBIDDEN_RAW_TEX {
                assert!(
                    !text.contains(raw),
                    "Tool `{tool_id}` rendered corrupted TeX/bar token `{raw}` in `{text}`"
                );
            }
            assert!(
                !text.contains("spec:") && !text.contains("registry:"),
                "Tool `{tool_id}` leaked spec:/registry: token in `{text}`"
            );
        }
        if node.label().is_some_and(|l| {
            l.starts_with("📄 View PDF")
                || l.starts_with("📝 View .tex Source")
                || l.starts_with("🔗 ")
        }) {
            link_count += 1;
        }
    }
    assert!(
        link_count >= 2,
        "Tool `{tool_id}` must expose at least PDF + .tex hyperlinks in portal, got {link_count}"
    );
    link_count
}

fn find_hyperlink_center(out: &egui::FullOutput, prefix: &str) -> Option<egui::Pos2> {
    let ak = out.platform_output.accesskit_update.as_ref()?;
    let bounds = ak.nodes.iter().find_map(|(_, n)| {
        n.label()
            .filter(|l| l.starts_with(prefix))
            .and_then(|_| n.bounds())
    })?;
    Some(egui::pos2(
        ((bounds.x0 + bounds.x1) * 0.5) as f32,
        ((bounds.y0 + bounds.y1) * 0.5) as f32,
    ))
}

fn run_portal_frame(
    ctx: &egui::Context,
    framework: &SimulationFramework,
    screen_rect: Option<egui::Rect>,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    ctx.run(
        egui::RawInput {
            screen_rect,
            events,
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                framework.show_theory_portal(ui, "portal_verify");
            });
        },
    )
}

fn assert_portal_hyperlink_opens_url(
    framework: &SimulationFramework,
    screen_rect: Option<egui::Rect>,
    prefix: &str,
    expected_url_substr: &str,
) {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let initial = run_portal_frame(&ctx, framework, screen_rect, Vec::new());
    let pos = find_hyperlink_center(&initial, prefix)
        .unwrap_or_else(|| panic!("Expected portal hyperlink starting with `{prefix}`"));
    let press_events = vec![
        egui::Event::PointerMoved(pos),
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = run_portal_frame(&ctx, framework, screen_rect, press_events);
    let release_events = vec![egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed: false,
        modifiers: egui::Modifiers::NONE,
    }];
    let release = run_portal_frame(&ctx, framework, screen_rect, release_events);
    let opened = release
        .platform_output
        .commands
        .iter()
        .any(|cmd| match cmd {
            egui::OutputCommand::OpenUrl(open) => open.url.contains(expected_url_substr),
            _ => false,
        });
    assert!(
        opened,
        "Clicking `{prefix}` must emit OpenUrl containing `{expected_url_substr}`, got {:?}",
        release.platform_output.commands
    );
}

#[test]
fn test_all_registered_tools_render_clean_portal_latex_and_hyperlinks() {
    let domains: BTreeSet<&'static str> = inventory::iter::<ToolMetadata>
        .into_iter()
        .map(|m| m.domain)
        .collect();
    let screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(1440.0, 900.0),
    ));
    let mut verified_tools = 0_usize;

    for domain in domains {
        let mut framework = SimulationFramework::new(domain);
        framework.show_theory_portal = true;
        for idx in 0..framework.available_tools.len() {
            framework.select_tool(idx);
            framework.show_theory_portal = true;
            let tool_id = framework.available_tools[idx].id;
            let ctx = egui::Context::default();
            ctx.enable_accesskit();
            let out = run_portal_frame(&ctx, &framework, screen_rect, Vec::new());
            let _ = assert_clean_portal_output(tool_id, &out);
            if tool_id == "SpinVisualizer" {
                assert_portal_hyperlink_opens_url(
                    &framework,
                    screen_rect,
                    "📄 View PDF",
                    "quantum_mechanics.pdf",
                );
            } else if tool_id == "CeraTool" {
                assert_portal_hyperlink_opens_url(
                    &framework,
                    screen_rect,
                    "🔗 arXiv:2509.00010",
                    "https://arxiv.org/abs/2509.00010",
                );
            }
            verified_tools += 1;
        }
    }

    #[cfg(all(feature = "ai", feature = "physics", feature = "pure_math"))]
    assert!(
        verified_tools >= 60,
        "Expected >= 60 tools, got {verified_tools}"
    );
    #[cfg(not(all(feature = "ai", feature = "physics", feature = "pure_math")))]
    assert!(verified_tools > 0);
}
