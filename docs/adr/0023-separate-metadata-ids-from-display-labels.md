# ADR 0023: Separate Metadata IDs, AST Spans, and Traceability Tags from Display Labels

## Status
Accepted

## Context
Across `math_explorer_gui` and core domain/math crates, internal machine identifiers were coupled directly to user-facing UI controls and screen-reader (`aria-live`) announcements:
- `ToolMetadata` exposed a single `name` field holding raw `CamelCase` struct identifiers for 51 of 52 tools, which leaked into sidebar selectors, launchpad cards, command palette rows, and `Switched to tool ...` accessibility announcements, alongside `snake_case` domain keys.
- `ParseError` and `LexError` in `crates/math_commons/src/ast/` formatted internal `Debug` `Span { start, end }` structs, 0-indexed byte ranges, and `<EOF>` sentinel tokens directly into live validation alerts in `MathExpressionInputBar`.
- `TheoryDescribable::theory_citation()` and `theory_description()` in `crates/domain_physics` (`LorenzSystem`, `QuantumState`) and four GUI tools returned raw `[cite:...]` traceability tags intended for `traceability_cli` source comments per ADR 0011, while `SimulationFramework::show_theory_portal` rendered raw `"default"` map keys.
- Error handlers and widgets formatted enums (`TdaError`, `KellyError`, `CrystalSystemEnum`, `SessionLifecycleState`, `egui::Modifiers`) with `{:?}`, and `UnifiedSimTool` bound raw `snake_case` parameter keys directly to sliders and observable sensitivity cards.

## Decision
1. Split `ToolMetadata` into a stable `id` key (used for `StarterPresetMetadata` matching and workspace state persistence) and a mandatory `display_name` label matching `InteractiveTool::name()`, while accepting both `id` and `display_name` in `load_state` and `select_tool_by_name` for backward session compatibility.
2. Added domain title formatting and dual key/title search indexing in `CommandPalette`, and ensured tool-switch `aria-live` announcements emit plain `display_name` titles.
3. Added `LexError::user_message`, `ParseError::user_message`, and `ParseError::user_message_for_input` in `crates/math_commons/src/ast/` reporting 1-indexed character positions and `"end of expression"` with zero `Span` `Debug` dumps or `<EOF>` tokens, wired into `MathExpressionInputBar`.
4. Replaced `{:?}` formatting across UI panels (paired with `CommandTrigger::display_label()` in the unified shortcut modal), combo boxes, session notifications, and `TdaError::fmt` (`crates/pure_math/src/error.rs`) with plain-language descriptions.
5. Preserved `// [cite:...]` source comments for `traceability_cli` while replacing `[cite:...]` runtime strings in `LorenzSystem`, `QuantumState`, and GUI tools with bibliographic citations, and omitted `"default"` keys in the Theory Context Portal.
6. Extended `UnifiedModel` with `parameter_label` and `parameter_choices` so `UnifiedSimTool` renders descriptive parameter labels, discrete combo-box selections (`Potential Profile`), and relabeled observable `SensitivityEngine` cards while continuing to pass stable parameter IDs to `evaluate_observable`.

## Consequences
- Visual selectors, tooltips, sliders, error banners, theory portals, and `aria-live` screen-reader announcements present respectful plain-language mathematical titles and citations.
- Stable `ToolMetadata.id` identifiers preserve preset linkage and backward-compatible workspace session deserialization.
- Static and automated unit tests guard against regressions on new tools, AST errors, citations, and `UnifiedSimTool` parameters while maintaining 100% `traceability_cli` and zero-allocation kernel compliance.
