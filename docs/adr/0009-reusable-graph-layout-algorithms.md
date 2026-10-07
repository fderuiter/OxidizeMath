# Architecture Decision Record 0009: Reusable Pure Math Graph Layout Algorithms with Toolbar Controls

## Context
`GraphEditorTool` stored node positions manually in a `HashMap<usize, Pos2>`. It lacked automated graph layout algorithms and UI layout controls, requiring users to manually position every vertex when generating or exploring complex graph structures.

## Decision
We implemented reusable pure math graph layout algorithms in `crates/pure_math/src/pure_math/graph_theory/layout.rs` and integrated corresponding auto-layout toolbar controls into `GraphEditorTool`:
1. Added layout algorithms (`circular_layout`, `grid_layout`, `force_directed_layout`) in `crates/pure_math/src/pure_math/graph_theory/layout.rs` operating on bounding box constraints `(min_x, min_y, max_x, max_y)`.
2. Re-exported the `layout` module in `crates/pure_math/src/pure_math/graph_theory/mod.rs`.
3. Integrated toolbar controls ("⭕ Circular", "▦ Grid", "🧲 Force-Directed") into `GraphEditorTool::show_ui` to trigger node position recalculation and update `self.node_positions`.
4. Added unit test suite in `crates/pure_math/tests/graph_theory.rs` verifying boundary compliance, non-overlapping vertex positioning, zero/single-node edge cases, and disconnected component handling.

## Consequences
- **Positive:** Automated graph layout algorithms are reusable across pure math algorithms and GUI visualization components.
- **Positive:** Interactive layout controls eliminate manual vertex dragging overhead for graph exploration.
- **Positive:** NASA Power of 10 Rule 4 (<= 60 statements per verified function) and file length limits are strictly maintained.
