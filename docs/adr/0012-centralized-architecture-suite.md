# 12. Centralized Architecture Guide Suite and Markdown Link Verification

## Status
Accepted

## Context
The repository lacked centralized architecture blueprints and onboarding documentation across the 18 domain and infrastructure crates. Furthermore, the contributing guide referenced an obsolete repository clone URL and incomplete test commands that only tested a single crate instead of the full workspace. In addition, internal markdown links were not validated systematically during pre-commit linting.

## Decision
1. Established a centralized Markdown architecture guide suite under `docs/architecture/` containing:
   - System overview, five-layer architectural model, and crate dependency topology (`system_overview.md`).
   - CERA framework guide describing representation alignment, latent space partitioning, and Earth Mover's Distance loss (`cera_framework.md`).
   - Zero-allocation runtime guide detailing NASA Power of 10 Rule 3 enforcement, `VerifiedAllocator`, and `VerifiedEngine` metrics (`zero_allocation_runtime.md`).
   - Contributor onboarding walkthrough for creating simulation tabs with `InteractiveTool` (`interactive_tool_walkthrough.md`).
2. Created a centralized documentation index at `docs/README.md`.
3. Updated `CONTRIBUTING.md` with the verified OxidizeMath clone URL (`https://github.com/fderuiter/OxidizeMath.git`), workspace-wide test commands (`cargo test --workspace`), and links to architecture guides.
4. Updated `README.md` to reference the correct repository URL and direct contributors to the new architecture documentation suite.
5. Integrated an automated markdown link verification task (`check-markdown-links`) into the workspace lint suite (`cargo run -p xtask -- lint-all`).

## Consequences
- New contributors can clone the correct repository and run verified tests across all workspace member crates.
- System architecture, CERA modeling, and runtime verification constraints are documented with Mermaid diagrams and valid Rust examples.
- Automated linting verifies that internal relative documentation links resolve to existing files.
