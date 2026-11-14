# OxidizeMath Architecture Guide Suite

This directory contains the core architecture documentation for the OxidizeMath repository. These documents explain system structure, domain models, runtime safety rules, and contributor workflows.

## Table of Contents

- [System Overview & Crate Topology](system_overview.md): High-level system layers, crate dependency topology, and workspace catalog across all 18 crates.
- [CERA Framework Guide](cera_framework.md): Climate-invariant Encoding through Representation Alignment, latent space alignment, and the climate modeling pipeline.
- [Zero-Allocation Runtime Guide](zero_allocation_runtime.md): NASA Power of 10 Rule 3 enforcement, `VerifiedAllocator`, and `VerifiedEngine` runtime checks.
- [Interactive Tool Walkthrough](interactive_tool_walkthrough.md): Step-by-step developer guide to implement interactive simulations using `InteractiveTool`.
- [Architectural Decision Records](../adr/0001-dynamic-workspace-discovery.md): Historical architecture records and design decisions.

## Guiding Principles

1. **Active Documentation**: Code explains how; documentation explains why. Keep documentation accurate and tested.
2. **Simplified Technical English**: Write guides in ASD-STE100 Simplified Technical English. Use short sentences and clear instructions.
3. **Formal Verification**: Verify simulation logic against theoretical invariants using compile-time and runtime checks.
4. **Zero Unintended Allocation**: Enforce static pre-allocation during real-time simulation loops.
