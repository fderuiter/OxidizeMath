# Architecture Decision Record: Decoupling Mathematical Models and Citation Databases

## Status

Accepted

## Context
Previously, pure mathematical simulation routines and the global, thread-safe citation database were heavily coupled under `math_commons` through macro-driven dynamic registration side-effects during metadata retrieval. This introduced unnecessary runtime state tracking and lock contention into headless computational tasks, while also requiring computational threads to link against global state.

## Decision
We physically separated the academic citation database and metadata traits from the low-level calculation routines:
1. Created a new dedicated scientific metadata library, `scientific_metadata`, containing `CitationRegistry` and `TheoryDescribable`.
2. Cleaned `math_commons` of all citation registry database structures, transforming it into a completely stateless core math library.
3. Updated the automated macro-driven compile-time metadata generation (`verified_engine_macros`) to output pure, side-effect-free getters for the `theory_citation` method instead of calling dynamic write-registries during evaluation.
4. Refactored the interactive visualization portal (`math_explorer_gui`) to perform on-demand citation resolution and registration when models are active in the graphical workspace.

## Consequences
- **Positive:** Mathematical computations can run completely statelessly without referencing or compiling global citation tracking registries.
- **Positive:** High performance and thread-safety for computational simulations are maintained due to the absence of global write locks during evaluation.
- **Positive:** Interactive visualization tools retain complete, on-demand academic reference resolution and portal reflection capabilities.
