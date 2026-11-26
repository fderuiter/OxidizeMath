# Oxidize CLI & Simulation Examples

Command-line application and runnable example suite for the [`oxidizemath`](../../crates/04_bindings/oxidizemath/README.md) scientific simulation library.

> **Note:** For full workspace architecture, see the [Project Root README](../../README.md) and the [Examples Domain Catalog](examples/README.md).

## 1. Overview & Architecture

`oxidize-cli` decouples the 32 runnable scientific demonstrations and micro-benchmarks from the core `oxidizemath` library facade so downstream library consumers can depend on `oxidizemath` with `default = []` and zero CLI target overhead.

- **Domain-Gated `[[example]]` Targets**: Every simulation example declares explicit `required-features` (`pure_math`, `applied`, `ai`, `biology`, `climate`, `epidemiology`, or `physics`) forwarded directly to `oxidizemath`.
- **Facade-Only Imports**: Examples consume public exports strictly through `oxidizemath::*` without bypassing into internal tier crates.
- **Automated Smoke Verification**: `cargo run -p xtask -- smoke-examples` discovers and executes all 32 examples with `MATH_EXPLORER_SMOKE=1` in isolated sandboxes.

## 2. Developer Instructions

### Inspecting CLI Metadata

```bash
cargo run --package oxidize-cli
```

### Running Domain Simulation Examples

Pass `--features <domain>` and `--example <name>` to run any cataloged demonstration:

```bash
cargo run --release --package oxidize-cli --features physics --example hello_world
cargo run --release --package oxidize-cli --features biology --example hodgkin_huxley_demo
cargo run --release --package oxidize-cli --features ai --example transformer_demo
```

See [examples/README.md](examples/README.md) for the complete 32-example catalog and [CONTRIBUTING.md](../../CONTRIBUTING.md) for contributor standards.
