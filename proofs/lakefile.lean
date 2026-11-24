import Lake
open Lake DSL

package «oxidize_math_proofs» where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib «OxidizeMath» where
  roots := #[`OxidizeMath]
