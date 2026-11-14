# ADR 0014: Fused Batch Linear Projections for Multi-Head Attention

## Context
In `crates/domain_ai/src/ai/transformer/attention.rs`, multi-head attention previously executed separate linear projection matrix multiplications per attention head in an isolated loop. For a layer with `h` attention heads, this triggered `3 * h` dynamic matrix allocations and fragmented matrix multiplications that prevented hardware GEMM parallelism.

## Decision
We modernized `MultiHeadAttention` to execute fused batch linear projections:
1. Fused the projection weights for Query, Key, and Value into unified matrices `w_q`, `w_k`, and `w_v` of dimension `(d_model, d_model)`.
2. Executed linear projections via a single matrix multiplication per tensor (`q * &self.w_q`, `k * &self.w_k`, `v * &self.w_v`), reducing linear projection dynamic allocations from `3 * h` to exactly 3.
3. Implemented head splitting using zero-allocation strided matrix views (`DMatrixView`), writing attention outputs directly into the concatenated output buffer.
4. Added backward conversion helpers (`from_head_weights` and `to_head_weights`) to guarantee full conversion compatibility with legacy per-head serialized weights formats.
5. Preserved the public `AttentionMechanism` trait contract, dimension checks, and exact numerical equivalence against reference implementations.

## Consequences
- **Performance**: Significant reduction in forward pass allocation overhead and improved tensor operation throughput.
- **Compatibility**: Backward conversion compatibility for model weights and full compatibility with existing transformers (`EncoderLayer`, `DecoderLayer`, `Transformer`).
- **Verification**: Validated numerical equivalence within floating point tolerance against reference unfused implementations and passed all architectural integrity checks.
