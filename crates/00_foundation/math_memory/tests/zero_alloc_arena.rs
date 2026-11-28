//! Zero-allocation runtime and misalignment/diagnostic verification tests for `math_memory`.

use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use math_memory::{ArenaError, ArenaLayout, BitSlice, ByteArena, OwnedByteArena, SolverWorkspace};
use num_complex::Complex;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOC: VerifiedAllocator = VerifiedAllocator;

struct CodeCapture {
    code: Option<&'static str>,
}

impl DiagnosticVisitor for CodeCapture {
    fn record_str(&mut self, key: &str, value: &str) {
        if key == "code" {
            self.code = match value {
                "MEM-001" => Some("MEM-001"),
                "MEM-002" => Some("MEM-002"),
                "MEM-003" => Some("MEM-003"),
                "MEM-004" => Some("MEM-004"),
                "MEM-005" => Some("MEM-005"),
                _ => None,
            };
        }
    }

    fn record_display(&mut self, _key: &str, _value: &dyn core::fmt::Display) {}
}

#[test]
fn misaligned_stack_byte_arena_heterogeneous_reborrow_zero_alloc() {
    let mut stack_raw = [0_u8; 512];
    let (value, _metrics) = VerifiedEngine::run_verified(|| {
        // Deliberately slice at odd byte offset 1 to exercise runtime address alignment padding.
        let mut arena = ByteArena::new(&mut stack_raw[1..]);
        let initial_cap = arena.byte_capacity();
        assert_eq!(initial_cap, 511);

        let cp = arena.checkpoint();
        {
            let head = arena.alloc_slice::<u64>(2).expect("head u64");
            assert_eq!((head.as_ptr() as usize) % core::mem::align_of::<u64>(), 0);
            head[0] = 11;
            head[1] = 22;
        }
        let after_head = arena.used_bytes();
        assert!(after_head >= 16);

        {
            let mut sub = arena.reborrow();
            let mut cur = sub.cursor();
            let idx = cur.alloc_slice::<usize>(4).expect("usize slice");
            let words = cur.alloc_slice::<u64>(3).expect("u64 slice");
            let f32s = cur.alloc_slice::<f32>(5).expect("f32 slice");
            let f64s = cur.alloc_slice::<f64>(4).expect("f64 slice");
            let c32s = cur.alloc_slice::<Complex<f32>>(2).expect("c32 slice");
            let c64s = cur.alloc_slice::<Complex<f64>>(2).expect("c64 slice");
            let mut flags = cur.alloc_flags(7).expect("flag slice");
            let mut bits = cur.alloc_bits(130).expect("bit slice");

            assert_eq!((f64s.as_ptr() as usize) % core::mem::align_of::<f64>(), 0);
            assert_eq!(
                (c64s.as_ptr() as usize) % core::mem::align_of::<Complex<f64>>(),
                0
            );

            idx[3] = 9;
            words[2] = 99;
            f32s[4] = 1.25;
            f64s[3] = -4.5;
            c32s[1] = Complex::new(1.0, -2.0);
            c64s[1] = Complex::new(3.0, 4.0);
            flags.set(5, true);
            bits.set(0, true);
            bits.set(64, true);
            bits.set(129, true);

            assert!(flags.get(5));
            assert!(!flags.get(4));
            assert_eq!(bits.count_ones(), 3);
            assert_eq!(idx[3] + words[2] as usize, 108);
        }

        // Dropping `sub` automatically restores `arena`'s cursor to `after_head`.
        assert_eq!(arena.used_bytes(), after_head);

        // Rewinding `cp` restores cursor to 0 and zeroes freshly carved memory.
        arena.rewind(cp).expect("valid checkpoint");
        assert_eq!(arena.used_bytes(), 0);
        let fresh = arena.alloc_slice::<u64>(2).expect("re-carved u64");
        assert_eq!(fresh, &[0, 0]);
        fresh.len()
    });

    assert_eq!(value, 2);
}

#[test]
fn owned_byte_arena_layout_and_solver_workspace_zero_alloc() {
    let mut layout = ArenaLayout::new();
    let r_f64 = layout.reserve::<f64>(8).unwrap();
    let r_flags = layout.reserve_flags(6).unwrap();
    let r_bits = layout.reserve_bits(70).unwrap();
    let mut owned = OwnedByteArena::from_layout(&layout);
    let mut scratch_owned = OwnedByteArena::with_capacity(256);

    let (value, _metrics) = VerifiedEngine::run_verified(|| {
        {
            let mut raw = owned.raw_cursor();
            let vals = raw.region_mut(r_f64);
            let mut flags = raw.flags_mut(r_flags);
            let mut bits = raw.bits_mut(r_bits);
            vals[0] = 2.5;
            vals[7] = 7.5;
            flags.set(3, true);
            bits.set(69, true);
        }
        assert_eq!(
            owned.region_slice(r_f64)[0] + owned.region_slice(r_f64)[7],
            10.0
        );
        assert!(owned.flag_get(r_flags, 3));
        assert_eq!(BitSlice::required_words(70), owned.bit_words(r_bits).len());

        // Exercise `&mut impl SolverWorkspace` scoped reborrow in a loop.
        fn step(ws: &mut impl SolverWorkspace, scale: f64) -> Result<f64, ArenaError> {
            let mut frame = ws.reborrow();
            let buf = frame.alloc_slice::<f64>(8)?;
            buf[0] = scale * 2.0;
            Ok(buf[0])
        }

        let mut acc = 0.0;
        for k in 1..=4 {
            acc += step(&mut scratch_owned, k as f64).unwrap();
            assert_eq!(scratch_owned.used_bytes(), 0);
        }
        acc
    });

    assert_eq!(value, 20.0);
}

#[test]
fn arena_error_overflow_exhaustion_and_serde_diagnostics() {
    let mut small = [0_u8; 16];
    let mut arena = ByteArena::new(&mut small);

    let ovf = arena.alloc_slice::<u64>(usize::MAX).unwrap_err();
    assert_eq!(ovf, ArenaError::ArithmeticOverflow);
    assert_eq!(arena.used_bytes(), 0);

    let exh = arena.alloc_slice::<f64>(4).unwrap_err();
    assert!(matches!(exh, ArenaError::InsufficientBytes { .. }));
    assert_eq!(arena.used_bytes(), 0);

    let mut cap = CodeCapture { code: None };
    let (_, _) = VerifiedEngine::run_verified(|| {
        exh.visit_fields(&mut cap);
    });
    assert_eq!(exh.severity(), Severity::Error);
    assert_eq!(cap.code, Some("MEM-003"));

    #[cfg(feature = "serde")]
    {
        let json = serde_json::to_string(&exh).expect("serialize ArenaError");
        let roundtrip: ArenaError = serde_json::from_str(&json).expect("deserialize ArenaError");
        assert_eq!(roundtrip, exh);
    }
}
