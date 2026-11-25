// [cite:math_memory]
use num_complex::Complex;

use crate::{ArenaError, BitSlice, ByteArena, FlagSlice, padded_slice_bytes, slice_byte_len};
#[cfg(feature = "alloc")]
use crate::{ArenaLayout, SolverWorkspace};

#[test]
fn carves_aligned_heterogeneous_pod_slices() {
    let mut raw = [0xFF_u8; 512];
    // Intentionally start at an odd byte offset to exercise runtime alignment rounding.
    let mut arena = ByteArena::new(&mut raw[1..]);
    let initial_cap = arena.capacity();
    assert_eq!(arena.used(), 0);

    {
        let mut scope = arena.reborrow();
        let u_idx = scope.alloc_slice::<usize>(3).unwrap();
        let words = scope.alloc_slice::<u64>(2).unwrap();
        let f32s = scope.alloc_slice::<f32>(5).unwrap();
        let f64s = scope.alloc_slice::<f64>(4).unwrap();
        let c32s = scope.alloc_slice::<Complex<f32>>(3).unwrap();
        let c64s = scope.alloc_slice::<Complex<f64>>(2).unwrap();

        assert_eq!(
            (u_idx.as_ptr() as usize) % core::mem::align_of::<usize>(),
            0
        );
        assert_eq!((words.as_ptr() as usize) % core::mem::align_of::<u64>(), 0);
        assert_eq!((f32s.as_ptr() as usize) % core::mem::align_of::<f32>(), 0);
        assert_eq!((f64s.as_ptr() as usize) % core::mem::align_of::<f64>(), 0);
        assert_eq!(
            (c32s.as_ptr() as usize) % core::mem::align_of::<Complex<f32>>(),
            0
        );
        assert_eq!(
            (c64s.as_ptr() as usize) % core::mem::align_of::<Complex<f64>>(),
            0
        );

        // Verify zero-initialization despite 0xFF backing buffer.
        assert!(u_idx.iter().all(|&v| v == 0));
        assert!(f64s.iter().all(|&v| v == 0.0));
        assert!(c64s.iter().all(|&z| z == Complex::new(0.0, 0.0)));

        u_idx[2] = 42;
        words[1] = 0xDEAD_BEEF;
        f32s[4] = -3.25;
        f64s[3] = 6.5;
        c32s[1] = Complex::new(1.5, -2.5);
        c64s[0] = Complex::new(-4.0, 8.0);

        assert_eq!(u_idx[2], 42);
        assert_eq!(words[1], 0xDEAD_BEEF);
        assert!((f32s[4] + 3.25).abs() < 1e-6);
        assert!((f64s[3] - 6.5).abs() < 1e-12);
        assert_eq!(c32s[1], Complex::new(1.5, -2.5));
        assert_eq!(c64s[0], Complex::new(-4.0, 8.0));
        assert!(scope.used() > 0);
    }

    // Scope dropped -> parent arena cursor automatically restored.
    assert_eq!(arena.used(), 0);
    assert_eq!(arena.remaining(), initial_cap);
}

#[test]
fn carves_zeroed_flag_and_bit_views() {
    let mut raw = [0xFF_u8; 256];
    let mut arena = ByteArena::new(&mut raw[1..]);
    {
        let mut scope = arena.reborrow();
        let mut flags = scope.alloc_flags(9).unwrap();
        let mut bits = scope.alloc_bits(130).unwrap();

        assert_eq!(flags.count_ones(), 0);
        assert_eq!(bits.count_ones(), 0);

        flags.set(0, true);
        flags.set(8, true);
        bits.set(0, true);
        bits.set(64, true);
        bits.set(129, true);

        assert!(flags.get(0) && !flags.get(1) && flags.get(8));
        assert_eq!(flags.count_ones(), 2);
        assert!(bits.get(0) && bits.get(64) && bits.get(129) && !bits.get(63));
        assert_eq!(bits.count_ones(), 3);
    }
    assert_eq!(arena.used(), 0);
}

#[test]
fn overflow_and_exhaustion_return_arena_errors() {
    let mut raw = [0_u8; 32];
    let mut arena = ByteArena::new(&mut raw);

    let overflow_err = arena.alloc_slice::<f64>(usize::MAX).unwrap_err();
    assert!(matches!(overflow_err, ArenaError::CapacityOverflow { .. }));

    let mut scope = arena.reborrow();
    assert!(matches!(
        scope.alloc_slice::<u64>(usize::MAX),
        Err(ArenaError::CapacityOverflow { .. })
    ));
    assert!(matches!(
        scope.alloc_slice::<f64>(5),
        Err(ArenaError::InsufficientCapacity { .. })
    ));
    assert!(matches!(
        slice_byte_len::<Complex<f64>>(usize::MAX),
        Err(ArenaError::CapacityOverflow { .. })
    ));
    assert!(matches!(
        padded_slice_bytes::<f64>(usize::MAX),
        Err(ArenaError::CapacityOverflow { .. })
    ));
}

#[test]
fn checkpoint_rewind_and_nested_reborrow_restore_cursors() {
    let mut raw = [0_u8; 128];
    let mut arena = ByteArena::new(&mut raw);

    {
        let first = arena.alloc_slice::<u64>(2).unwrap();
        first[0] = 7;
    }
    let cp = arena.checkpoint();
    assert!(cp.offset() >= 16);

    {
        let second = arena.alloc_slice::<f64>(4).unwrap();
        second[3] = 9.0;
    }
    assert!(arena.used() > cp.offset());

    arena.rewind(cp).unwrap();
    assert_eq!(arena.used(), cp.offset());
    arena.reset();
    assert_eq!(arena.used(), 0);
    assert!(matches!(
        arena.rewind(cp),
        Err(ArenaError::InvalidCheckpoint { .. })
    ));

    let mut outer = arena.reborrow();
    let head = outer.alloc_slice::<u64>(2).unwrap();
    head[1] = 99;
    let outer_rem_before = outer.remaining();
    {
        let mut inner = outer.reborrow();
        let scratch = inner.alloc_slice::<f64>(4).unwrap();
        scratch[0] = 123.0;
        assert!(inner.remaining() < outer_rem_before);
    }
    assert_eq!(outer.remaining(), outer_rem_before);
    assert_eq!(head[1], 99);
}

#[test]
fn flag_and_bit_slices_mask_and_normalize_safely() {
    let mut noisy = [0_u8, 255_u8, 2_u8, 0_u8];
    let mut flags = FlagSlice::from_bytes(&mut noisy);
    assert!(!flags.get(0) && flags.get(1) && flags.get(2) && !flags.get(3));
    assert_eq!(flags.as_bytes(), &[0, 1, 1, 0]);
    flags.fill(true);
    assert_eq!(flags.count_ones(), 4);
    flags.clear_all();
    assert_eq!(flags.count_ones(), 0);

    let mut words = [u64::MAX; 2];
    let mut bits = BitSlice::from_words(&mut words, 70).unwrap();
    assert_eq!(bits.count_ones(), 70);
    assert_eq!(bits.try_get(70), None);
    bits.set(69, false);
    assert_eq!(bits.count_ones(), 69);
    bits.fill(false);
    assert_eq!(bits.count_ones(), 0);
}

#[cfg(feature = "alloc")]
#[test]
fn owned_byte_arena_layout_and_solver_workspace_without_std() {
    use crate::OwnedByteArena;

    let mut layout = ArenaLayout::new();
    let r_idx = layout.add_slice::<usize>(4).unwrap();
    let r_val = layout.add_slice::<f64>(4).unwrap();
    let mut owned = OwnedByteArena::with_capacity(layout.total_bytes());
    assert!(SolverWorkspace::byte_capacity(&owned) >= layout.total_bytes());
    owned.region_mut(r_idx)[3] = 11;
    owned.region_mut(r_val)[2] = 4.25;
    assert_eq!(owned.region(r_idx)[3], 11);
    assert!((owned.region(r_val)[2] - 4.25).abs() < 1e-12);
    SolverWorkspace::reset(&mut owned);
}

#[cfg(feature = "std")]
#[test]
fn owned_byte_arena_and_solver_workspace_run_zero_allocation() {
    use crate::OwnedByteArena;

    let mut layout = ArenaLayout::new();
    let r_f64 = layout.add_slice::<f64>(8).unwrap();
    let r_u32 = layout.add_slice::<u32>(5).unwrap();
    let r_c64 = layout.add_slice::<Complex<f64>>(4).unwrap();
    let mut owned = OwnedByteArena::with_capacity(layout.total_bytes());

    let (output, _metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        owned.region_mut(r_f64)[0] = 3.5;
        owned.region_mut(r_u32)[4] = 19;
        owned.region_mut(r_c64)[2] = Complex::new(2.0, -5.0);

        fn kernel(ws: &mut impl SolverWorkspace) -> f64 {
            let mut scope = ws.reborrow();
            let a = scope.retain_slice::<f64>(8).unwrap();
            let b = scope.retain_slice::<u32>(5).unwrap();
            let c = scope.retain_slice::<Complex<f64>>(4).unwrap();
            a[0] + f64::from(b[4]) + c[2].re
        }

        kernel(&mut owned)
    });

    assert!((output - 24.5).abs() < 1e-12);
}
