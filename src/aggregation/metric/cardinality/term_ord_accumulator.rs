//! Per-bucket term-ordinal accumulators used by the str cardinality
//! collector.
//!
//! A str cardinality bucket needs a *set of term ordinals*, and the right
//! representation depends on how large the segment's dictionary slice is:
//!
//!   * [`BitSet`] (from `common`): used when `column.max_value()` is small (<
//!     [`BITSET_MAX_TERM_ORD`]). Pre-allocated, no promotion machinery.
//!   * [`TermOrdSet`]: adaptive. Starts as an `FxHashSet` (cheap when few ords are seen) and
//!     promotes to a [`PagedBitset`] once occupancy crosses the density threshold.
//!
//! Both are exposed through the [`TermOrdAccumulator`] trait so that
//! `SegmentStrCardinalityCollector` can be generic over the choice and the hot
//! `collect()` loop monomorphizes to a direct call (no enum dispatch per
//! insert).

use common::{BitSet, TinySet};
use rustc_hash::FxHashSet;

/// Promote FxHashSet<u64> -> PagedBitset at ~3% density (`len * 32 >
/// dict_num_terms`). Past this point the bitset (~`dict_num_terms / 7.5`
/// bytes) is smaller than the hashset (~10 B/entry minimum) and avoids
/// the per-insert hash.
const PROMOTION_RATIO: u64 = 32;

// =================================================================
// PagedBitset: a sparse bitset indexed by term_ord.
//
// Used as the dense alternative to FxHashSet<u64> once a string
// cardinality bucket has accumulated enough unique term ordinals.
// Memory is bounded to (touched pages) * (page bytes), not
// (max_term_ord / 8).
//
// Page geometry mirrors `PagedTermMap` in `term_agg.rs`: 1024 ords
// per page, lazy `Vec<Option<Box<Page>>>` directory.
// =================================================================
const BITSET_PAGE_SHIFT: u32 = 10;
const BITSET_PAGE_BITS: u64 = 1u64 << BITSET_PAGE_SHIFT; // 1024
const BITSET_PAGE_MASK: u64 = BITSET_PAGE_BITS - 1;
const BITSET_WORDS_PER_PAGE: usize = (BITSET_PAGE_BITS / 64) as usize; // 16

#[derive(Clone)]
struct PagedBitsetPage {
    words: [TinySet; BITSET_WORDS_PER_PAGE],
}

impl PagedBitsetPage {
    fn new() -> Self {
        Self {
            words: [TinySet::empty(); BITSET_WORDS_PER_PAGE],
        }
    }
}

pub(crate) struct PagedBitset {
    pages: Vec<Option<Box<PagedBitsetPage>>>,
    /// Cached number of set bits, maintained on insert.
    count: u64,
}

impl PagedBitset {
    /// Allocates a directory big enough to hold ords up to and including
    /// `max_term_ord`. Pages are allocated lazily on first set.
    fn with_max_term_ord(max_term_ord: u64) -> Self {
        let max_page_idx = (max_term_ord >> BITSET_PAGE_SHIFT) as usize;
        let num_pages = max_page_idx + 1;
        Self {
            pages: vec![None; num_pages],
            count: 0,
        }
    }

    #[inline]
    fn insert(&mut self, term_ord: u64) {
        let page_idx = (term_ord >> BITSET_PAGE_SHIFT) as usize;
        let intra = term_ord & BITSET_PAGE_MASK;
        let word_idx = (intra >> 6) as usize;
        let bit_idx = (intra & 63) as u32;

        let page = match &mut self.pages[page_idx] {
            Some(p) => p,
            None => {
                self.pages[page_idx] = Some(Box::new(PagedBitsetPage::new()));
                self.pages[page_idx].as_mut().unwrap()
            }
        };
        let inserted = page.words[word_idx].insert_mut(bit_idx);
        self.count += u64::from(inserted);
    }

    /// Number of set bits. O(1).
    #[inline]
    fn len(&self) -> u64 {
        self.count
    }

    /// Iterate set ords in ascending order.
    fn iter_sorted(&self) -> impl Iterator<Item = u64> + '_ {
        self.pages
            .iter()
            .enumerate()
            .filter_map(|(page_idx, page_opt)| page_opt.as_ref().map(|p| (page_idx, p)))
            .flat_map(|(page_idx, page)| {
                let page_base_ord = (page_idx as u64) << BITSET_PAGE_SHIFT;
                page.words
                    .iter()
                    .enumerate()
                    .flat_map(move |(word_idx, &word)| {
                        let word_base_ord = page_base_ord + (word_idx as u64) * 64;
                        word.into_iter()
                            .map(move |bit| word_base_ord + u64::from(bit))
                    })
            })
    }
}

/// Threshold below which we use `BitSet` instead of `TermOrdSet`.
///
/// Both `BitSet` and `FxHashSet<u64>` have the same 32-byte struct, so the comparison is heap only:
///   * `BitSet` at T=256: 5 `TinySet` words covering 258 bits (with the missing-value sentinel) =
///     40 bytes.
///   * `FxHashSet<u64>` after one insert: 4-bucket hashbrown table ≈ 56 bytes
pub(crate) const BITSET_MAX_TERM_ORD: u64 = 256;

// =================================================================
// TermOrdAccumulator: per-bucket abstraction over the entries set.
//
// Implementations:
//   - `BitSet` (from `common`): used when `column.max_value()` is small (< BITSET_MAX_TERM_ORD).
//     Pre-allocated, no promotion.
//   - `TermOrdSet`: adaptive, starts as FxHashSet and promotes to a paged bitset when occupancy
//     crosses the density threshold (only if promotion is enabled — typically gated on top-level
//     aggregation).
//
// The trait lets `SegmentStrCardinalityCollector` be generic over the choice
// so the hot collect() loop monomorphizes to a direct call (no enum
// dispatch per insert).
// =================================================================
pub(crate) trait TermOrdAccumulator: Sized {
    /// Construct an empty accumulator.
    /// `max_term_ord_inclusive` is the largest term_ord that may be
    /// inserted (used to size pre-allocated bitsets and the dense bitset
    /// on promotion).
    fn new(max_term_ord_inclusive: u64) -> Self;
    fn insert(&mut self, term_ord: u64);
    fn extend_from_iter(&mut self, ords: impl IntoIterator<Item = u64>);
    /// Hook called once per ingested block. Adaptive impls use this to
    /// decide on sparse->dense promotion.
    fn maybe_compact(&mut self) {}
    fn len(&self) -> usize;
    fn iter_ords(&self) -> impl Iterator<Item = u64> + '_;
    /// Returns `true` if `iter_ords()` yields strictly ascending ordinals.
    fn is_iter_sorted(&self) -> bool;
}

impl TermOrdAccumulator for BitSet {
    #[inline]
    fn new(max_term_ord_inclusive: u64) -> Self {
        // `BitSet::with_max_value(M)` accepts ords in [0, M).
        // We need ords up to and including `max_term_ord_inclusive`, plus
        // the missing-value sentinel `column.max_value() + 1`.
        BitSet::with_max_value((max_term_ord_inclusive + 2) as u32)
    }
    #[inline]
    fn insert(&mut self, term_ord: u64) {
        BitSet::insert(self, term_ord as u32);
    }
    #[inline]
    fn len(&self) -> usize {
        BitSet::len(self)
    }
    fn iter_ords(&self) -> impl Iterator<Item = u64> + '_ {
        // `BitSet` itself doesn't expose iteration, but
        // `BitSet::tinyset(bucket)` does. Walk per-bucket and yield each
        // set bit. The capacity is `max_value()`; iterating to
        // `div_ceil(64)` covers every possible ord exactly once.
        let num_buckets = self.max_value().div_ceil(64);
        (0..num_buckets).flat_map(move |bucket| {
            let chunk_base = u64::from(bucket) * 64;
            self.tinyset(bucket)
                .into_iter()
                .map(move |bit| chunk_base + u64::from(bit))
        })
    }
    #[inline]
    fn is_iter_sorted(&self) -> bool {
        true
    }

    #[inline(never)] //< required to not have a perf regression
    fn extend_from_iter(&mut self, ords: impl IntoIterator<Item = u64>) {
        for ord in ords {
            <Self as TermOrdAccumulator>::insert(self, ord);
        }
    }
}

/// Upper bound (exclusive) for promoting `TermOrdSet` into a flat contiguous `Vec<u64>` bitset
/// (524_288 bits = 64 KiB) instead of a two-level `PagedBitset`. Avoids pointer chasing through
/// `Option<Box<PagedBitsetPage>>` for high-cardinality segments that still fit comfortably in L2.
const FLAT_BITSET_MAX_TERM_ORD: u64 = 524_288;

/// Contiguous flat bitset backed by `Vec<u64>` with branchless count maintenance.
pub(crate) struct FlatBitset {
    words: Vec<u64>,
    count: u64,
}

impl FlatBitset {
    fn with_max_term_ord(max_term_ord: u64) -> Self {
        let num_words = (max_term_ord as usize >> 6) + 1;
        Self {
            words: vec![0u64; num_words],
            count: 0,
        }
    }

    #[inline(always)]
    fn insert(&mut self, term_ord: u64) {
        let word_idx = (term_ord >> 6) as usize;
        let bit_mask = 1u64 << (term_ord & 63);
        let word = &mut self.words[word_idx];
        let prev = *word;
        let next = prev | bit_mask;
        *word = next;
        self.count += u64::from(next != prev);
    }

    #[inline]
    fn len(&self) -> u64 {
        self.count
    }

    fn iter_sorted(&self) -> impl Iterator<Item = u64> + '_ {
        self.words.iter().enumerate().flat_map(|(word_idx, &word)| {
            let base = (word_idx as u64) << 6;
            FlatWordIter { word, base }
        })
    }
}

struct FlatWordIter {
    word: u64,
    base: u64,
}

impl Iterator for FlatWordIter {
    type Item = u64;

    #[inline(always)]
    fn next(&mut self) -> Option<u64> {
        if self.word == 0 {
            return None;
        }
        let bit = self.word.trailing_zeros() as u64;
        self.word &= self.word - 1;
        Some(self.base + bit)
    }
}

// TermOrdSet: adaptive sparse->dense accumulator.
//
// Starts as an HashSet (cheap when few ords are seen). When occupancy
// crosses `len * PROMOTION_RATIO > max_term_ord_inclusive`, drains into
// a `FlatBitset` (for <= 64 KiB) or `PagedBitset` and continues dense.
pub(crate) struct TermOrdSet {
    inner: TermOrdSetInner,
    /// Largest term_ord that may be inserted. Used for both sizing the
    /// dense bitset on promotion and as the promotion-threshold reference.
    max_term_ord_inclusive: u64,
}

enum TermOrdSetInner {
    Sparse(FxHashSet<u64>),
    Flat(FlatBitset),
    Dense(PagedBitset),
}

impl TermOrdAccumulator for TermOrdSet {
    fn new(max_term_ord_inclusive: u64) -> Self {
        Self {
            inner: TermOrdSetInner::Sparse(FxHashSet::default()),
            max_term_ord_inclusive,
        }
    }

    #[inline]
    fn insert(&mut self, term_ord: u64) {
        match &mut self.inner {
            TermOrdSetInner::Sparse(set) => {
                set.insert(term_ord);
            }
            TermOrdSetInner::Flat(bitset) => bitset.insert(term_ord),
            TermOrdSetInner::Dense(bitset) => bitset.insert(term_ord),
        }
    }

    fn extend_from_iter(&mut self, ords: impl IntoIterator<Item = u64>) {
        match &mut self.inner {
            TermOrdSetInner::Sparse(set) => {
                set.extend(ords);
            }
            TermOrdSetInner::Flat(bitset) => {
                for ord in ords {
                    bitset.insert(ord);
                }
            }
            TermOrdSetInner::Dense(bitset) => {
                for ord in ords {
                    bitset.insert(ord);
                }
            }
        }
    }

    #[inline]
    fn is_iter_sorted(&self) -> bool {
        matches!(
            self.inner,
            TermOrdSetInner::Flat(_) | TermOrdSetInner::Dense(_)
        )
    }

    fn maybe_compact(&mut self) {
        let TermOrdSetInner::Sparse(set) = &mut self.inner else {
            return;
        };
        if set.len() as u64 * PROMOTION_RATIO <= self.max_term_ord_inclusive {
            return;
        }
        let max_with_missing = self.max_term_ord_inclusive + 1;
        let set = std::mem::take(set);
        if max_with_missing < FLAT_BITSET_MAX_TERM_ORD {
            let mut bitset = FlatBitset::with_max_term_ord(max_with_missing);
            for ord in set {
                bitset.insert(ord);
            }
            self.inner = TermOrdSetInner::Flat(bitset);
        } else {
            let mut bitset = PagedBitset::with_max_term_ord(max_with_missing);
            for ord in set {
                bitset.insert(ord);
            }
            self.inner = TermOrdSetInner::Dense(bitset);
        }
    }

    fn len(&self) -> usize {
        match &self.inner {
            TermOrdSetInner::Sparse(set) => set.len(),
            TermOrdSetInner::Flat(bitset) => bitset.len() as usize,
            TermOrdSetInner::Dense(bitset) => bitset.len() as usize,
        }
    }

    fn iter_ords(&self) -> impl Iterator<Item = u64> + '_ {
        match &self.inner {
            TermOrdSetInner::Sparse(set) => itertools::Either::Left(set.iter().copied()),
            TermOrdSetInner::Flat(bitset) => {
                itertools::Either::Right(itertools::Either::Left(bitset.iter_sorted()))
            }
            TermOrdSetInner::Dense(bitset) => {
                itertools::Either::Right(itertools::Either::Right(bitset.iter_sorted()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use common::BitSet;

    use super::{FlatBitset, PagedBitset, TermOrdAccumulator, TermOrdSet, PROMOTION_RATIO};

    /// Unit-test `FlatBitset`: cross-word inserts, idempotent duplicates, and sorted iteration.
    #[test]
    fn flat_bitset_basic() {
        let ords = [0u64, 1, 63, 64, 127, 128, 1024, 65_535, 100_000];
        let max_ord = *ords.iter().max().unwrap();
        let mut bitset = FlatBitset::with_max_term_ord(max_ord);
        for &ord in &ords {
            bitset.insert(ord);
            bitset.insert(ord);
        }
        assert_eq!(bitset.len(), ords.len() as u64);
        let collected: Vec<u64> = bitset.iter_sorted().collect();
        assert_eq!(collected, ords);
    }

    /// Unit-test the PagedBitset itself: cross-page inserts produce sorted
    /// iteration, len() matches the inserted set, and duplicates are
    /// idempotent.
    #[test]
    fn paged_bitset_basic() {
        // Span several pages: BITSET_PAGE_BITS = 1024, so ords > 1024 land
        // on the second page, > 2048 on the third, etc.
        let ords = [0u64, 1, 63, 64, 1023, 1024, 1025, 4096, 4097, 9999, 10_000];
        let max_ord = *ords.iter().max().unwrap();
        let mut bitset = PagedBitset::with_max_term_ord(max_ord);
        for &ord in &ords {
            bitset.insert(ord);
            // Idempotent: inserting again must not increase count.
            bitset.insert(ord);
        }
        assert_eq!(bitset.len(), ords.len() as u64);
        let collected: Vec<u64> = bitset.iter_sorted().collect();
        let mut expected: Vec<u64> = ords.to_vec();
        expected.sort_unstable();
        assert_eq!(collected, expected);
    }

    /// Unit-test `TermOrdSet`: starts Sparse, promotes to Dense on
    /// `maybe_compact` once the density threshold is crossed, and
    /// `iter_ords()` yields the same set in either state. Ords spanning
    /// multiple paged-bitset pages exercise the Dense iter ordering.
    #[test]
    fn term_ord_set_promotes_on_maybe_compact() {
        // Pick max so promotion needs few inserts: len * RATIO > max with
        // RATIO=32 and max=64 trips at len=3 (3*32=96 > 64).
        let max_term_ord = 64u64;
        let mut set = <TermOrdSet as TermOrdAccumulator>::new(max_term_ord);
        // Two inserts: should stay Sparse after maybe_compact (2 * RATIO = 64, not > 64).
        set.insert(0);
        set.insert(7);
        set.maybe_compact();
        assert_eq!(set.len(), 2);

        // Third insert promotes on next maybe_compact.
        set.insert(20);
        assert_eq!(set.len(), 3);
        // Sanity check: at len=3, 3 * PROMOTION_RATIO = 96 > 64.
        assert!(3u64 * PROMOTION_RATIO > max_term_ord);
        set.maybe_compact();

        // Post-promotion: extending continues to work.
        set.insert(15);
        set.insert(15); // dup
        assert_eq!(set.len(), 4);

        let mut collected: Vec<u64> = set.iter_ords().collect();
        collected.sort_unstable();
        assert_eq!(collected, vec![0, 7, 15, 20]);
    }

    /// Unit-test the `BitSet` impl of `TermOrdAccumulator`: insert,
    /// dedup, and iter_ords order.
    #[test]
    fn bitset_accumulator_basic() {
        let mut set = <BitSet as TermOrdAccumulator>::new(255);
        for ord in [0u64, 1, 63, 64, 65, 128, 200, 200, 0] {
            <BitSet as TermOrdAccumulator>::insert(&mut set, ord);
        }
        assert_eq!(<BitSet as TermOrdAccumulator>::len(&set), 7);
        let collected: Vec<u64> = set.iter_ords().collect();
        assert_eq!(collected, vec![0, 1, 63, 64, 65, 128, 200]);
    }
}
