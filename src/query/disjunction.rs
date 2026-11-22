#[cfg(test)]
use std::cmp::Ordering;

use common::TinySet;

#[cfg(test)]
use crate::docset::SeekDangerResult;
use crate::docset::COLLECT_BLOCK_BUFFER_LEN;
use crate::query::score_combiner::DoNothingCombiner;
use crate::query::{ScoreCombiner, Scorer};
use crate::{DocId, DocSet, Score, TERMINATED};

const HORIZON: u32 = 64u32 * 64u32;
const HORIZON_NUM_TINYBITSETS: usize = HORIZON as usize / 64;

/// `Disjunction` is responsible for merging `DocSet` from multiple
/// sources and filtering out documents that appear fewer times than
/// `minimum_matches_required`.
///
/// Uses a 4096-doc sliding window (`HORIZON`) with per-doc match counts and a
/// `TinySet` presence bitmap, plus pivot seeking across clauses to skip empty
/// gaps without touching intermediate documents.
pub struct Disjunction<TScorer, TScoreCombiner = DoNothingCombiner> {
    docsets: Vec<TScorer>,
    minimum_matches_required: usize,
    counts: Box<[u32; HORIZON as usize]>,
    bitsets: Box<[TinySet; HORIZON_NUM_TINYBITSETS]>,
    scores: Box<[TScoreCombiner; HORIZON as usize]>,
    bucket_idx: usize,
    window_start_doc: DocId,
    current_doc: DocId,
    current_score: Score,
}

#[cfg(test)]
struct ScorerWrapper<T> {
    scorer: T,
    current_doc: DocId,
}

#[cfg(test)]
pub(super) fn seek_danger_test_wrapper(scorer: Box<dyn Scorer>) -> impl DocSet {
    let current_doc = scorer.doc();
    ScorerWrapper {
        scorer,
        current_doc,
    }
}

#[cfg(test)]
impl<T: Scorer> PartialEq for ScorerWrapper<T> {
    fn eq(&self, other: &Self) -> bool {
        self.doc() == other.doc()
    }
}

#[cfg(test)]
impl<T: Scorer> Eq for ScorerWrapper<T> {}

#[cfg(test)]
impl<T: Scorer> PartialOrd for ScorerWrapper<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
impl<T: Scorer> Ord for ScorerWrapper<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.doc().cmp(&other.doc()).reverse()
    }
}

#[cfg(test)]
impl<T: Scorer> DocSet for ScorerWrapper<T> {
    fn advance(&mut self) -> DocId {
        let doc_id = self.scorer.advance();
        self.current_doc = doc_id;
        doc_id
    }
    fn seek(&mut self, target: DocId) -> DocId {
        let doc_id = self.scorer.seek(target);
        self.current_doc = doc_id;
        doc_id
    }
    fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        let result = self.scorer.seek_danger(target);
        if result == SeekDangerResult::Found {
            self.current_doc = target;
        }
        result
    }

    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn size_hint(&self) -> u32 {
        self.scorer.size_hint()
    }

    fn cost(&self) -> u64 {
        self.scorer.cost()
    }
}

#[inline(never)]
fn refill_window<TScorer: Scorer, TScoreCombiner: ScoreCombiner>(
    docsets: &mut Vec<TScorer>,
    minimum_matches_required: usize,
    counts: &mut [u32; HORIZON as usize],
    bitsets: &mut [TinySet; HORIZON_NUM_TINYBITSETS],
    scores: &mut [TScoreCombiner; HORIZON as usize],
    min_doc: DocId,
) -> bool {
    let horizon = min_doc.saturating_add(HORIZON).min(TERMINATED);
    counts.fill(0);
    let min_match_u32 = minimum_matches_required as u32;
    let mut any_match = false;
    for scorer in docsets.iter_mut() {
        loop {
            let doc = scorer.doc();
            if doc >= horizon {
                break;
            }
            let delta = (doc - min_doc) as usize;
            let prev = counts[delta];
            if prev == 0 {
                scores[delta].clear();
            }
            let next = prev + 1;
            counts[delta] = next;
            scores[delta].update(scorer);
            if next == min_match_u32 {
                bitsets[delta / 64].insert_mut((delta % 64) as u32);
                any_match = true;
            }
            scorer.advance();
        }
    }
    docsets.retain(|scorer| scorer.doc() != TERMINATED);
    any_match
}

impl<TScorer: Scorer, TScoreCombiner: ScoreCombiner> Disjunction<TScorer, TScoreCombiner> {
    pub fn new<T: IntoIterator<Item = TScorer>>(
        docsets: T,
        score_combiner: TScoreCombiner,
        minimum_matches_required: usize,
    ) -> Self {
        debug_assert!(
            minimum_matches_required > 1,
            "union scorer works better if just one matches required"
        );
        let active_docsets: Vec<TScorer> = docsets
            .into_iter()
            .filter(|doc| doc.doc() != TERMINATED)
            .collect();
        let mut disjunction = Self {
            docsets: active_docsets,
            minimum_matches_required,
            counts: vec![0u32; HORIZON as usize]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            bitsets: Box::new([TinySet::empty(); HORIZON_NUM_TINYBITSETS]),
            scores: Box::new([score_combiner; HORIZON as usize]),
            bucket_idx: HORIZON_NUM_TINYBITSETS,
            window_start_doc: 0,
            current_doc: TERMINATED,
            current_score: 0.0,
        };
        if minimum_matches_required > disjunction.docsets.len() {
            disjunction.docsets.clear();
            return disjunction;
        }
        disjunction.advance();
        disjunction
    }

    /// Finds the earliest doc where at least `minimum_matches_required` clauses can match,
    /// seeks lagging clauses to that pivot, and populates the sliding window.
    #[inline(never)]
    fn refill(&mut self) -> bool {
        let min_match = self.minimum_matches_required;
        loop {
            if self.docsets.len() < min_match {
                return false;
            }
            self.docsets.sort_unstable_by_key(DocSet::doc);
            let pivot = self.docsets[min_match - 1].doc();
            if pivot == TERMINATED {
                self.docsets.clear();
                return false;
            }
            let mut sought_past_pivot = false;
            for scorer in &mut self.docsets[..min_match - 1] {
                if scorer.doc() < pivot {
                    if scorer.seek(pivot) > pivot {
                        sought_past_pivot = true;
                    }
                }
            }
            if sought_past_pivot {
                self.docsets.retain(|scorer| scorer.doc() != TERMINATED);
                continue;
            }
            self.window_start_doc = pivot;
            self.bucket_idx = 0;
            if refill_window(
                &mut self.docsets,
                min_match,
                &mut self.counts,
                &mut self.bitsets,
                &mut self.scores,
                pivot,
            ) {
                return true;
            }
        }
    }

    #[inline]
    fn advance_buffered(&mut self) -> bool {
        while self.bucket_idx < HORIZON_NUM_TINYBITSETS {
            if let Some(val) = self.bitsets[self.bucket_idx].pop_lowest() {
                let delta = val + (self.bucket_idx as u32) * 64;
                self.current_doc = self.window_start_doc + delta;
                self.current_score = self.scores[delta as usize].score();
                return true;
            }
            self.bucket_idx += 1;
        }
        false
    }
}

impl<TScorer: Scorer, TScoreCombiner: ScoreCombiner> DocSet
    for Disjunction<TScorer, TScoreCombiner>
{
    fn advance(&mut self) -> DocId {
        loop {
            if self.advance_buffered() {
                return self.current_doc;
            }
            if !self.refill() {
                self.current_doc = TERMINATED;
                self.current_score = 0.0;
                return TERMINATED;
            }
        }
    }

    fn fill_buffer(&mut self, buffer: &mut [DocId; COLLECT_BLOCK_BUFFER_LEN]) -> usize {
        if self.current_doc == TERMINATED {
            return 0;
        }
        buffer[0] = self.current_doc;
        let mut count = 1;

        loop {
            while self.bucket_idx < HORIZON_NUM_TINYBITSETS {
                let mut tinyset = self.bitsets[self.bucket_idx];
                while let Some(val) = tinyset.pop_lowest() {
                    let delta = val + (self.bucket_idx as u32) * 64;
                    self.current_doc = self.window_start_doc + delta;
                    if count >= COLLECT_BLOCK_BUFFER_LEN {
                        self.bitsets[self.bucket_idx] = tinyset;
                        self.current_score = self.scores[delta as usize].score();
                        return COLLECT_BLOCK_BUFFER_LEN;
                    }
                    buffer[count] = self.current_doc;
                    count += 1;
                }
                self.bitsets[self.bucket_idx] = TinySet::empty();
                self.bucket_idx += 1;
            }

            if !self.refill() {
                self.current_doc = TERMINATED;
                self.current_score = 0.0;
                return count;
            }
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.current_doc >= target {
            return self.current_doc;
        }
        let gap = target.saturating_sub(self.window_start_doc);
        if self.bucket_idx < HORIZON_NUM_TINYBITSETS && gap < HORIZON {
            let new_bucket_idx = gap as usize / 64;
            if new_bucket_idx > self.bucket_idx {
                for obsolete in &mut self.bitsets[self.bucket_idx..new_bucket_idx] {
                    obsolete.clear();
                }
                self.bucket_idx = new_bucket_idx;
            }
            let mut doc = self.current_doc;
            while doc < target {
                doc = self.advance();
            }
            return doc;
        }
        for obsolete in self.bitsets.iter_mut() {
            *obsolete = TinySet::empty();
        }
        self.bucket_idx = HORIZON_NUM_TINYBITSETS;
        for docset in &mut self.docsets {
            if docset.doc() < target {
                docset.seek(target);
            }
        }
        self.docsets.retain(|docset| docset.doc() != TERMINATED);
        self.advance()
    }

    #[inline]
    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn size_hint(&self) -> u32 {
        self.docsets
            .iter()
            .map(DocSet::size_hint)
            .max()
            .unwrap_or(0u32)
    }

    fn cost(&self) -> u64 {
        self.docsets.iter().map(DocSet::cost).max().unwrap_or(0u64)
    }
}

impl<TScorer: Scorer, TScoreCombiner: ScoreCombiner> Scorer
    for Disjunction<TScorer, TScoreCombiner>
{
    #[inline]
    fn score(&mut self) -> Score {
        self.current_score
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::Disjunction;
    use crate::query::score_combiner::DoNothingCombiner;
    use crate::query::{ConstScorer, Scorer, SumCombiner, VecDocSet};
    use crate::{DocId, DocSet, Score, TERMINATED};

    fn conjunct<T: Ord + Copy>(arrays: &[Vec<T>], pass_line: usize) -> Vec<T> {
        let mut counts = BTreeMap::new();
        for array in arrays {
            for &element in array {
                *counts.entry(element).or_insert(0) += 1;
            }
        }
        counts
            .iter()
            .filter_map(|(&element, &count)| {
                if count >= pass_line {
                    Some(element)
                } else {
                    None
                }
            })
            .collect()
    }

    fn aux_test_conjunction(vals: Vec<Vec<u32>>, min_match: usize) {
        let mut union_expected = VecDocSet::from(conjunct(&vals, min_match));
        let make_scorer = || {
            Disjunction::new(
                vals.iter()
                    .cloned()
                    .map(VecDocSet::from)
                    .map(|d| ConstScorer::new(d, 1.0)),
                DoNothingCombiner,
                min_match,
            )
        };
        let mut scorer: Disjunction<_, DoNothingCombiner> = make_scorer();
        let mut count = 0;
        while scorer.doc() != TERMINATED {
            assert_eq!(union_expected.doc(), scorer.doc());
            assert_eq!(union_expected.advance(), scorer.advance());
            count += 1;
        }
        assert_eq!(union_expected.advance(), TERMINATED);
        assert_eq!(count, make_scorer().count_including_deleted());
    }

    #[should_panic]
    #[test]
    fn test_arg_check1() {
        aux_test_conjunction(vec![], 0);
    }

    #[should_panic]
    #[test]
    fn test_arg_check2() {
        aux_test_conjunction(vec![], 1);
    }

    #[test]
    fn test_corner_case() {
        aux_test_conjunction(vec![], 2);
        aux_test_conjunction(vec![vec![]; 1000], 2);
        aux_test_conjunction(vec![vec![]; 100], usize::MAX);
        aux_test_conjunction(vec![vec![0xC0FFEE]; 10000], usize::MAX);
        aux_test_conjunction((1..10000u32).map(|i| vec![i]).collect::<Vec<_>>(), 2);
    }

    #[test]
    fn test_conjunction() {
        aux_test_conjunction(
            vec![
                vec![1, 3333, 100000000u32],
                vec![1, 2, 100000000u32],
                vec![1, 2, 100000000u32],
            ],
            2,
        );
        aux_test_conjunction(
            vec![vec![8], vec![3, 4, 0xC0FFEEu32], vec![1, 2, 100000000u32]],
            2,
        );
        aux_test_conjunction(
            vec![
                vec![1, 3333, 100000000u32],
                vec![1, 2, 100000000u32],
                vec![1, 2, 100000000u32],
            ],
            3,
        )
    }

    // This dummy scorer does nothing but yield doc id increasingly.
    // with constant score 1.0
    #[derive(Clone)]
    struct DummyScorer {
        cursor: usize,
        foo: Vec<(DocId, f32)>,
    }

    impl DummyScorer {
        fn new(doc_score: Vec<(DocId, f32)>) -> Self {
            Self {
                cursor: 0,
                foo: doc_score,
            }
        }
    }

    impl DocSet for DummyScorer {
        fn advance(&mut self) -> DocId {
            self.cursor += 1;
            self.doc()
        }

        fn doc(&self) -> DocId {
            self.foo.get(self.cursor).map(|x| x.0).unwrap_or(TERMINATED)
        }

        fn size_hint(&self) -> u32 {
            self.foo.len() as u32
        }
    }

    impl Scorer for DummyScorer {
        #[inline]
        fn score(&mut self) -> Score {
            self.foo.get(self.cursor).map(|x| x.1).unwrap_or(0.0)
        }
    }

    #[test]
    fn test_score_calculate() {
        let mut scorer = Disjunction::new(
            vec![
                DummyScorer::new(vec![(1, 1f32), (2, 1f32)]),
                DummyScorer::new(vec![(1, 1f32), (3, 1f32)]),
                DummyScorer::new(vec![(1, 1f32), (4, 1f32)]),
                DummyScorer::new(vec![(1, 1f32), (2, 1f32)]),
                DummyScorer::new(vec![(1, 1f32), (2, 1f32)]),
            ],
            SumCombiner::default(),
            3,
        );
        assert_eq!(scorer.score(), 5.0);
        assert_eq!(scorer.advance(), 2);
        assert_eq!(scorer.score(), 3.0);
    }

    #[test]
    fn test_score_calculate_corner_case() {
        let mut scorer = Disjunction::new(
            vec![
                DummyScorer::new(vec![(1, 1f32), (2, 1f32)]),
                DummyScorer::new(vec![(1, 1f32), (3, 1f32)]),
                DummyScorer::new(vec![(1, 1f32), (3, 1f32)]),
            ],
            SumCombiner::default(),
            2,
        );
        assert_eq!(scorer.doc(), 1);
        assert_eq!(scorer.score(), 3.0);
        assert_eq!(scorer.advance(), 3);
        assert_eq!(scorer.score(), 2.0);
    }
}
