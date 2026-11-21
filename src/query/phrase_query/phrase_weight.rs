use super::PhraseScorer;
use crate::docset::SeekDangerResult;
use crate::fieldnorm::FieldNormReader;
use crate::index::SegmentReader;
use crate::postings::{Postings, SegmentPostings};
use crate::query::bm25::Bm25Weight;
use crate::query::explanation::does_not_match;
use crate::query::{EmptyScorer, Explanation, Scorer, Weight};
use crate::schema::{IndexRecordOption, Term};
use crate::{DocId, DocSet, Score, TERMINATED};

pub struct PhraseWeight {
    phrase_terms: Vec<(usize, Term)>,
    similarity_weight_opt: Option<Bm25Weight>,
    slop: u32,
}

impl PhraseWeight {
    /// Creates a new phrase weight.
    /// If `similarity_weight_opt` is None, then scoring is disabled
    pub fn new(
        phrase_terms: Vec<(usize, Term)>,
        similarity_weight_opt: Option<Bm25Weight>,
    ) -> PhraseWeight {
        let slop = 0;
        PhraseWeight {
            phrase_terms,
            similarity_weight_opt,
            slop,
        }
    }

    fn fieldnorm_reader(&self, reader: &SegmentReader) -> crate::Result<FieldNormReader> {
        let field = self.phrase_terms[0].1.field();
        if self.similarity_weight_opt.is_some() {
            if let Some(fieldnorm_reader) = reader.fieldnorms_readers().get_field(field)? {
                return Ok(fieldnorm_reader);
            }
        }
        Ok(FieldNormReader::constant(reader.max_doc(), 1))
    }

    pub(crate) fn phrase_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
    ) -> crate::Result<Option<PhraseScorer<SegmentPostings>>> {
        let Some((seek_result, mut scorer)) = self.phrase_scorer_danger(reader, 0, boost)? else {
            return Ok(None);
        };
        if let SeekDangerResult::SeekLowerBound(target) = seek_result {
            if target < TERMINATED {
                scorer.seek(target);
            }
        }
        Ok(Some(scorer))
    }

    fn phrase_scorer_danger(
        &self,
        reader: &SegmentReader,
        target: DocId,
        boost: Score,
    ) -> crate::Result<Option<(SeekDangerResult, PhraseScorer<SegmentPostings>)>> {
        let similarity_weight_opt = self
            .similarity_weight_opt
            .as_ref()
            .map(|similarity_weight| similarity_weight.boost_by(boost));
        let fieldnorm_reader = self.fieldnorm_reader(reader)?;
        let mut term_postings_list = Vec::new();
        for &(offset, ref term) in &self.phrase_terms {
            let Some(postings) = reader
                .inverted_index(term.field())?
                .read_postings(term, IndexRecordOption::WithFreqsAndPositions)?
            else {
                return Ok(None);
            };
            term_postings_list.push((offset, postings));
        }
        Ok(Some(PhraseScorer::new_danger(
            term_postings_list,
            similarity_weight_opt,
            fieldnorm_reader,
            self.slop,
            0,
            target,
        )))
    }

    pub fn slop(&mut self, slop: u32) {
        self.slop = slop;
    }

    pub(crate) fn is_prunable_exact_phrase(&self) -> bool {
        self.slop == 0 && self.similarity_weight_opt.is_some() && self.phrase_terms.len() >= 2
    }
}

impl Weight for PhraseWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        if let Some(scorer) = self.phrase_scorer(reader, boost)? {
            Ok(Box::new(scorer))
        } else {
            Ok(Box::new(EmptyScorer))
        }
    }

    fn scorer_danger(
        &self,
        reader: &SegmentReader,
        target: DocId,
        boost: Score,
    ) -> crate::Result<(SeekDangerResult, Box<dyn Scorer>)> {
        let Some((seek_result, scorer)) = self.phrase_scorer_danger(reader, target, boost)? else {
            return Ok((
                SeekDangerResult::SeekLowerBound(TERMINATED),
                Box::new(EmptyScorer),
            ));
        };
        Ok((seek_result, Box::new(scorer)))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let scorer_opt = self.phrase_scorer(reader, 1.0)?;
        if scorer_opt.is_none() {
            return Err(does_not_match(doc));
        }
        let mut scorer = scorer_opt.unwrap();
        if scorer.seek(doc) != doc {
            return Err(does_not_match(doc));
        }
        let fieldnorm_reader = self.fieldnorm_reader(reader)?;
        let fieldnorm_id = fieldnorm_reader.fieldnorm_id(doc);
        let phrase_count = scorer.phrase_count();
        let mut explanation = Explanation::new("Phrase Scorer", scorer.score());
        if let Some(similarity_weight) = self.similarity_weight_opt.as_ref() {
            explanation.add_detail(similarity_weight.explain(fieldnorm_id, phrase_count));
        }
        Ok(explanation)
    }

    fn for_each_pruning(
        &self,
        mut threshold: Score,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score) -> Score,
    ) -> crate::Result<()> {
        let Some(bm25_weight) = self.similarity_weight_opt.as_ref() else {
            let mut scorer = self.scorer(reader, 1.0)?;
            crate::query::weight::for_each_pruning_scorer(scorer.as_mut(), threshold, callback);
            return Ok(());
        };
        if self.slop > 0 || self.phrase_terms.len() < 2 {
            let mut scorer = self.scorer(reader, 1.0)?;
            crate::query::weight::for_each_pruning_scorer(scorer.as_mut(), threshold, callback);
            return Ok(());
        }
        if bm25_weight.max_score() <= threshold {
            return Ok(());
        }

        let fieldnorm_reader = self.fieldnorm_reader(reader)?;
        let max_offset = self
            .phrase_terms
            .iter()
            .map(|&(offset, _)| offset)
            .max()
            .unwrap_or(0);
        let mut postings_list: Vec<(u32, SegmentPostings)> =
            Vec::with_capacity(self.phrase_terms.len());
        for &(offset, ref term) in &self.phrase_terms {
            let Some(postings) = reader
                .inverted_index(term.field())?
                .read_postings(term, IndexRecordOption::WithFreqsAndPositions)?
            else {
                return Ok(());
            };
            postings_list.push(((max_offset - offset) as u32, postings));
        }
        // Sort by doc_freq ascending so the rarest term drives block windows and intersection.
        postings_list.sort_unstable_by_key(|(_, postings)| postings.doc_freq());

        let mut left_positions = Vec::with_capacity(32);
        let mut right_positions = Vec::with_capacity(32);

        let mut doc = postings_list[0].1.doc();
        while doc < TERMINATED {
            // Phase 1: Block-max pruning across all phrase terms.
            // For any exact phrase match in a doc, `phrase_count <= min_i term_freq_i(doc)` and
            // `fieldnorm(doc) >= block_min_fieldnorm_i`, so `score(doc) <= min_i block_max_score_i`.
            let mut window_end = TERMINATED;
            let mut min_block_max_score = Score::MAX;
            let mut exhausted = false;
            for (_, postings) in &mut postings_list {
                postings.block_cursor.seek_block(doc);
                if !postings.block_cursor.has_remaining_docs() {
                    exhausted = true;
                    break;
                }
                window_end =
                    window_end.min(postings.block_cursor.skip_reader().last_doc_in_block());
                let bms = postings
                    .block_cursor
                    .block_max_score(&fieldnorm_reader, bm25_weight);
                if bms < min_block_max_score {
                    min_block_max_score = bms;
                }
            }
            if exhausted {
                return Ok(());
            }
            if min_block_max_score <= threshold {
                if window_end == TERMINATED {
                    return Ok(());
                }
                doc = window_end + 1;
                continue;
            }

            // Phase 2: Intersect and verify positions within `[doc, window_end]`.
            let (leader, secondaries) = postings_list.split_first_mut().unwrap();
            let mut candidate = leader.1.seek_after_shallow(doc);
            'window: while candidate <= window_end && candidate < TERMINATED {
                // Early per-doc BM25 upper bound from leader's tf before seeking secondaries.
                let fieldnorm_id = fieldnorm_reader.fieldnorm_id(candidate);
                let mut min_tf = leader.1.term_freq();
                if bm25_weight.score(fieldnorm_id, min_tf) <= threshold {
                    candidate = leader.1.advance();
                    continue;
                }

                // Intersect with all secondary terms.
                for (_, sec_postings) in secondaries.iter_mut() {
                    let sec_doc = sec_postings.seek_after_shallow(candidate);
                    if sec_doc != candidate {
                        if sec_doc > window_end || sec_doc == TERMINATED {
                            break 'window;
                        }
                        candidate = leader.1.seek(sec_doc);
                        continue 'window;
                    }
                    let sec_tf = sec_postings.term_freq();
                    if sec_tf < min_tf {
                        min_tf = sec_tf;
                        if bm25_weight.score(fieldnorm_id, min_tf) <= threshold {
                            candidate = leader.1.advance();
                            continue 'window;
                        }
                    }
                }

                // All terms match on `candidate` and upper bound exceeds `threshold`: check positions.
                let phrase_count = if let Some(pos0) =
                    leader.1.single_position_with_offset(leader.0)
                {
                    let mut matched = true;
                    for (offset, sec_postings) in secondaries.iter_mut() {
                        if let Some(sec_pos) = sec_postings.single_position_with_offset(*offset) {
                            if sec_pos != pos0 {
                                matched = false;
                                break;
                            }
                        } else {
                            sec_postings.positions_with_offset(*offset, &mut right_positions);
                            if right_positions.binary_search(&pos0).is_err() {
                                matched = false;
                                break;
                            }
                        }
                    }
                    u32::from(matched)
                } else {
                    leader
                        .1
                        .positions_with_offset(leader.0, &mut left_positions);
                    let num_sec = secondaries.len();
                    let mut empty = false;
                    for (offset, sec_postings) in &mut secondaries[..num_sec - 1] {
                        sec_postings.positions_with_offset(*offset, &mut right_positions);
                        super::phrase_scorer::intersect_positions(
                            &mut left_positions,
                            &right_positions,
                        );
                        if left_positions.is_empty() {
                            empty = true;
                            break;
                        }
                    }
                    if empty {
                        0
                    } else {
                        let (last_offset, last_postings) = &mut secondaries[num_sec - 1];
                        last_postings.positions_with_offset(*last_offset, &mut right_positions);
                        super::phrase_scorer::intersection_count(&left_positions, &right_positions)
                            as u32
                    }
                };
                if phrase_count > 0 {
                    let score = bm25_weight.score(fieldnorm_id, phrase_count);
                    if score > threshold {
                        threshold = callback(candidate, score);
                        if bm25_weight.max_score() <= threshold {
                            return Ok(());
                        }
                        if min_block_max_score <= threshold {
                            break 'window;
                        }
                    }
                }
                candidate = leader.1.advance();
            }

            if window_end == TERMINATED {
                return Ok(());
            }
            doc = window_end + 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::create_index;
    use crate::docset::{SeekDangerResult, TERMINATED};
    use crate::query::{EnableScoring, PhraseQuery, Weight};
    use crate::{DocSet, Term};

    #[test]
    pub fn test_phrase_count() -> crate::Result<()> {
        let index = create_index(&["a c", "a a b d a b c", " a b"])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();
        let phrase_query = PhraseQuery::new(vec![
            Term::from_field_text(text_field, "a"),
            Term::from_field_text(text_field, "b"),
        ]);
        let enable_scoring = EnableScoring::enabled_from_searcher(&searcher);
        let phrase_weight = phrase_query.phrase_weight(enable_scoring).unwrap();
        let mut phrase_scorer = phrase_weight
            .phrase_scorer(searcher.segment_reader(0u32), 1.0)?
            .unwrap();
        assert_eq!(phrase_scorer.doc(), 1);
        assert_eq!(phrase_scorer.phrase_count(), 2);
        assert_eq!(phrase_scorer.advance(), 2);
        assert_eq!(phrase_scorer.doc(), 2);
        assert_eq!(phrase_scorer.phrase_count(), 1);
        assert_eq!(phrase_scorer.advance(), TERMINATED);
        Ok(())
    }

    #[test]
    fn test_phrase_weight_scorer_danger() -> crate::Result<()> {
        let index = create_index(&["a b", "a c b", "a b", "a c"])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();
        let phrase_query = PhraseQuery::new(vec![
            Term::from_field_text(text_field, "a"),
            Term::from_field_text(text_field, "b"),
        ]);
        let phrase_weight =
            phrase_query.phrase_weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let reader = searcher.segment_reader(0);

        let (seek_result, mut scorer) = phrase_weight.scorer_danger(reader, 1, 1.0)?;
        assert_eq!(seek_result, SeekDangerResult::SeekLowerBound(2));
        assert_eq!(scorer.seek_danger(2), SeekDangerResult::Found);
        assert_eq!(scorer.doc(), 2);

        let (seek_result, scorer) = phrase_weight.scorer_danger(reader, 2, 1.0)?;
        assert_eq!(seek_result, SeekDangerResult::Found);
        assert_eq!(scorer.doc(), 2);

        let (seek_result, _) = phrase_weight.scorer_danger(reader, 3, 1.0)?;
        assert_eq!(seek_result, SeekDangerResult::SeekLowerBound(TERMINATED));

        let scoring_phrase_weight =
            phrase_query.phrase_weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let (seek_result, mut danger_scorer) =
            scoring_phrase_weight.scorer_danger(reader, 2, 2.0)?;
        assert_eq!(seek_result, SeekDangerResult::Found);
        let mut regular_scorer = scoring_phrase_weight.scorer(reader, 2.0)?;
        assert_eq!(regular_scorer.seek(2), 2);
        assert_eq!(danger_scorer.score(), regular_scorer.score());
        Ok(())
    }

    #[test]
    fn test_phrase_weight_for_each_pruning_matches_unpruned() -> crate::Result<()> {
        use crate::collector::TopDocs;

        let mut docs = Vec::with_capacity(400);
        for i in 0..400 {
            let text = match i % 11 {
                0 => "place order place order place order",
                3 => "place order place order",
                5 => "place order",
                7 => "place something else order",
                _ => "unrelated log entry",
            };
            docs.push(text);
        }
        let index = create_index(&docs)?;
        let text_field = index.schema().get_field("text").unwrap();
        let searcher = index.reader()?.searcher();
        let seg = searcher.segment_reader(0);
        let phrase_query = PhraseQuery::new(vec![
            Term::from_field_text(text_field, "place"),
            Term::from_field_text(text_field, "order"),
        ]);
        let weight = phrase_query.phrase_weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let mut exact = Vec::new();
        weight.for_each(seg, &mut |doc, score| exact.push((score, doc)))?;
        exact.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        exact.truncate(15);

        let top_docs =
            searcher.search(&phrase_query, &TopDocs::with_limit(15).order_by_score())?;
        assert_eq!(top_docs.len(), exact.len());
        for ((got_score, _), (exp_score, _)) in top_docs.iter().zip(exact.iter()) {
            assert!((got_score - exp_score).abs() < 1e-5);
        }
        Ok(())
    }
}
