use std::io;
use std::sync::Arc;

use common::BitSet;
use tantivy_fst::Automaton;

use super::phrase_prefix_query::prefix_end;
use crate::index::SegmentReader;
use crate::postings::TermInfo;
use crate::query::{BitSetDocSet, ConstScorer, Explanation, Scorer, Weight};
use crate::schema::{Field, IndexRecordOption};
use crate::termdict::{TermDictionary, TermStreamer, WithoutKeys};
use crate::{DocId, DocSet, Score, TantivyError};

/// Optional callback providing pre-resolved matching [`TermInfo`] entries for a segment
/// (for example, populated during async warmup to avoid scanning the term dictionary twice).
pub type SegmentTermInfosLookup =
    Arc<dyn Fn(crate::index::SegmentId) -> Option<Arc<[TermInfo]>> + Send + Sync>;

/// A weight struct for Fuzzy Term and Regex Queries
pub struct AutomatonWeight<A> {
    field: Field,
    automaton: Arc<A>,
    // For JSON fields, the term dictionary include terms from all paths.
    // We apply additional filtering based on the given JSON path, when searching within the term
    // dictionary. This prevents terms from unrelated paths from matching the search criteria.
    json_path_bytes: Option<Box<[u8]>>,
    term_infos_lookup: Option<SegmentTermInfosLookup>,
}

impl<A> AutomatonWeight<A>
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    /// Create a new AutomationWeight
    pub fn new<IntoArcA: Into<Arc<A>>>(field: Field, automaton: IntoArcA) -> AutomatonWeight<A> {
        AutomatonWeight {
            field,
            automaton: automaton.into(),
            json_path_bytes: None,
            term_infos_lookup: None,
        }
    }

    /// Create a new AutomationWeight for a json path
    pub fn new_for_json_path<IntoArcA: Into<Arc<A>>>(
        field: Field,
        automaton: IntoArcA,
        json_path_bytes: &[u8],
    ) -> AutomatonWeight<A> {
        AutomatonWeight {
            field,
            automaton: automaton.into(),
            json_path_bytes: Some(json_path_bytes.to_vec().into_boxed_slice()),
            term_infos_lookup: None,
        }
    }

    /// Attach a pre-resolved per-segment [`TermInfo`] lookup to avoid re-scanning the term
    /// dictionary when warmup already enumerated the matching terms.
    pub fn with_term_infos_lookup(mut self, lookup: SegmentTermInfosLookup) -> Self {
        self.term_infos_lookup = Some(lookup);
        self
    }

    fn cached_term_infos(&self, reader: &SegmentReader) -> Option<Arc<[TermInfo]>> {
        self.term_infos_lookup
            .as_ref()
            .and_then(|lookup| lookup(reader.segment_id()))
    }

    fn automaton_stream<'a>(
        &'a self,
        term_dict: &'a TermDictionary,
    ) -> io::Result<TermStreamer<'a, &'a A, WithoutKeys>> {
        let automaton: &A = &self.automaton;
        // Matching terms are only used for their `TermInfo`.
        let mut term_stream_builder = term_dict.search(automaton).without_keys();

        if let Some(json_path_bytes) = &self.json_path_bytes {
            term_stream_builder = term_stream_builder.ge(json_path_bytes);
            if let Some(end) = prefix_end(json_path_bytes) {
                term_stream_builder = term_stream_builder.lt(&end);
            }
        }

        term_stream_builder.into_stream()
    }

    /// Returns the term infos that match the automaton
    pub fn get_match_term_infos(&self, reader: &SegmentReader) -> crate::Result<Vec<TermInfo>> {
        if let Some(cached) = self.cached_term_infos(reader) {
            return Ok(cached.to_vec());
        }
        let inverted_index = reader.inverted_index(self.field)?;
        let term_dict = inverted_index.terms();
        let mut term_stream = self.automaton_stream(term_dict)?;
        let mut term_infos = Vec::new();
        while term_stream.advance() {
            term_infos.push(term_stream.value().clone());
        }
        Ok(term_infos)
    }
}

fn fill_bitset_from_term_info(
    inverted_index: &crate::InvertedIndexReader,
    term_info: &TermInfo,
    doc_bitset: &mut BitSet,
) -> crate::Result<()> {
    let mut block_segment_postings =
        inverted_index.read_block_postings_from_terminfo(term_info, IndexRecordOption::Basic)?;
    loop {
        let docs = block_segment_postings.docs();
        if docs.is_empty() {
            break;
        }
        for &doc in docs {
            doc_bitset.insert(doc);
        }
        block_segment_postings.advance();
    }
    Ok(())
}

impl<A> Weight for AutomatonWeight<A>
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let max_doc = reader.max_doc();
        let inverted_index = reader.inverted_index(self.field)?;
        if let Some(cached) = self.cached_term_infos(reader) {
            if cached.is_empty() {
                return Ok(Box::new(crate::query::EmptyScorer));
            }
            if cached.len() == 1 {
                let postings = inverted_index
                    .read_postings_from_terminfo(&cached[0], IndexRecordOption::Basic)?;
                return Ok(Box::new(ConstScorer::new(postings, boost)));
            }
            let mut doc_bitset = BitSet::with_max_value(max_doc);
            for term_info in cached.iter() {
                fill_bitset_from_term_info(&inverted_index, term_info, &mut doc_bitset)?;
            }
            return Ok(Box::new(ConstScorer::new(
                BitSetDocSet::from(doc_bitset),
                boost,
            )));
        }
        let mut doc_bitset = BitSet::with_max_value(max_doc);
        let term_dict = inverted_index.terms();
        let mut term_stream = self.automaton_stream(term_dict)?;
        while term_stream.advance() {
            fill_bitset_from_term_info(&inverted_index, term_stream.value(), &mut doc_bitset)?;
        }
        let doc_bitset = BitSetDocSet::from(doc_bitset);
        let const_scorer = ConstScorer::new(doc_bitset, boost);
        Ok(Box::new(const_scorer))
    }

    fn count(&self, reader: &SegmentReader) -> crate::Result<u32> {
        if let Some(alive_bitset) = reader.alive_bitset() {
            return Ok(self.scorer(reader, 1.0)?.count(alive_bitset));
        }
        let inverted_index = reader.inverted_index(self.field)?;
        if let Some(cached) = self.cached_term_infos(reader) {
            return match cached.as_ref() {
                [] => Ok(0),
                [single] => Ok(single.doc_freq),
                many => {
                    let mut doc_bitset = BitSet::with_max_value(reader.max_doc());
                    for term_info in many {
                        fill_bitset_from_term_info(&inverted_index, term_info, &mut doc_bitset)?;
                    }
                    Ok(doc_bitset.len() as u32)
                }
            };
        }
        let term_dict = inverted_index.terms();
        let mut term_stream = self.automaton_stream(term_dict)?;
        let Some(first_term_info) = term_stream.next_without_key().cloned() else {
            return Ok(0);
        };
        let Some(second_term_info) = term_stream.next_without_key().cloned() else {
            // Single matching term with no deletes: doc_freq is exact in O(1).
            return Ok(first_term_info.doc_freq);
        };

        let mut doc_bitset = BitSet::with_max_value(reader.max_doc());
        fill_bitset_from_term_info(&inverted_index, &first_term_info, &mut doc_bitset)?;
        fill_bitset_from_term_info(&inverted_index, &second_term_info, &mut doc_bitset)?;
        while term_stream.advance() {
            fill_bitset_from_term_info(&inverted_index, term_stream.value(), &mut doc_bitset)?;
        }
        Ok(doc_bitset.len() as u32)
    }

    fn for_each_pruning(
        &self,
        mut threshold: Score,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score) -> Score,
    ) -> crate::Result<()> {
        const CONST_SCORE: Score = 1.0;
        if CONST_SCORE <= threshold {
            return Ok(());
        }
        let inverted_index = reader.inverted_index(self.field)?;
        const STREAMING_TERM_CAP: usize = 256;

        let cached_opt = self.cached_term_infos(reader);
        let mut owned_term_infos = Vec::new();
        let mut term_stream_opt = None;
        let (term_infos_slice, exceeded_cap): (&[TermInfo], bool) =
            if let Some(cached) = cached_opt.as_ref() {
                if cached.is_empty() {
                    return Ok(());
                }
                (cached.as_ref(), cached.len() > STREAMING_TERM_CAP)
            } else {
                let term_dict = inverted_index.terms();
                let mut term_stream = self.automaton_stream(term_dict)?;
                let mut exceeded = false;
                while term_stream.advance() {
                    if owned_term_infos.len() == STREAMING_TERM_CAP {
                        exceeded = true;
                        break;
                    }
                    owned_term_infos.push(term_stream.value().clone());
                }
                if owned_term_infos.is_empty() {
                    return Ok(());
                }
                if exceeded {
                    term_stream_opt = Some(term_stream);
                }
                (owned_term_infos.as_slice(), exceeded)
            };

        if !exceeded_cap {
            if term_infos_slice.len() == 1 {
                let mut postings = inverted_index.read_block_postings_from_terminfo(
                    &term_infos_slice[0],
                    IndexRecordOption::Basic,
                )?;
                loop {
                    let docs = postings.docs();
                    if docs.is_empty() {
                        break;
                    }
                    for &doc in docs {
                        threshold = callback(doc, CONST_SCORE);
                        if CONST_SCORE <= threshold {
                            return Ok(());
                        }
                    }
                    postings.advance();
                }
                return Ok(());
            }

            use std::cmp::Reverse;
            use std::collections::BinaryHeap;

            let mut cursors: Vec<(crate::postings::BlockSegmentPostings, usize)> =
                Vec::with_capacity(term_infos_slice.len());
            let mut heap: BinaryHeap<Reverse<(DocId, usize)>> =
                BinaryHeap::with_capacity(term_infos_slice.len());
            for term_info in term_infos_slice {
                let postings = inverted_index
                    .read_block_postings_from_terminfo(term_info, IndexRecordOption::Basic)?;
                if let Some(&first_doc) = postings.docs().first() {
                    let idx = cursors.len();
                    cursors.push((postings, 0));
                    heap.push(Reverse((first_doc, idx)));
                }
            }
            let mut last_doc: Option<DocId> = None;
            while let Some(Reverse((doc, idx))) = heap.pop() {
                let (postings, pos) = &mut cursors[idx];
                *pos += 1;
                if *pos >= postings.docs().len() {
                    postings.advance();
                    *pos = 0;
                }
                if let Some(&next_doc) = postings.docs().get(*pos) {
                    heap.push(Reverse((next_doc, idx)));
                }
                if last_doc == Some(doc) {
                    continue;
                }
                last_doc = Some(doc);
                threshold = callback(doc, CONST_SCORE);
                if CONST_SCORE <= threshold {
                    return Ok(());
                }
            }
            return Ok(());
        }

        // Many matching terms: fall back to the BitSet scorer, still stopping as soon as
        // `threshold >= CONST_SCORE`.
        let mut doc_bitset = BitSet::with_max_value(reader.max_doc());
        for term_info in term_infos_slice {
            fill_bitset_from_term_info(&inverted_index, term_info, &mut doc_bitset)?;
        }
        if let Some(mut term_stream) = term_stream_opt {
            loop {
                fill_bitset_from_term_info(&inverted_index, term_stream.value(), &mut doc_bitset)?;
                if !term_stream.advance() {
                    break;
                }
            }
        }
        let mut doc_bitset = BitSetDocSet::from(doc_bitset);
        let mut doc = doc_bitset.doc();
        while doc != crate::TERMINATED {
            threshold = callback(doc, CONST_SCORE);
            if CONST_SCORE <= threshold {
                return Ok(());
            }
            doc = doc_bitset.advance();
        }
        Ok(())
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) == doc {
            Ok(Explanation::new("AutomatonScorer", 1.0))
        } else {
            Err(TantivyError::InvalidArgument(
                "Document does not exist".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use tantivy_fst::Automaton;

    use super::AutomatonWeight;
    use crate::docset::TERMINATED;
    use crate::query::Weight;
    use crate::schema::{Schema, STRING};
    use crate::{Index, IndexWriter};

    fn create_index() -> crate::Result<Index> {
        let mut schema = Schema::builder();
        let title = schema.add_text_field("title", STRING);
        let index = Index::create_in_ram(schema.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(title=>"abc"))?;
        index_writer.add_document(doc!(title=>"bcd"))?;
        index_writer.add_document(doc!(title=>"abcd"))?;
        index_writer.commit()?;
        Ok(index)
    }

    #[derive(Clone, Copy)]
    enum State {
        Start,
        NotMatching,
        AfterA,
    }

    struct PrefixedByA;

    impl Automaton for PrefixedByA {
        type State = State;

        fn start(&self) -> Self::State {
            State::Start
        }

        fn is_match(&self, state: &Self::State) -> bool {
            matches!(*state, State::AfterA)
        }

        fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
            match *state {
                State::Start => {
                    if byte == b'a' {
                        State::AfterA
                    } else {
                        State::NotMatching
                    }
                }
                State::AfterA => State::AfterA,
                State::NotMatching => State::NotMatching,
            }
        }
    }

    #[test]
    fn test_automaton_weight() -> crate::Result<()> {
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let automaton_weight = AutomatonWeight::new(field, PrefixedByA);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        assert_eq!(scorer.doc(), 0u32);
        assert_eq!(scorer.score(), 1.0);
        assert_eq!(scorer.advance(), 2u32);
        assert_eq!(scorer.doc(), 2u32);
        assert_eq!(scorer.score(), 1.0);
        assert_eq!(scorer.advance(), TERMINATED);
        Ok(())
    }

    #[test]
    fn test_automaton_weight_boost() -> crate::Result<()> {
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let automaton_weight = AutomatonWeight::new(field, PrefixedByA);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.32)?;
        assert_eq!(scorer.doc(), 0u32);
        assert_eq!(scorer.score(), 1.32);
        Ok(())
    }

    #[test]
    fn test_automaton_weight_count_pruning_and_cached_term_infos() -> crate::Result<()> {
        let mut schema = Schema::builder();
        let title = schema.add_text_field("title", STRING);
        let index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_for_tests()?;
        for i in 0..40 {
            let term = match i % 4 {
                0 => "alpha",
                1 => "apricot",
                2 => "banana",
                _ => "avocado",
            };
            writer.add_document(doc!(title => term))?;
        }
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let seg_reader = searcher.segment_reader(0);
        let weight = AutomatonWeight::new(title, PrefixedByA);
        assert_eq!(weight.count(seg_reader)?, 30);

        // Verify for_each_pruning stops early once threshold reaches 1.0.
        let mut collected = Vec::new();
        weight.for_each_pruning(0.0, seg_reader, &mut |doc, score| {
            collected.push((doc, score));
            if collected.len() >= 5 {
                score
            } else {
                0.0
            }
        })?;
        assert_eq!(collected.len(), 5);
        assert!(collected.windows(2).all(|w| w[0].0 < w[1].0));

        // Verify pre-resolved cached term infos produce identical count and pruning hits.
        let term_infos: std::sync::Arc<[crate::postings::TermInfo]> =
            std::sync::Arc::from(weight.get_match_term_infos(seg_reader)?);
        let seg_id = seg_reader.segment_id();
        let cached_weight = AutomatonWeight::new(title, PrefixedByA).with_term_infos_lookup(
            std::sync::Arc::new(move |id| (id == seg_id).then(|| term_infos.clone())),
        );
        assert_eq!(cached_weight.count(seg_reader)?, 30);
        let mut cached_collected = Vec::new();
        cached_weight.for_each_pruning(0.0, seg_reader, &mut |doc, score| {
            cached_collected.push((doc, score));
            if cached_collected.len() >= 5 {
                score
            } else {
                0.0
            }
        })?;
        assert_eq!(cached_collected, collected);
        Ok(())
    }
}
