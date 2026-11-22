//! Breadth-first two-pass segment terms collector for top-level terms aggregations with
//! sub-aggregations ordered by `_count` or `_key`.
//!
//! Pass 1 (`collect`) counts into a compact `TermMap<Slot = ()>` while recording matched doc IDs
//! (zero-allocation `[start, end)` range for contiguous streams). Pass 2 (`flush`) prunes to the
//! top `segment_size` winners and replays only matching documents into the child collector.

use rustc_hash::FxHashMap;

use super::{
    materialize_term_buckets, select_and_cut_off_count_or_key, Bucket, TermAggregationMap,
    TermsAggReqData, MAX_NUM_TERMS_FOR_VEC,
};
use crate::aggregation::agg_data::AggregationsSegmentCtx;
use crate::aggregation::agg_limits::AggregationLimitsGuard;
use crate::aggregation::buffered_sub_aggs::{
    BufferedSubAggs, HighCardBufferedSubAggs, LowCardBufferedSubAggs, SubAggBuffer,
};
use crate::aggregation::intermediate_agg_result::{
    IntermediateAggregationResult, IntermediateAggregationResults,
};
use crate::aggregation::segment_agg_result::{BucketIdProvider, SegmentAggregationCollector};
use crate::aggregation::BucketId;
use crate::DocId;

/// Block size used when replaying matched documents in Pass 2.
pub(super) const PASS2_CHUNK_SIZE: usize = 256;

/// Maximum number of kept buckets for which Pass 2 groups docs into per-bucket vectors via
/// [`LowCardBufferedSubAggs`].
const LOW_CARD_DEFERRED_SUBAGG_THRESHOLD: usize = 256;

/// Larger batch flush threshold for Pass 2 [`LowCardBufferedSubAggs`] so each kept bucket
/// receives longer contiguous document slices (fewer redundant fast-field block decodes and
/// tighter single-bucket metric/terms inner loops while staying within ~32 KiB).
const DEFERRED_LOW_CARD_FLUSH_THRESHOLD: usize = 8192;

/// Sentinel value in [`KeptKeyLookup::Dense`] indicating that a term key did not survive cutoff.
const NOT_KEPT: BucketId = u32::MAX;

/// Number of `u64` words in the 2048-bit (256-byte, L1-resident) hash prefilter.
const FILTER_WORDS: usize = 32;
const FX_MIX: u64 = 0x517c_c1b7_2722_0a95;

#[inline(always)]
pub(super) fn hash_filter_bit(key: u64) -> (usize, u64) {
    let mixed = key.wrapping_mul(FX_MIX);
    let bit_idx = (mixed >> 53) as usize; // 11 bits -> 0..2048
    (bit_idx >> 6, 1u64 << (bit_idx & 63))
}

/// Compact representation of matched documents collected during Pass 1.
///
/// Hidden contract: callers must submit strictly increasing document IDs across all calls to
/// [`Self::push_docs`] (the standard [`SegmentAggregationCollector::collect`] contract).
#[derive(Debug, Default)]
enum MatchedDocs {
    #[default]
    Empty,
    /// Contiguous half-open doc ID range `[start, end)`.
    Contiguous { start: DocId, end: DocId },
    /// General strictly increasing doc ID list.
    List(Vec<DocId>),
}

impl MatchedDocs {
    #[inline]
    fn is_contiguous_slice(docs: &[DocId]) -> bool {
        debug_assert!(!docs.is_empty());
        docs[docs.len() - 1] - docs[0] == (docs.len() - 1) as u32
    }

    fn push_docs(
        &mut self,
        docs: &[DocId],
        limits: &mut AggregationLimitsGuard,
    ) -> crate::Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        match self {
            MatchedDocs::Empty => {
                if Self::is_contiguous_slice(docs) {
                    let start = docs[0];
                    *self = MatchedDocs::Contiguous {
                        start,
                        end: start + docs.len() as u32,
                    };
                } else {
                    let list = docs.to_vec();
                    let mem_bytes = list.capacity() * std::mem::size_of::<DocId>();
                    limits.add_memory_consumed(mem_bytes as u64)?;
                    *self = MatchedDocs::List(list);
                }
            }
            MatchedDocs::Contiguous { start, end } => {
                if docs[0] == *end && Self::is_contiguous_slice(docs) {
                    *end += docs.len() as u32;
                } else {
                    let existing_len = (*end - *start) as usize;
                    let mut list = Vec::with_capacity(existing_len + docs.len());
                    list.extend(*start..*end);
                    list.extend_from_slice(docs);
                    let mem_bytes = list.capacity() * std::mem::size_of::<DocId>();
                    limits.add_memory_consumed(mem_bytes as u64)?;
                    *self = MatchedDocs::List(list);
                }
            }
            MatchedDocs::List(list) => {
                let prev_cap = list.capacity();
                list.extend_from_slice(docs);
                let new_cap = list.capacity();
                if new_cap > prev_cap {
                    let delta_bytes = (new_cap - prev_cap) * std::mem::size_of::<DocId>();
                    limits.add_memory_consumed(delta_bytes as u64)?;
                }
            }
        }
        Ok(())
    }

    fn for_each_chunk(
        &self,
        mut callback: impl FnMut(&[DocId]) -> crate::Result<()>,
    ) -> crate::Result<()> {
        match self {
            MatchedDocs::Empty => Ok(()),
            MatchedDocs::Contiguous { start, end } => {
                let mut chunk_buf = [0u32; PASS2_CHUNK_SIZE];
                let mut cursor = *start;
                while cursor < *end {
                    let remaining = (*end - cursor) as usize;
                    let chunk_len = remaining.min(PASS2_CHUNK_SIZE);
                    for (offset, slot) in chunk_buf[..chunk_len].iter_mut().enumerate() {
                        *slot = cursor + offset as u32;
                    }
                    callback(&chunk_buf[..chunk_len])?;
                    cursor += chunk_len as u32;
                }
                Ok(())
            }
            MatchedDocs::List(list) => {
                for chunk in list.chunks(PASS2_CHUNK_SIZE) {
                    callback(chunk)?;
                }
                Ok(())
            }
        }
    }
}

/// Fast lookup structure mapping surviving raw term keys to their compact `0..kept` [`BucketId`].
enum KeptKeyLookup {
    Empty,
    Single {
        key: u64,
        bucket_id: BucketId,
    },
    Dense {
        offset: u64,
        table: Vec<BucketId>,
    },
    HashFiltered {
        filter: [u64; FILTER_WORDS],
        map: FxHashMap<u64, BucketId>,
    },
    Hash(FxHashMap<u64, BucketId>),
}

impl KeptKeyLookup {
    fn new(entries: &[(u64, Bucket<BucketId>)], min_term_id: u64, max_term_id: u64) -> Self {
        if entries.is_empty() {
            return KeptKeyLookup::Empty;
        }
        if entries.len() == 1 {
            return KeptKeyLookup::Single {
                key: entries[0].0,
                bucket_id: entries[0].1.bucket_id,
            };
        }
        let span = max_term_id.saturating_sub(min_term_id);
        if span < MAX_NUM_TERMS_FOR_VEC {
            let mut table = vec![NOT_KEPT; (span + 1) as usize];
            for &(key, bucket) in entries {
                if let Some(idx) = key.checked_sub(min_term_id) {
                    if let Some(slot) = table.get_mut(idx as usize) {
                        *slot = bucket.bucket_id;
                    }
                }
            }
            return KeptKeyLookup::Dense {
                offset: min_term_id,
                table,
            };
        }
        let mut map = FxHashMap::with_capacity_and_hasher(entries.len(), Default::default());
        if entries.len() <= 512 {
            let mut filter = [0u64; FILTER_WORDS];
            for &(key, bucket) in entries {
                let (word, mask) = hash_filter_bit(key);
                filter[word] |= mask;
                map.insert(key, bucket.bucket_id);
            }
            return KeptKeyLookup::HashFiltered { filter, map };
        }
        for &(key, bucket) in entries {
            map.insert(key, bucket.bucket_id);
        }
        KeptKeyLookup::Hash(map)
    }
}

#[derive(Debug, Default)]
pub(super) struct FinalizedDeferredBuckets {
    pub(super) entries: Vec<(u64, Bucket<BucketId>)>,
    pub(super) term_doc_count_before_cutoff: u64,
    pub(super) sum_other_doc_count: u64,
}

/// Top-level segment terms collector that defers sub-aggregation execution until `flush`, after
/// pruning buckets to the top `segment_size` winners.
#[derive(Debug)]
pub(super) struct DeferredSubAggTermCollector<TermMap: TermAggregationMap<Slot = ()>> {
    term_buckets: Option<TermMap>,
    matched_docs: MatchedDocs,
    sub_agg_collector: Option<Box<dyn SegmentAggregationCollector>>,
    finalized: Option<FinalizedDeferredBuckets>,
    bucket_id_provider: BucketIdProvider,
    min_term_id: u64,
    max_term_id: u64,
    terms_req_data: TermsAggReqData,
}

impl<TermMap: TermAggregationMap<Slot = ()>> DeferredSubAggTermCollector<TermMap> {
    pub(super) fn new(
        term_buckets: TermMap,
        sub_agg_collector: Box<dyn SegmentAggregationCollector>,
        bucket_id_provider: BucketIdProvider,
        min_term_id: u64,
        max_term_id: u64,
        terms_req_data: TermsAggReqData,
    ) -> Self {
        Self {
            term_buckets: Some(term_buckets),
            matched_docs: MatchedDocs::Empty,
            sub_agg_collector: Some(sub_agg_collector),
            finalized: None,
            bucket_id_provider,
            min_term_id,
            max_term_id,
            terms_req_data,
        }
    }

    fn replay_into_sub_agg<B: SubAggBuffer>(
        matched_docs: &MatchedDocs,
        terms_req_data: &mut TermsAggReqData,
        kept_lookup: &KeptKeyLookup,
        buffered: &mut BufferedSubAggs<B>,
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        matched_docs.for_each_chunk(|chunk| {
            agg_data
                .column_block_accessor
                .fetch_block_with_missing_unique_per_doc(
                    chunk,
                    &mut *terms_req_data.accessor,
                    terms_req_data.missing_value_for_accessor,
                    true,
                );
            match kept_lookup {
                KeptKeyLookup::Empty => {}
                KeptKeyLookup::Single { key, bucket_id } => {
                    let target_key = *key;
                    let target_bucket = *bucket_id;
                    for (doc, term_id) in agg_data.column_block_accessor.iter_docid_vals(chunk) {
                        if term_id == target_key {
                            buffered.push(target_bucket, doc);
                        }
                    }
                }
                KeptKeyLookup::Dense { offset, table } => {
                    let base = *offset;
                    for (doc, term_id) in agg_data.column_block_accessor.iter_docid_vals(chunk) {
                        if let Some(idx) = term_id.checked_sub(base) {
                            if let Some(&bucket_id) = table.get(idx as usize) {
                                if bucket_id != NOT_KEPT {
                                    buffered.push(bucket_id, doc);
                                }
                            }
                        }
                    }
                }
                KeptKeyLookup::HashFiltered { filter, map } => {
                    for (doc, term_id) in agg_data.column_block_accessor.iter_docid_vals(chunk) {
                        let (word, mask) = hash_filter_bit(term_id);
                        if (filter[word] & mask) != 0 {
                            if let Some(&bucket_id) = map.get(&term_id) {
                                buffered.push(bucket_id, doc);
                            }
                        }
                    }
                }
                KeptKeyLookup::Hash(map) => {
                    for (doc, term_id) in agg_data.column_block_accessor.iter_docid_vals(chunk) {
                        if let Some(&bucket_id) = map.get(&term_id) {
                            buffered.push(bucket_id, doc);
                        }
                    }
                }
            }
            buffered.check_flush_local(agg_data)
        })?;
        buffered.flush(agg_data)
    }
}

impl<TermMap: TermAggregationMap<Slot = ()>> SegmentAggregationCollector
    for DeferredSubAggTermCollector<TermMap>
{
    fn add_intermediate_aggregation_result(
        &mut self,
        agg_data: &AggregationsSegmentCtx,
        results: &mut IntermediateAggregationResults,
        parent_bucket_id: BucketId,
    ) -> crate::Result<()> {
        debug_assert_eq!(parent_bucket_id, 0);
        let finalized = self.finalized.take().unwrap_or_default();
        let name = self.terms_req_data.name.clone();
        let sub_agg_ref: Option<&mut dyn SegmentAggregationCollector> = self
            .sub_agg_collector
            .as_mut()
            .map(|boxed| boxed.as_mut() as &mut dyn SegmentAggregationCollector);
        let bucket = materialize_term_buckets(
            &self.terms_req_data,
            sub_agg_ref,
            finalized.entries,
            finalized.term_doc_count_before_cutoff,
            finalized.sum_other_doc_count,
            agg_data,
        )?;
        results.push(name, IntermediateAggregationResult::Bucket(bucket))?;
        Ok(())
    }

    #[inline]
    fn collect(
        &mut self,
        parent_bucket_id: BucketId,
        docs: &[DocId],
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        debug_assert_eq!(parent_bucket_id, 0);
        if docs.is_empty() {
            return Ok(());
        }
        self.matched_docs
            .push_docs(docs, &mut agg_data.context.limits)?;

        let term_buckets = self
            .term_buckets
            .as_mut()
            .expect("DeferredSubAggTermCollector::collect called after flush");
        let mem_pre = term_buckets.get_memory_consumption();

        agg_data
            .column_block_accessor
            .fetch_block_with_missing_unique_per_doc(
                docs,
                &mut *self.terms_req_data.accessor,
                self.terms_req_data.missing_value_for_accessor,
                false,
            );

        if let Some(allowed_bs) = self.terms_req_data.allowed_term_ids.as_ref() {
            for term_id in agg_data.column_block_accessor.iter_vals() {
                if allowed_bs.contains(term_id as u32) {
                    term_buckets.term_entry(term_id, &mut self.bucket_id_provider);
                }
            }
        } else {
            for term_id in agg_data.column_block_accessor.iter_vals() {
                term_buckets.term_entry(term_id, &mut self.bucket_id_provider);
            }
        }

        let mem_delta = term_buckets.get_memory_consumption() - mem_pre;
        if mem_delta > 0 {
            agg_data
                .context
                .limits
                .add_memory_consumed(mem_delta as u64)?;
        }
        Ok(())
    }

    fn flush(&mut self, agg_data: &mut AggregationsSegmentCtx) -> crate::Result<()> {
        if self.finalized.is_some() {
            return Ok(());
        }
        let Some(term_buckets) = self.term_buckets.take() else {
            return Ok(());
        };

        let mut raw_entries = term_buckets.into_vec();
        let (term_doc_count_before_cutoff, sum_other_doc_count) = select_and_cut_off_count_or_key(
            &mut raw_entries,
            &self.terms_req_data,
            TermMap::SORTED_BY_KEY,
        );

        let entries: Vec<(u64, Bucket<BucketId>)> = raw_entries
            .into_iter()
            .enumerate()
            .map(|(idx, (key, bucket))| {
                (
                    key,
                    Bucket {
                        count: bucket.count,
                        bucket_id: idx as BucketId,
                    },
                )
            })
            .collect();

        if !entries.is_empty() {
            let kept_lookup = KeptKeyLookup::new(&entries, self.min_term_id, self.max_term_id);
            let mut sub_agg_collector = self
                .sub_agg_collector
                .take()
                .expect("DeferredSubAggTermCollector missing sub_agg_collector");
            sub_agg_collector.prepare_max_bucket((entries.len() - 1) as BucketId, agg_data)?;

            let sub_agg_collector = if entries.len() <= LOW_CARD_DEFERRED_SUBAGG_THRESHOLD {
                let mut buffered = LowCardBufferedSubAggs::with_flush_threshold(
                    sub_agg_collector,
                    DEFERRED_LOW_CARD_FLUSH_THRESHOLD,
                );
                Self::replay_into_sub_agg(
                    &self.matched_docs,
                    &mut self.terms_req_data,
                    &kept_lookup,
                    &mut buffered,
                    agg_data,
                )?;
                buffered.into_inner()
            } else {
                let mut buffered = HighCardBufferedSubAggs::new(sub_agg_collector);
                Self::replay_into_sub_agg(
                    &self.matched_docs,
                    &mut self.terms_req_data,
                    &kept_lookup,
                    &mut buffered,
                    agg_data,
                )?;
                buffered.into_inner()
            };
            self.sub_agg_collector = Some(sub_agg_collector);
        } else if let Some(sub_agg_collector) = &mut self.sub_agg_collector {
            sub_agg_collector.flush(agg_data)?;
        }

        // Release Pass-1 doc buffer immediately once Pass 2 completes.
        self.matched_docs = MatchedDocs::Empty;
        self.finalized = Some(FinalizedDeferredBuckets {
            entries,
            term_doc_count_before_cutoff,
            sum_other_doc_count,
        });
        Ok(())
    }

    fn prepare_max_bucket(
        &mut self,
        max_bucket: BucketId,
        _agg_data: &AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        debug_assert_eq!(max_bucket, 0);
        Ok(())
    }

    fn compute_metric_value(
        &self,
        _bucket_id: BucketId,
        _sub_agg_name: &str,
        _sub_agg_property: &str,
        _agg_data: &AggregationsSegmentCtx,
    ) -> Option<f64> {
        None
    }
}
