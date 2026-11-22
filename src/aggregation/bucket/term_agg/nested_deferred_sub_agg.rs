//! Breadth-first two-pass segment terms collector for nested (non-top-level) terms aggregations
//! with sub-aggregations ordered by `_count` or `_key`.
//!
//! When a nested `terms` aggregation over a high-cardinality field (such as `ClientIP` inside
//! `SearchEngineID`) has metric or bucket sub-aggregations, depth-first collection allocates and
//! populates sub-aggregation state for every `(parent_bucket, child_term)` pair seen in the
//! segment before pruning each parent to `segment_size`.
//!
//! [`NestedDeferredSubAggTermCollector`] counts each parent bucket into a compact
//! `HashMapTermBuckets<()>` while recording `(parent_bucket_id, doc)` submissions. At `flush`, it
//! prunes every parent bucket to its top `segment_size` winners, assigns compact child
//! `BucketId`s, and replays only matching documents directly into `sub_agg.collect_multiple`.

use rustc_hash::FxHashMap;

use super::deferred_sub_agg::{hash_filter_bit, FinalizedDeferredBuckets, PASS2_CHUNK_SIZE};
use super::{
    materialize_term_buckets, select_and_cut_off_count_or_key, Bucket, HashMapTermBuckets,
    TermAggregationMap, TermsAggReqData,
};
use crate::aggregation::agg_data::AggregationsSegmentCtx;
use crate::aggregation::intermediate_agg_result::{
    IntermediateAggregationResult, IntermediateAggregationResults,
};
use crate::aggregation::segment_agg_result::{
    collect_multiple_fallback, is_strictly_increasing, BucketIdProvider,
    SegmentAggregationCollector,
};
use crate::aggregation::BucketId;
use crate::DocId;

const NESTED_FILTER_WORDS: usize = 64; // 4096 bits = 512 bytes

/// Strictly-increasing batch emitter forwarding `(BucketId, DocId)` pairs directly to
/// `sub_agg.collect_multiple`.
struct DirectCollectMultipleEmitter {
    bucket_ids: Vec<BucketId>,
    docs: Vec<DocId>,
}

impl DirectCollectMultipleEmitter {
    fn new() -> Self {
        Self {
            bucket_ids: Vec::with_capacity(PASS2_CHUNK_SIZE),
            docs: Vec::with_capacity(PASS2_CHUNK_SIZE),
        }
    }

    #[inline]
    fn push(
        &mut self,
        bucket_id: BucketId,
        doc: DocId,
        sub_agg: &mut dyn SegmentAggregationCollector,
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        if self.docs.len() >= PASS2_CHUNK_SIZE
            || self.docs.last().is_some_and(|&prev_doc| prev_doc >= doc)
        {
            sub_agg.collect_multiple(&self.bucket_ids, &self.docs, agg_data)?;
            self.bucket_ids.clear();
            self.docs.clear();
        }
        self.bucket_ids.push(bucket_id);
        self.docs.push(doc);
        Ok(())
    }

    fn finish(
        &mut self,
        sub_agg: &mut dyn SegmentAggregationCollector,
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        if !self.docs.is_empty() {
            sub_agg.collect_multiple(&self.bucket_ids, &self.docs, agg_data)?;
            self.bucket_ids.clear();
            self.docs.clear();
        }
        sub_agg.flush(agg_data)
    }
}

#[inline(always)]
fn nested_filter_bit(parent_bucket_id: BucketId, term_id: u64) -> (usize, u64) {
    let combined = term_id ^ (u64::from(parent_bucket_id).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let (word32, mask) = hash_filter_bit(combined);
    let high_bit = ((combined >> 63) as usize) << 5;
    (word32 | high_bit, mask)
}

#[derive(Debug)]
pub(super) struct NestedDeferredSubAggTermCollector {
    parent_buckets: Vec<HashMapTermBuckets<()>>,
    matched_parents: Vec<BucketId>,
    matched_docs: Vec<DocId>,
    sub_agg_collector: Box<dyn SegmentAggregationCollector>,
    finalized: Option<Vec<FinalizedDeferredBuckets>>,
    bucket_id_provider: BucketIdProvider,
    terms_req_data: TermsAggReqData,
}

impl NestedDeferredSubAggTermCollector {
    pub(super) fn new(
        sub_agg_collector: Box<dyn SegmentAggregationCollector>,
        bucket_id_provider: BucketIdProvider,
        terms_req_data: TermsAggReqData,
    ) -> Self {
        Self {
            parent_buckets: vec![HashMapTermBuckets::<()>::default()],
            matched_parents: Vec::new(),
            matched_docs: Vec::new(),
            sub_agg_collector,
            finalized: None,
            bucket_id_provider,
            terms_req_data,
        }
    }

    fn record_single_parent(
        &mut self,
        parent_bucket_id: BucketId,
        docs: &[DocId],
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        let prev_cap = self.matched_docs.capacity();
        self.matched_parents
            .resize(self.matched_parents.len() + docs.len(), parent_bucket_id);
        self.matched_docs.extend_from_slice(docs);
        let new_cap = self.matched_docs.capacity();
        if new_cap > prev_cap {
            let delta = (new_cap - prev_cap)
                * (std::mem::size_of::<BucketId>() + std::mem::size_of::<DocId>());
            agg_data.context.limits.add_memory_consumed(delta as u64)?;
        }
        Ok(())
    }

    fn record_multi_parent(
        &mut self,
        parent_bucket_ids: &[BucketId],
        docs: &[DocId],
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        let prev_cap = self.matched_docs.capacity();
        self.matched_parents.extend_from_slice(parent_bucket_ids);
        self.matched_docs.extend_from_slice(docs);
        let new_cap = self.matched_docs.capacity();
        if new_cap > prev_cap {
            let delta = (new_cap - prev_cap)
                * (std::mem::size_of::<BucketId>() + std::mem::size_of::<DocId>());
            agg_data.context.limits.add_memory_consumed(delta as u64)?;
        }
        Ok(())
    }
}

impl SegmentAggregationCollector for NestedDeferredSubAggTermCollector {
    fn add_intermediate_aggregation_result(
        &mut self,
        agg_data: &AggregationsSegmentCtx,
        results: &mut IntermediateAggregationResults,
        parent_bucket_id: BucketId,
    ) -> crate::Result<()> {
        let finalized_entry = self
            .finalized
            .as_mut()
            .and_then(|vec| vec.get_mut(parent_bucket_id as usize))
            .map(std::mem::take)
            .unwrap_or_default();
        let name = self.terms_req_data.name.clone();
        let sub_agg_ref: &mut dyn SegmentAggregationCollector = self.sub_agg_collector.as_mut();
        let bucket = materialize_term_buckets(
            &self.terms_req_data,
            Some(sub_agg_ref),
            finalized_entry.entries,
            finalized_entry.term_doc_count_before_cutoff,
            finalized_entry.sum_other_doc_count,
            agg_data,
        )?;
        results.push(name, IntermediateAggregationResult::Bucket(bucket))?;
        Ok(())
    }

    fn collect(
        &mut self,
        parent_bucket_id: BucketId,
        docs: &[DocId],
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        self.record_single_parent(parent_bucket_id, docs, agg_data)?;

        let term_buckets = &mut self.parent_buckets[parent_bucket_id as usize];
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

    fn collect_multiple(
        &mut self,
        parent_bucket_ids: &[BucketId],
        docs: &[DocId],
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        debug_assert_eq!(parent_bucket_ids.len(), docs.len());
        if docs.is_empty() {
            return Ok(());
        }
        if !is_strictly_increasing(docs) {
            return collect_multiple_fallback(self, parent_bucket_ids, docs, agg_data);
        }
        self.record_multi_parent(parent_bucket_ids, docs, agg_data)?;

        agg_data
            .column_block_accessor
            .fetch_block_with_missing_unique_per_doc(
                docs,
                &mut *self.terms_req_data.accessor,
                self.terms_req_data.missing_value_for_accessor,
                true,
            );

        let allowed_bs = self.terms_req_data.allowed_term_ids.as_ref();
        let vals = agg_data.column_block_accessor.values();
        let mut mem_delta: usize = 0;

        if agg_data.column_block_accessor.cardinality().is_full() {
            for i in 0..docs.len() {
                let term_id = vals[i];
                if let Some(allowed) = allowed_bs {
                    if !allowed.contains(term_id as u32) {
                        continue;
                    }
                }
                let term_buckets = &mut self.parent_buckets[parent_bucket_ids[i] as usize];
                let mem_pre = term_buckets.get_memory_consumption();
                term_buckets.term_entry(term_id, &mut self.bucket_id_provider);
                mem_delta += term_buckets.get_memory_consumption() - mem_pre;
            }
        } else {
            let docids = agg_data.column_block_accessor.docids();
            let mut doc_idx = 0usize;
            for (&doc, &term_id) in docids.iter().zip(vals.iter()) {
                while docs[doc_idx] < doc {
                    doc_idx += 1;
                }
                debug_assert_eq!(docs[doc_idx], doc);
                if let Some(allowed) = allowed_bs {
                    if !allowed.contains(term_id as u32) {
                        continue;
                    }
                }
                let term_buckets = &mut self.parent_buckets[parent_bucket_ids[doc_idx] as usize];
                let mem_pre = term_buckets.get_memory_consumption();
                term_buckets.term_entry(term_id, &mut self.bucket_id_provider);
                mem_delta += term_buckets.get_memory_consumption() - mem_pre;
            }
        }

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

        let parent_maps = std::mem::take(&mut self.parent_buckets);
        let mut finalized_vec = Vec::with_capacity(parent_maps.len());
        let mut kept_map: FxHashMap<(BucketId, u64), BucketId> = FxHashMap::default();
        let mut next_child_bucket_id: BucketId = 0;

        for (parent_idx, term_map) in parent_maps.into_iter().enumerate() {
            let parent_id = parent_idx as BucketId;
            let mut raw_entries = term_map.into_vec();
            let (term_doc_count_before_cutoff, sum_other_doc_count) =
                select_and_cut_off_count_or_key(&mut raw_entries, &self.terms_req_data, false);
            let mut entries = Vec::with_capacity(raw_entries.len());
            for (key, bucket) in raw_entries {
                let child_bucket_id = next_child_bucket_id;
                next_child_bucket_id += 1;
                kept_map.insert((parent_id, key), child_bucket_id);
                entries.push((
                    key,
                    Bucket {
                        count: bucket.count,
                        bucket_id: child_bucket_id,
                    },
                ));
            }
            finalized_vec.push(FinalizedDeferredBuckets {
                entries,
                term_doc_count_before_cutoff,
                sum_other_doc_count,
            });
        }

        if next_child_bucket_id > 0 {
            self.sub_agg_collector
                .prepare_max_bucket(next_child_bucket_id - 1, agg_data)?;

            let use_filter = kept_map.len() <= 1024;
            let mut filter = [0u64; NESTED_FILTER_WORDS];
            if use_filter {
                for &(parent_id, key) in kept_map.keys() {
                    let (word, mask) = nested_filter_bit(parent_id, key);
                    filter[word] |= mask;
                }
            }

            let mut emitter = DirectCollectMultipleEmitter::new();
            let mut chunk_matches: Vec<(BucketId, DocId)> = Vec::with_capacity(PASS2_CHUNK_SIZE);
            let total = self.matched_docs.len();
            let mut start = 0usize;

            while start < total {
                let max_end = (start + PASS2_CHUNK_SIZE).min(total);
                let mut end = start + 1;
                while end < max_end && self.matched_docs[end] > self.matched_docs[end - 1] {
                    end += 1;
                }
                let run_docs = &self.matched_docs[start..end];
                let run_parents = &self.matched_parents[start..end];

                agg_data
                    .column_block_accessor
                    .fetch_block_with_missing_unique_per_doc(
                        run_docs,
                        &mut *self.terms_req_data.accessor,
                        self.terms_req_data.missing_value_for_accessor,
                        true,
                    );
                chunk_matches.clear();
                let vals = agg_data.column_block_accessor.values();
                if agg_data.column_block_accessor.cardinality().is_full() {
                    for i in 0..run_docs.len() {
                        let parent_id = run_parents[i];
                        let term_id = vals[i];
                        if use_filter {
                            let (word, mask) = nested_filter_bit(parent_id, term_id);
                            if (filter[word] & mask) == 0 {
                                continue;
                            }
                        }
                        if let Some(&child_bucket_id) = kept_map.get(&(parent_id, term_id)) {
                            chunk_matches.push((child_bucket_id, run_docs[i]));
                        }
                    }
                } else {
                    let docids = agg_data.column_block_accessor.docids();
                    let mut doc_idx = 0usize;
                    for (&doc, &term_id) in docids.iter().zip(vals.iter()) {
                        while run_docs[doc_idx] < doc {
                            doc_idx += 1;
                        }
                        let parent_id = run_parents[doc_idx];
                        if use_filter {
                            let (word, mask) = nested_filter_bit(parent_id, term_id);
                            if (filter[word] & mask) == 0 {
                                continue;
                            }
                        }
                        if let Some(&child_bucket_id) = kept_map.get(&(parent_id, term_id)) {
                            chunk_matches.push((child_bucket_id, doc));
                        }
                    }
                }

                for &(child_bucket_id, doc) in &chunk_matches {
                    emitter.push(
                        child_bucket_id,
                        doc,
                        self.sub_agg_collector.as_mut(),
                        agg_data,
                    )?;
                }
                start = end;
            }
            emitter.finish(self.sub_agg_collector.as_mut(), agg_data)?;
        } else {
            self.sub_agg_collector.flush(agg_data)?;
        }

        // Release Pass-1 submission buffers immediately once Pass 2 completes.
        self.matched_parents = Vec::new();
        self.matched_docs = Vec::new();
        self.finalized = Some(finalized_vec);
        Ok(())
    }

    fn prepare_max_bucket(
        &mut self,
        max_bucket: BucketId,
        _agg_data: &AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        while self.parent_buckets.len() <= max_bucket as usize {
            self.parent_buckets
                .push(HashMapTermBuckets::<()>::default());
        }
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
