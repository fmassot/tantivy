use std::fmt::Debug;

use super::segment_agg_result::SegmentAggregationCollector;
use crate::aggregation::agg_data::AggregationsSegmentCtx;
use crate::aggregation::BucketId;
use crate::DocId;

/// A buffer for sub-aggregations, storing doc ids per bucket id.
/// Depending on the cardinality of the parent aggregation, we use different
/// storage strategies.
///
/// ## Low Cardinality
/// Cardinality here refers to the number of unique flattened buckets that can be created
/// by the parent aggregation.
/// Flattened buckets are the result of combining all buckets per collector
/// into a single list of buckets, where each bucket is identified by its BucketId.
///
/// ## Usage
/// Since this is caching for sub-aggregations, it is only used by bucket
/// aggregations.
///
/// TODO: consider using a more advanced data structure for high cardinality
/// aggregations.
/// What this datastructure does in general is to group docs by bucket id.
///
/// Submissions can accumulate across parent collection calls. Both backends preserve
/// submission order for each bucket without sorting or deduplicating, so callers must
/// supply strictly increasing document IDs per bucket, as required by
/// [`SegmentAggregationCollector`]. Otherwise a flush can pass unordered or duplicate
/// document IDs to a child collector even when each parent call contained sorted,
/// duplicate-free input.
#[derive(Debug)]
pub(crate) struct BufferedSubAggs<B: SubAggBuffer> {
    buffer: B,
    sub_agg_collector: Box<dyn SegmentAggregationCollector>,
    num_docs: usize,
    flush_threshold: usize,
}

pub type LowCardBufferedSubAggs = BufferedSubAggs<LowCardSubAggBuffer>;
pub type HighCardBufferedSubAggs = BufferedSubAggs<HighCardSubAggBuffer>;

const FLUSH_THRESHOLD: usize = 2048;

/// A trait for buffering sub-aggregation doc ids per bucket id.
/// Different implementations can be used depending on the cardinality
/// of the parent aggregation.
pub trait SubAggBuffer: Debug {
    fn new() -> Self;
    fn push(&mut self, bucket_id: BucketId, doc_id: DocId);
    /// Ensures internal storage can accept bucket IDs up to `max_bucket_id` without resizing on
    /// every push.
    #[inline]
    fn ensure_buckets(&mut self, _max_bucket_id: BucketId) {}
    /// Pushes `(bucket_id, doc_id)` assuming `ensure_buckets` has already been called for at least
    /// `bucket_id`.
    #[inline]
    fn push_preallocated(&mut self, bucket_id: BucketId, doc_id: DocId) {
        self.push(bucket_id, doc_id);
    }
    fn flush_local(
        &mut self,
        sub_agg: &mut Box<dyn SegmentAggregationCollector>,
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()>;
}

impl<Backend: SubAggBuffer + Debug> BufferedSubAggs<Backend> {
    pub fn new(sub_agg: Box<dyn SegmentAggregationCollector>) -> Self {
        Self::with_flush_threshold(sub_agg, FLUSH_THRESHOLD)
    }

    pub fn with_flush_threshold(
        sub_agg: Box<dyn SegmentAggregationCollector>,
        flush_threshold: usize,
    ) -> Self {
        Self {
            buffer: Backend::new(),
            sub_agg_collector: sub_agg,
            num_docs: 0,
            flush_threshold,
        }
    }

    pub fn get_sub_agg_collector(&mut self) -> &mut dyn SegmentAggregationCollector {
        &mut *self.sub_agg_collector
    }

    #[inline]
    pub fn push(&mut self, bucket_id: BucketId, doc_id: DocId) {
        self.buffer.push(bucket_id, doc_id);
        self.num_docs += 1;
    }

    /// Ensures internal storage can accept bucket IDs up to `max_bucket_id` without resizing on
    /// every push.
    #[inline]
    pub fn ensure_buckets(&mut self, max_bucket_id: BucketId) {
        self.buffer.ensure_buckets(max_bucket_id);
    }

    /// Pushes `(bucket_id, doc_id)` without incrementing `num_docs` or checking bucket table
    /// bounds. Callers must call [`Self::ensure_buckets`] beforehand and
    /// [`Self::add_buffered_docs`] afterwards.
    #[inline]
    pub fn push_preallocated(&mut self, bucket_id: BucketId, doc_id: DocId) {
        self.buffer.push_preallocated(bucket_id, doc_id);
    }

    /// Accounts for `count` documents pushed via [`Self::push_preallocated`].
    #[inline]
    pub fn add_buffered_docs(&mut self, count: usize) {
        self.num_docs += count;
    }

    /// Check if we need to flush based on the number of documents cached.
    /// If so, flushes the cache to the provided aggregation collector.
    pub fn check_flush_local(
        &mut self,
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        if self.num_docs >= self.flush_threshold {
            self.buffer
                .flush_local(&mut self.sub_agg_collector, agg_data)?;
            self.num_docs = 0;
        }
        Ok(())
    }

    /// Note: this _does_ flush the sub aggregations.
    pub fn flush(&mut self, agg_data: &mut AggregationsSegmentCtx) -> crate::Result<()> {
        if self.num_docs != 0 {
            self.buffer
                .flush_local(&mut self.sub_agg_collector, agg_data)?;
            self.num_docs = 0;
        }
        self.sub_agg_collector.flush(agg_data)?;
        Ok(())
    }
}

/// Number of partitions for high cardinality sub-aggregation buffer.
const NUM_PARTITIONS: usize = 16;

#[derive(Debug)]
pub(crate) struct HighCardSubAggBuffer {
    /// This weird partitioning is used to do some cheap grouping on the bucket ids.
    /// bucket ids are dense, e.g. when we don't detect the cardinality as low cardinality,
    /// but there are just 16 bucket ids, each bucket id will go to its own partition.
    ///
    /// We want to keep this cheap, because high cardinality aggregations can have a lot of
    /// buckets, and there may be nothing to group.
    partitions: Box<[PartitionEntry; NUM_PARTITIONS]>,
}

impl HighCardSubAggBuffer {
    #[inline]
    fn clear(&mut self) {
        for partition in self.partitions.iter_mut() {
            partition.clear();
        }
    }
}

#[derive(Debug, Clone, Default)]
struct PartitionEntry {
    bucket_ids: Vec<BucketId>,
    docs: Vec<DocId>,
}

impl PartitionEntry {
    #[inline]
    fn clear(&mut self) {
        self.bucket_ids.clear();
        self.docs.clear();
    }
}

impl SubAggBuffer for HighCardSubAggBuffer {
    fn new() -> Self {
        Self {
            partitions: Box::new(core::array::from_fn(|_| PartitionEntry::default())),
        }
    }

    #[inline]
    fn push(&mut self, bucket_id: BucketId, doc_id: DocId) {
        let idx = bucket_id % NUM_PARTITIONS as u32;
        let slot = &mut self.partitions[idx as usize];
        slot.bucket_ids.push(bucket_id);
        slot.docs.push(doc_id);
    }

    fn flush_local(
        &mut self,
        sub_agg: &mut Box<dyn SegmentAggregationCollector>,
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        let mut max_bucket = 0u32;
        for partition in self.partitions.iter() {
            if let Some(&local_max) = partition.bucket_ids.iter().max() {
                max_bucket = max_bucket.max(local_max);
            }
        }

        sub_agg.prepare_max_bucket(max_bucket, agg_data)?;

        for slot in self.partitions.iter() {
            if !slot.bucket_ids.is_empty() {
                // Reduce dynamic dispatch overhead by collecting a full partition in one call.
                sub_agg.collect_multiple(&slot.bucket_ids, &slot.docs, agg_data)?;
            }
        }

        self.clear();
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct LowCardSubAggBuffer {
    /// Buffer doc ids per bucket for sub-aggregations.
    ///
    /// The outer Vec is indexed by BucketId.
    per_bucket_docs: Vec<Vec<DocId>>,
}

impl LowCardSubAggBuffer {
    #[inline]
    fn clear(&mut self) {
        for v in &mut self.per_bucket_docs {
            v.clear();
        }
    }
}

impl SubAggBuffer for LowCardSubAggBuffer {
    fn new() -> Self {
        Self {
            per_bucket_docs: Vec::new(),
        }
    }

    #[inline]
    fn push(&mut self, bucket_id: BucketId, doc_id: DocId) {
        let idx = bucket_id as usize;
        if self.per_bucket_docs.len() <= idx {
            self.per_bucket_docs.resize_with(idx + 1, Vec::new);
        }
        self.per_bucket_docs[idx].push(doc_id);
    }

    #[inline]
    fn ensure_buckets(&mut self, max_bucket_id: BucketId) {
        let required_len = max_bucket_id as usize + 1;
        if self.per_bucket_docs.len() < required_len {
            self.per_bucket_docs.resize_with(required_len, Vec::new);
        }
    }

    #[inline]
    fn push_preallocated(&mut self, bucket_id: BucketId, doc_id: DocId) {
        self.per_bucket_docs[bucket_id as usize].push(doc_id);
    }

    fn flush_local(
        &mut self,
        sub_agg: &mut Box<dyn SegmentAggregationCollector>,
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        // Pre-aggregated: call collect per bucket.
        let max_bucket = (self.per_bucket_docs.len() as BucketId).saturating_sub(1);
        sub_agg.prepare_max_bucket(max_bucket, agg_data)?;
        for (bucket_id, docs) in self.per_bucket_docs.iter().enumerate() {
            sub_agg.collect(bucket_id as BucketId, docs, agg_data)?;
        }

        self.clear();
        Ok(())
    }
}
