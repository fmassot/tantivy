use std::fmt::Debug;
use std::mem;
use std::net::Ipv6Addr;

use columnar::column_values::CompactSpaceU64Accessor;
use columnar::{
    Cardinality, Column, ColumnType, Dictionary, MonotonicallyMappableToU128,
    MonotonicallyMappableToU64, NumericalValue, StrColumn,
};
use fastdivide::DividerU64;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::aggregation::agg_data::AggregationsSegmentCtx;
use crate::aggregation::bucket::composite::accessors::{
    CompositeAccessor, CompositeAggReqData, PrecomputedDateInterval,
};
use crate::aggregation::bucket::composite::calendar_interval;
use crate::aggregation::bucket::composite::map::{DynArrayHeapMap, MAX_DYN_ARRAY_SIZE};
use crate::aggregation::bucket::{
    CalendarInterval, CompositeAggregationSource, MissingOrder, Order,
};
use crate::aggregation::buffered_sub_aggs::{BufferedSubAggs, HighCardSubAggBuffer};
use crate::aggregation::intermediate_agg_result::{
    CompositeIntermediateKey, IntermediateAggregationResult, IntermediateAggregationResults,
    IntermediateBucketResult, IntermediateCompositeBucketEntry, IntermediateCompositeBucketResult,
};
use crate::aggregation::segment_agg_result::{BucketIdProvider, SegmentAggregationCollector};
use crate::aggregation::BucketId;
use crate::TantivyError;

#[derive(Clone, Debug)]
struct CompositeBucketCollector {
    count: u32,
    bucket_id: BucketId,
}

/// Compact sortable representation of a single source value within a composite key.
///
/// The struct encodes both the column identity and the fast field value in a way
/// that preserves the desired sort order via the derived `Ord` implementation
/// (fields are compared top-to-bottom: `sort_key` first, then `encoded_value`).
///
/// ## `sort_key` encoding
/// - `0` — missing value, sorted first
/// - `1..=254` — present value; the original accessor index is `sort_key - 1`
/// - `u8::MAX` (255) — missing value, sorted last
///
/// ## `encoded_value` encoding
/// - `0` when the field is missing
/// - The raw u64 fast-field representation when order is ascending
/// - Bitwise NOT of the raw u64 when order is descending
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
struct InternalValueRepr {
    /// Column index biased by +1 (so 0 and u8::MAX are reserved for missing sentinels).
    sort_key: u8,
    /// Fast field value, possibly bit-flipped for descending order.
    encoded_value: u64,
}

impl InternalValueRepr {
    #[inline]
    fn new_term(raw: u64, accessor_idx: u8, order: Order) -> Self {
        let encoded_value = match order {
            Order::Asc => raw,
            Order::Desc => !raw,
        };
        InternalValueRepr {
            sort_key: accessor_idx + 1,
            encoded_value,
        }
    }

    /// For histogram sources the column index is irrelevant (always 1).
    #[inline]
    fn new_histogram(raw: u64, order: Order) -> Self {
        let encoded_value = match order {
            Order::Asc => raw,
            Order::Desc => !raw,
        };
        InternalValueRepr {
            sort_key: 1,
            encoded_value,
        }
    }

    #[inline]
    fn new_missing(order: Order, missing_order: MissingOrder) -> Self {
        let sort_key = match (missing_order, order) {
            (MissingOrder::First, _) | (MissingOrder::Default, Order::Asc) => 0,
            (MissingOrder::Last, _) | (MissingOrder::Default, Order::Desc) => u8::MAX,
        };
        InternalValueRepr {
            sort_key,
            encoded_value: 0,
        }
    }

    /// Decode back to `(accessor_idx, raw_value)`.
    /// Returns `None` when the value represents a missing field.
    #[inline]
    fn decode(self, order: Order) -> Option<(u8, u64)> {
        if self.sort_key == 0 || self.sort_key == u8::MAX {
            return None;
        }
        let raw = match order {
            Order::Asc => self.encoded_value,
            Order::Desc => !self.encoded_value,
        };
        Some((self.sort_key - 1, raw))
    }
}

/// Dense flat counter array for single-source fixed `date_histogram` composite aggregations.
struct DenseDateBuckets {
    /// Bucket index (`ts_ns / interval_ns`) corresponding to slot 0.
    base_bucket_idx: i64,
    /// Number of slots per parent bucket (`<= 4096`).
    span: usize,
    /// One flat counter slice per `parent_bucket_id`.
    parent_counts: Vec<Vec<u32>>,
}

/// Pre-validated fast execution strategy for common single-column-per-source composite queries.
enum CompositeFastPath {
    /// Single `date_histogram` source with `FixedNanoseconds` interval, single non-multivalued
    /// column, no `after` cursor, no `missing_bucket`, no sub-aggregations.
    SingleDateHistogramFixed {
        column: Column<u64>,
        is_full: bool,
        interval_ns: i64,
        divider: DividerU64,
        order: Order,
        /// Optional dense flat counter array per parent bucket when the column's timestamp span
        /// covers at most 4096 buckets (avoids all hash-map lookups during collection).
        dense: Option<DenseDateBuckets>,
    },
    /// 1..=4 `terms` sources, each backed by a single `Cardinality::Full` column, no `after`
    /// cursor, no `missing_bucket`, no sub-aggregations. Enables block fetching and top-K
    /// prefix pruning before reading trailing columns.
    MultiTermsFull {
        columns: SmallVec<[Column<u64>; 4]>,
        orders: SmallVec<[Order; 4]>,
    },
    /// General recursive visitor supporting all source types, `after`, `missing_bucket`,
    /// multivalued/JSON columns, and sub-aggregations.
    General,
}

impl Debug for CompositeFastPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompositeFastPath::SingleDateHistogramFixed {
                is_full,
                interval_ns,
                order,
                ..
            } => f
                .debug_struct("SingleDateHistogramFixed")
                .field("is_full", is_full)
                .field("interval_ns", interval_ns)
                .field("order", order)
                .finish(),
            CompositeFastPath::MultiTermsFull { orders, .. } => f
                .debug_struct("MultiTermsFull")
                .field("orders", orders)
                .finish(),
            CompositeFastPath::General => f.write_str("General"),
        }
    }
}

/// The collector puts values from the fast field into the correct buckets and
/// does a conversion to the correct datatype.
#[derive(Debug)]
pub struct SegmentCompositeCollector {
    /// One DynArrayHeapMap per parent bucket.
    parent_buckets: Vec<DynArrayHeapMap<InternalValueRepr, CompositeBucketCollector>>,
    req_data: CompositeAggReqData,
    sub_agg: Option<BufferedSubAggs<HighCardSubAggBuffer>>,
    bucket_id_provider: BucketIdProvider,
    /// Number of sources, needed when creating new DynArrayHeapMaps.
    num_sources: usize,
    /// Fast path selected once at segment collector construction.
    fast_path: CompositeFastPath,
}

impl SegmentAggregationCollector for SegmentCompositeCollector {
    fn add_intermediate_aggregation_result(
        &mut self,
        agg_data: &AggregationsSegmentCtx,
        results: &mut IntermediateAggregationResults,
        parent_bucket_id: BucketId,
    ) -> crate::Result<()> {
        let name = self.req_data.name.clone();

        let buckets = self.add_intermediate_bucket_result(agg_data, parent_bucket_id)?;
        results.push(
            name,
            IntermediateAggregationResult::Bucket(IntermediateBucketResult::Composite { buckets }),
        )?;

        Ok(())
    }

    fn collect(
        &mut self,
        parent_bucket_id: BucketId,
        docs: &[crate::DocId],
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        let mem_pre = self.get_memory_consumption(parent_bucket_id);
        let limit = self.req_data.req.size as usize;

        match &mut self.fast_path {
            CompositeFastPath::SingleDateHistogramFixed {
                column,
                is_full,
                interval_ns,
                divider,
                order,
                dense,
            } => {
                let accessor = &mut agg_data.column_block_accessor;
                if *is_full {
                    accessor.fetch_full_column_block(docs, column.values.as_ref());
                } else {
                    let mut col_source = (column.clone(), ColumnType::DateTime);
                    accessor.fetch_block_with_missing(docs, &mut col_source, None);
                }
                let interval_ns = *interval_ns;
                let interval_u64 = interval_ns as u64;
                let divider = *divider;
                let order = *order;

                if let Some(dense_buckets) = dense.as_mut() {
                    let base = dense_buckets.base_bucket_idx;
                    let counts = &mut dense_buckets.parent_counts[parent_bucket_id as usize];
                    for raw_val in accessor.iter_vals() {
                        let bucket_idx =
                            compute_fixed_date_bucket_idx(raw_val, interval_ns, divider);
                        let slot = (bucket_idx - base) as usize;
                        debug_assert!(slot < counts.len());
                        counts[slot] += 1;
                    }
                    return Ok(());
                }

                let buckets = &mut self.parent_buckets[parent_bucket_id as usize];
                let mut vals = accessor.iter_vals();
                if let Some(first_raw) = vals.next() {
                    let mut run_bucket =
                        compute_fixed_date_bucket(first_raw, interval_ns, interval_u64, divider);
                    let mut run_count: u32 = 1;
                    for raw_val in vals {
                        let bucket_val =
                            compute_fixed_date_bucket(raw_val, interval_ns, interval_u64, divider);
                        if bucket_val == run_bucket {
                            run_count += 1;
                        } else {
                            let key = [InternalValueRepr::new_histogram(run_bucket, order)];
                            add_count_to_bucket(limit, buckets, &key, run_count);
                            run_bucket = bucket_val;
                            run_count = 1;
                        }
                    }
                    let key = [InternalValueRepr::new_histogram(run_bucket, order)];
                    add_count_to_bucket(limit, buckets, &key, run_count);
                }
            }
            CompositeFastPath::MultiTermsFull { columns, orders } => {
                let buckets = &mut self.parent_buckets[parent_bucket_id as usize];
                collect_multi_terms_full_fast(docs, columns, orders, limit, buckets);
            }
            CompositeFastPath::General => {
                for doc in docs {
                    let mut visitor = CompositeKeyVisitor {
                        doc_id: *doc,
                        composite_agg_data: &self.req_data,
                        buckets: &mut self.parent_buckets[parent_bucket_id as usize],
                        sub_agg: &mut self.sub_agg,
                        bucket_id_provider: &mut self.bucket_id_provider,
                        sub_level_values: SmallVec::new(),
                    };
                    visitor.visit(0, true)?;
                }

                if let Some(sub_agg) = &mut self.sub_agg {
                    sub_agg.check_flush_local(agg_data)?;
                }
            }
        }

        let mem_delta = self
            .get_memory_consumption(parent_bucket_id)
            .saturating_sub(mem_pre);
        if mem_delta > 0 {
            agg_data.context.limits.add_memory_consumed(mem_delta)?;
        }

        Ok(())
    }

    fn flush(&mut self, agg_data: &mut AggregationsSegmentCtx) -> crate::Result<()> {
        if let Some(sub_agg) = &mut self.sub_agg {
            sub_agg.flush(agg_data)?;
        }
        Ok(())
    }

    fn prepare_max_bucket(
        &mut self,
        max_bucket: BucketId,
        _agg_data: &AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        let required_len = max_bucket as usize + 1;
        while self.parent_buckets.len() < required_len {
            let map = DynArrayHeapMap::try_new(self.num_sources)?;
            self.parent_buckets.push(map);
        }
        if let CompositeFastPath::SingleDateHistogramFixed {
            dense: Some(dense_buckets),
            ..
        } = &mut self.fast_path
        {
            while dense_buckets.parent_counts.len() < required_len {
                dense_buckets
                    .parent_counts
                    .push(vec![0u32; dense_buckets.span]);
            }
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
        // Composite is a multi-bucket agg with no single value to extract.
        None
    }
}

impl SegmentCompositeCollector {
    fn get_memory_consumption(&self, parent_bucket_id: BucketId) -> u64 {
        self.parent_buckets[parent_bucket_id as usize].memory_consumption()
    }

    pub(crate) fn from_req_and_validate(
        composite_req_data: CompositeAggReqData,
        sub_agg: Option<Box<dyn SegmentAggregationCollector>>,
    ) -> crate::Result<Self> {
        validate_req(&composite_req_data)?;
        let sub_agg = sub_agg.map(BufferedSubAggs::new);

        let num_sources = composite_req_data.req.sources.len();
        let fast_path = select_composite_fast_path(&composite_req_data, sub_agg.is_some());

        Ok(SegmentCompositeCollector {
            parent_buckets: vec![DynArrayHeapMap::try_new(num_sources)?],
            req_data: composite_req_data,
            sub_agg,
            bucket_id_provider: BucketIdProvider::default(),
            num_sources,
            fast_path,
        })
    }

    #[inline]
    fn add_intermediate_bucket_result(
        &mut self,
        agg_data: &AggregationsSegmentCtx,
        parent_bucket_id: BucketId,
    ) -> crate::Result<IntermediateCompositeBucketResult> {
        let composite_data = &self.req_data;
        if let CompositeFastPath::SingleDateHistogramFixed {
            interval_ns,
            order,
            dense: Some(dense_buckets),
            ..
        } = &mut self.fast_path
        {
            let limit = composite_data.req.size as usize;
            let base = dense_buckets.base_bucket_idx;
            let interval_ns = *interval_ns;
            let order = *order;
            let counts = mem::take(&mut dense_buckets.parent_counts[parent_bucket_id as usize]);
            let mut dict: FxHashMap<
                Vec<CompositeIntermediateKey>,
                IntermediateCompositeBucketEntry,
            > = Default::default();
            dict.reserve(limit.min(counts.len()));
            let emit = |slot: usize,
                        count: u32,
                        dict: &mut FxHashMap<
                Vec<CompositeIntermediateKey>,
                IntermediateCompositeBucketEntry,
            >| {
                if count > 0 {
                    let bucket_ts = (base + slot as i64) * interval_ns;
                    dict.insert(
                        vec![CompositeIntermediateKey::DateTime(bucket_ts)],
                        IntermediateCompositeBucketEntry {
                            doc_count: count as u64,
                            sub_aggregation: IntermediateAggregationResults::default(),
                        },
                    );
                }
            };
            match order {
                Order::Asc => {
                    for (slot, &count) in counts.iter().enumerate() {
                        emit(slot, count, &mut dict);
                        if dict.len() >= limit {
                            break;
                        }
                    }
                }
                Order::Desc => {
                    for (slot, &count) in counts.iter().enumerate().rev() {
                        emit(slot, count, &mut dict);
                        if dict.len() >= limit {
                            break;
                        }
                    }
                }
            }
            return Ok(IntermediateCompositeBucketResult {
                entries: dict,
                target_size: composite_data.req.size,
                orders: composite_data
                    .req
                    .sources
                    .iter()
                    .map(|source| match source {
                        CompositeAggregationSource::Terms(t) => (t.order, t.missing_order),
                        CompositeAggregationSource::Histogram(h) => (h.order, h.missing_order),
                        CompositeAggregationSource::DateHistogram(d) => (d.order, d.missing_order),
                    })
                    .collect(),
            });
        }

        let empty_map = DynArrayHeapMap::try_new(self.num_sources)?;
        let heap_map = mem::replace(
            &mut self.parent_buckets[parent_bucket_id as usize],
            empty_map,
        );

        let mut dict: FxHashMap<Vec<CompositeIntermediateKey>, IntermediateCompositeBucketEntry> =
            Default::default();
        dict.reserve(heap_map.size());
        for (key_internal_repr, agg) in heap_map.into_iter() {
            let key = resolve_key(&key_internal_repr, composite_data)?;
            let mut sub_aggregation_res = IntermediateAggregationResults::default();
            if let Some(sub_agg) = &mut self.sub_agg {
                sub_agg
                    .get_sub_agg_collector()
                    .add_intermediate_aggregation_result(
                        agg_data,
                        &mut sub_aggregation_res,
                        agg.bucket_id,
                    )?;
            }

            dict.insert(
                key,
                IntermediateCompositeBucketEntry {
                    doc_count: agg.count as u64,
                    sub_aggregation: sub_aggregation_res,
                },
            );
        }

        Ok(IntermediateCompositeBucketResult {
            entries: dict,
            target_size: composite_data.req.size,
            orders: composite_data
                .req
                .sources
                .iter()
                .map(|source| match source {
                    CompositeAggregationSource::Terms(t) => (t.order, t.missing_order),
                    CompositeAggregationSource::Histogram(h) => (h.order, h.missing_order),
                    CompositeAggregationSource::DateHistogram(d) => (d.order, d.missing_order),
                })
                .collect(),
        })
    }
}

#[inline(always)]
fn compute_fixed_date_bucket_idx(raw_val: u64, interval_ns: i64, divider: DividerU64) -> i64 {
    let val_ns = i64::from_u64(raw_val);
    if val_ns >= 0 {
        divider.divide(val_ns as u64) as i64
    } else {
        val_ns / interval_ns
    }
}

#[inline(always)]
fn compute_fixed_date_bucket(
    raw_val: u64,
    interval_ns: i64,
    interval_u64: u64,
    divider: DividerU64,
) -> u64 {
    let val_ns = i64::from_u64(raw_val);
    let bucket_ns = if val_ns >= 0 {
        (divider.divide(val_ns as u64).wrapping_mul(interval_u64)) as i64
    } else {
        (val_ns / interval_ns) * interval_ns
    };
    i64::to_u64(bucket_ns)
}

#[inline(always)]
fn add_count_to_bucket(
    limit_num_buckets: usize,
    buckets: &mut DynArrayHeapMap<InternalValueRepr, CompositeBucketCollector>,
    key: &[InternalValueRepr],
    count: u32,
) {
    if buckets.size() < limit_num_buckets {
        let bucket = buckets.get_or_insert_with(key, || CompositeBucketCollector {
            count: 0,
            bucket_id: 0,
        });
        bucket.count += count;
        return;
    }

    if let Some(bucket) = buckets.get_mut(key) {
        bucket.count += count;
        return;
    }

    if let Some(highest_key) = buckets.peek_highest() {
        if key < highest_key {
            buckets.evict_highest();
            let bucket = buckets.get_or_insert_with(key, || CompositeBucketCollector {
                count: 0,
                bucket_id: 0,
            });
            bucket.count += count;
        }
    }
}

fn select_composite_fast_path(
    req_data: &CompositeAggReqData,
    has_sub_agg: bool,
) -> CompositeFastPath {
    if has_sub_agg || !req_data.req.after.is_empty() {
        return CompositeFastPath::General;
    }
    let sources = &req_data.req.sources;
    if sources.len() == 1 {
        if let CompositeAggregationSource::DateHistogram(date_src) = &sources[0] {
            let level = &req_data.composite_accessors[0];
            if !date_src.missing_bucket && level.accessors.len() == 1 {
                let acc = &level.accessors[0];
                let card = acc.column.index.get_cardinality();
                if acc.column_type == ColumnType::DateTime && !card.is_multivalue() {
                    if let PrecomputedDateInterval::FixedNanoseconds(interval_ns) =
                        acc.date_histogram_interval
                    {
                        if interval_ns > 0 {
                            let divider = DividerU64::divide_by(interval_ns as u64);
                            let dense = if acc.column.values.num_vals() > 0 {
                                let min_idx = compute_fixed_date_bucket_idx(
                                    acc.column.min_value(),
                                    interval_ns,
                                    divider,
                                );
                                let max_idx = compute_fixed_date_bucket_idx(
                                    acc.column.max_value(),
                                    interval_ns,
                                    divider,
                                );
                                const MAX_DENSE_DATE_BUCKETS: i64 = 4096;
                                if max_idx >= min_idx
                                    && (max_idx - min_idx) < MAX_DENSE_DATE_BUCKETS
                                {
                                    let span = (max_idx - min_idx + 1) as usize;
                                    Some(DenseDateBuckets {
                                        base_bucket_idx: min_idx,
                                        span,
                                        parent_counts: vec![vec![0u32; span]],
                                    })
                                } else {
                                    None
                                }
                            } else {
                                None
                            };
                            return CompositeFastPath::SingleDateHistogramFixed {
                                column: acc.column.clone(),
                                is_full: card == Cardinality::Full,
                                interval_ns,
                                divider,
                                order: date_src.order,
                                dense,
                            };
                        }
                    }
                }
            }
        }
    }

    if !sources.is_empty() && sources.len() <= 4 {
        let mut columns = SmallVec::<[Column<u64>; 4]>::new();
        let mut orders = SmallVec::<[Order; 4]>::new();
        let mut eligible = true;
        for (source, level) in sources.iter().zip(&req_data.composite_accessors) {
            let CompositeAggregationSource::Terms(terms_src) = source else {
                eligible = false;
                break;
            };
            if terms_src.missing_bucket || level.accessors.len() != 1 {
                eligible = false;
                break;
            }
            let acc = &level.accessors[0];
            if acc.column.index.get_cardinality() != Cardinality::Full {
                eligible = false;
                break;
            }
            columns.push(acc.column.clone());
            orders.push(terms_src.order);
        }
        if eligible {
            return CompositeFastPath::MultiTermsFull { columns, orders };
        }
    }

    CompositeFastPath::General
}

#[inline(always)]
fn encode_term_value(raw: u64, order: Order) -> InternalValueRepr {
    InternalValueRepr::new_term(raw, 0, order)
}

fn collect_multi_terms_full_fast(
    docs: &[crate::DocId],
    columns: &[Column<u64>],
    orders: &[Order],
    limit: usize,
    buckets: &mut DynArrayHeapMap<InternalValueRepr, CompositeBucketCollector>,
) {
    let num_sources = columns.len();
    debug_assert!((1..=4).contains(&num_sources));

    let mut vals_buf = [0u64; 64];
    let mut surv_docs = [0u32; 64];
    let mut surv_keys = [[InternalValueRepr::default(); 4]; 64];

    for chunk in docs.chunks(64) {
        let chunk_len = chunk.len();
        columns[0]
            .values
            .get_vals(chunk, &mut vals_buf[..chunk_len]);
        let order0 = orders[0];

        let mut surv_len = 0usize;
        if buckets.size() >= limit {
            let max_key_0 = buckets.peek_highest().unwrap()[0];
            for i in 0..chunk_len {
                let encoded0 = encode_term_value(vals_buf[i], order0);
                if encoded0 <= max_key_0 {
                    surv_docs[surv_len] = chunk[i];
                    surv_keys[surv_len][0] = encoded0;
                    surv_len += 1;
                }
            }
            if surv_len == 0 {
                continue;
            }

            // Read subsequent columns only for surviving docs, pruning at each prefix level.
            let mut max_key = [InternalValueRepr::default(); 4];
            if let Some(highest) = buckets.peek_highest() {
                max_key[..num_sources].copy_from_slice(highest);
            }
            for src_idx in 1..num_sources {
                columns[src_idx]
                    .values
                    .get_vals(&surv_docs[..surv_len], &mut vals_buf[..surv_len]);
                let order = orders[src_idx];
                let mut next_surv = 0usize;
                for i in 0..surv_len {
                    surv_keys[i][src_idx] = encode_term_value(vals_buf[i], order);
                    if surv_keys[i][..=src_idx] <= max_key[..=src_idx] {
                        if next_surv != i {
                            surv_docs[next_surv] = surv_docs[i];
                            surv_keys[next_surv] = surv_keys[i];
                        }
                        next_surv += 1;
                    }
                }
                surv_len = next_surv;
                if surv_len == 0 {
                    break;
                }
            }
            for key_arr in &surv_keys[..surv_len] {
                add_count_to_bucket(limit, buckets, &key_arr[..num_sources], 1);
            }
        } else {
            for i in 0..chunk_len {
                surv_keys[i][0] = encode_term_value(vals_buf[i], order0);
            }
            for src_idx in 1..num_sources {
                columns[src_idx]
                    .values
                    .get_vals(chunk, &mut vals_buf[..chunk_len]);
                let order = orders[src_idx];
                for i in 0..chunk_len {
                    surv_keys[i][src_idx] = encode_term_value(vals_buf[i], order);
                }
            }
            for key_arr in &surv_keys[..chunk_len] {
                add_count_to_bucket(limit, buckets, &key_arr[..num_sources], 1);
            }
        }
    }
}

fn validate_req(composite_data: &CompositeAggReqData) -> crate::Result<()> {
    let req = &composite_data.req;
    if req.sources.is_empty() {
        return Err(TantivyError::InvalidArgument(
            "composite aggregation must have at least one source".to_string(),
        ));
    }
    if req.size == 0 {
        return Err(TantivyError::InvalidArgument(
            "composite aggregation 'size' must be > 0".to_string(),
        ));
    }

    if composite_data.composite_accessors.len() > MAX_DYN_ARRAY_SIZE {
        return Err(TantivyError::InvalidArgument(format!(
            "composite aggregation source supports maximum {MAX_DYN_ARRAY_SIZE} sources",
        )));
    }

    let column_types_for_sources = composite_data.composite_accessors.iter().map(|item| {
        item.accessors
            .iter()
            .map(|a| a.column_type)
            .collect::<Vec<_>>()
    });

    for column_types in column_types_for_sources {
        if column_types.contains(&ColumnType::Bytes) {
            return Err(TantivyError::InvalidArgument(
                "composite aggregation does not support 'bytes' field type".to_string(),
            ));
        }
    }
    Ok(())
}

fn collect_bucket_with_limit(
    doc_id: crate::DocId,
    limit_num_buckets: usize,
    buckets: &mut DynArrayHeapMap<InternalValueRepr, CompositeBucketCollector>,
    key: &[InternalValueRepr],
    sub_agg: &mut Option<BufferedSubAggs<HighCardSubAggBuffer>>,
    bucket_id_provider: &mut BucketIdProvider,
) {
    let mut record_in_bucket = |bucket: &mut CompositeBucketCollector| {
        bucket.count += 1;
        if let Some(sub_agg) = sub_agg {
            sub_agg.push(bucket.bucket_id, doc_id);
        }
    };

    // We still have room for buckets, just insert
    if buckets.size() < limit_num_buckets {
        let bucket = buckets.get_or_insert_with(key, || CompositeBucketCollector {
            count: 0,
            bucket_id: bucket_id_provider.next_bucket_id(),
        });
        record_in_bucket(bucket);
        return;
    }

    // Map is full, but we can still update the bucket if it already exists
    if let Some(bucket) = buckets.get_mut(key) {
        record_in_bucket(bucket);
        return;
    }

    // Check if the item qualifies to enter the top-k, and evict the highest if it does
    if let Some(highest_key) = buckets.peek_highest() {
        if key < highest_key {
            buckets.evict_highest();
            let bucket = buckets.get_or_insert_with(key, || CompositeBucketCollector {
                count: 0,
                bucket_id: bucket_id_provider.next_bucket_id(),
            });
            record_in_bucket(bucket);
        }
    }
}

/// Converts the composite key from its internal column space representation
/// (segment specific) into its intermediate form.
fn resolve_key(
    internal_key: &[InternalValueRepr],
    agg_data: &CompositeAggReqData,
) -> crate::Result<Vec<CompositeIntermediateKey>> {
    internal_key
        .iter()
        .enumerate()
        .map(|(idx, val)| {
            resolve_internal_value_repr(
                *val,
                &agg_data.req.sources[idx],
                &agg_data.composite_accessors[idx].accessors,
            )
        })
        .collect()
}

fn resolve_internal_value_repr(
    internal_value_repr: InternalValueRepr,
    source: &CompositeAggregationSource,
    composite_accessors: &[CompositeAccessor],
) -> crate::Result<CompositeIntermediateKey> {
    let decoded_value_opt = match source {
        CompositeAggregationSource::Terms(source) => internal_value_repr.decode(source.order),
        CompositeAggregationSource::Histogram(source) => internal_value_repr.decode(source.order),
        CompositeAggregationSource::DateHistogram(source) => {
            internal_value_repr.decode(source.order)
        }
    };
    let Some((decoded_accessor_idx, val)) = decoded_value_opt else {
        return Ok(CompositeIntermediateKey::Null);
    };
    let key = match source {
        CompositeAggregationSource::Terms(_) => {
            let CompositeAccessor {
                column_type,
                str_dict_column,
                column,
                ..
            } = &composite_accessors[decoded_accessor_idx as usize];
            resolve_term(val, column_type, str_dict_column, column)?
        }
        CompositeAggregationSource::Histogram(source) => {
            CompositeIntermediateKey::F64(i64::from_u64(val) as f64 * source.interval)
        }
        CompositeAggregationSource::DateHistogram(_) => {
            CompositeIntermediateKey::DateTime(i64::from_u64(val))
        }
    };

    Ok(key)
}

fn resolve_term(
    val: u64,
    column_type: &ColumnType,
    str_dict_column: &Option<StrColumn>,
    column: &Column,
) -> crate::Result<CompositeIntermediateKey> {
    let key = if *column_type == ColumnType::Str {
        let fallback_dict = Dictionary::empty();
        let term_dict = str_dict_column
            .as_ref()
            .map(|el| el.dictionary())
            .unwrap_or_else(|| &fallback_dict);

        let mut buffer = Vec::new();
        term_dict.ord_to_term(val, &mut buffer)?;
        CompositeIntermediateKey::Str(
            String::from_utf8(buffer.to_vec()).expect("could not convert to String"),
        )
    } else if *column_type == ColumnType::DateTime {
        let val = i64::from_u64(val);
        CompositeIntermediateKey::DateTime(val)
    } else if *column_type == ColumnType::Bool {
        let val = bool::from_u64(val);
        CompositeIntermediateKey::Bool(val)
    } else if *column_type == ColumnType::IpAddr {
        let compact_space_accessor = column
            .values
            .clone()
            .downcast_arc::<CompactSpaceU64Accessor>()
            .map_err(|_| {
                TantivyError::AggregationError(crate::aggregation::AggregationError::InternalError(
                    "Type mismatch: Could not downcast to CompactSpaceU64Accessor".to_string(),
                ))
            })?;
        let val: u128 = compact_space_accessor.compact_to_u128(val as u32);
        let val = Ipv6Addr::from_u128(val);
        CompositeIntermediateKey::IpAddr(val)
    } else if *column_type == ColumnType::U64 {
        CompositeIntermediateKey::U64(val)
    } else if *column_type == ColumnType::I64 {
        CompositeIntermediateKey::I64(i64::from_u64(val))
    } else {
        let val = f64::from_u64(val);
        let val: NumericalValue = val.into();

        match val.normalize() {
            NumericalValue::U64(val) => CompositeIntermediateKey::U64(val),
            NumericalValue::I64(val) => CompositeIntermediateKey::I64(val),
            NumericalValue::F64(val) => CompositeIntermediateKey::F64(val),
        }
    };
    Ok(key)
}

/// Browse through the cardinal product obtained by the different values of the doc composite key
/// sources.
///
/// For each of those tuple-key, that are after the limit key, we call collect_bucket_with_limit.
struct CompositeKeyVisitor<'a> {
    doc_id: crate::DocId,
    composite_agg_data: &'a CompositeAggReqData,
    buckets: &'a mut DynArrayHeapMap<InternalValueRepr, CompositeBucketCollector>,
    sub_agg: &'a mut Option<BufferedSubAggs<HighCardSubAggBuffer>>,
    bucket_id_provider: &'a mut BucketIdProvider,
    sub_level_values: SmallVec<[InternalValueRepr; MAX_DYN_ARRAY_SIZE]>,
}

impl CompositeKeyVisitor<'_> {
    /// Depth-first walk of the accessors to build the composite key combinations
    /// and update the buckets.
    ///
    /// `source_idx` is the current source index in the recursion.
    /// `is_on_after_key` tracks whether we still need to consider the after_key
    /// for pruning at this level and below.
    fn visit(&mut self, source_idx: usize, is_on_after_key: bool) -> crate::Result<()> {
        if source_idx == self.composite_agg_data.req.sources.len() {
            if !is_on_after_key {
                collect_bucket_with_limit(
                    self.doc_id,
                    self.composite_agg_data.req.size as usize,
                    self.buckets,
                    &self.sub_level_values,
                    self.sub_agg,
                    self.bucket_id_provider,
                );
            }
            return Ok(());
        }

        let current_level_accessors = &self.composite_agg_data.composite_accessors[source_idx];
        let current_level_source = &self.composite_agg_data.req.sources[source_idx];
        let mut missing = true;
        for (accessor_idx, accessor) in current_level_accessors.accessors.iter().enumerate() {
            let values = accessor.column.values_for_doc(self.doc_id);
            for value in values {
                missing = false;
                match current_level_source {
                    CompositeAggregationSource::Terms(_) => {
                        let preceeds_after_key_type =
                            accessor_idx < current_level_accessors.after_key_accessor_idx;
                        if is_on_after_key && preceeds_after_key_type {
                            break;
                        }
                        let matches_after_key_type =
                            accessor_idx == current_level_accessors.after_key_accessor_idx;

                        if matches_after_key_type && is_on_after_key {
                            let should_skip = match current_level_source.order() {
                                Order::Asc => current_level_accessors.after_key.gt(value),
                                Order::Desc => current_level_accessors.after_key.lt(value),
                            };
                            if should_skip {
                                continue;
                            }
                        }
                        self.sub_level_values.push(InternalValueRepr::new_term(
                            value,
                            accessor_idx as u8,
                            current_level_source.order(),
                        ));
                        let still_on_after_key = matches_after_key_type
                            && current_level_accessors.after_key.equals(value);
                        self.visit(source_idx + 1, is_on_after_key && still_on_after_key)?;
                        self.sub_level_values.pop();
                    }
                    CompositeAggregationSource::Histogram(source) => {
                        let float_value = match accessor.column_type {
                            ColumnType::U64 => value as f64,
                            ColumnType::I64 => i64::from_u64(value) as f64,
                            ColumnType::DateTime => i64::from_u64(value) as f64 / 1_000_000.,
                            ColumnType::F64 => f64::from_u64(value),
                            _ => {
                                panic!(
                                    "unexpected type {:?}. This should not happen",
                                    accessor.column_type
                                )
                            }
                        };
                        let bucket_index = (float_value / source.interval).floor() as i64;
                        let bucket_value = i64::to_u64(bucket_index);
                        if is_on_after_key {
                            let should_skip = match current_level_source.order() {
                                Order::Asc => current_level_accessors.after_key.gt(bucket_value),
                                Order::Desc => current_level_accessors.after_key.lt(bucket_value),
                            };
                            if should_skip {
                                continue;
                            }
                        }
                        self.sub_level_values.push(InternalValueRepr::new_histogram(
                            bucket_value,
                            current_level_source.order(),
                        ));
                        let still_on_after_key =
                            current_level_accessors.after_key.equals(bucket_value);
                        self.visit(source_idx + 1, is_on_after_key && still_on_after_key)?;
                        self.sub_level_values.pop();
                    }
                    CompositeAggregationSource::DateHistogram(_) => {
                        let value_ns = match accessor.column_type {
                            ColumnType::DateTime => i64::from_u64(value),
                            _ => {
                                panic!(
                                    "unexpected type {:?}. This should not happen",
                                    accessor.column_type
                                )
                            }
                        };
                        let bucket_index = match accessor.date_histogram_interval {
                            PrecomputedDateInterval::FixedNanoseconds(fixed_interval_ns) => {
                                (value_ns / fixed_interval_ns) * fixed_interval_ns
                            }
                            PrecomputedDateInterval::Calendar(CalendarInterval::Year) => {
                                calendar_interval::try_year_bucket(value_ns)?
                            }
                            PrecomputedDateInterval::Calendar(CalendarInterval::Month) => {
                                calendar_interval::try_month_bucket(value_ns)?
                            }
                            PrecomputedDateInterval::Calendar(CalendarInterval::Week) => {
                                calendar_interval::week_bucket(value_ns)
                            }
                            PrecomputedDateInterval::NotApplicable => {
                                panic!("interval not precomputed for date histogram source")
                            }
                        };
                        let bucket_value = i64::to_u64(bucket_index);
                        if is_on_after_key {
                            let should_skip = match current_level_source.order() {
                                Order::Asc => current_level_accessors.after_key.gt(bucket_value),
                                Order::Desc => current_level_accessors.after_key.lt(bucket_value),
                            };
                            if should_skip {
                                continue;
                            }
                        }
                        self.sub_level_values.push(InternalValueRepr::new_histogram(
                            bucket_value,
                            current_level_source.order(),
                        ));
                        let still_on_after_key =
                            current_level_accessors.after_key.equals(bucket_value);
                        self.visit(source_idx + 1, is_on_after_key && still_on_after_key)?;
                        self.sub_level_values.pop();
                    }
                };
            }
        }
        if missing && current_level_source.missing_bucket() {
            if is_on_after_key && current_level_accessors.skip_missing {
                return Ok(());
            }
            self.sub_level_values.push(InternalValueRepr::new_missing(
                current_level_source.order(),
                current_level_source.missing_order(),
            ));
            self.visit(
                source_idx + 1,
                is_on_after_key && current_level_accessors.is_after_key_explicit_missing,
            )?;
            self.sub_level_values.pop();
        }
        Ok(())
    }
}
