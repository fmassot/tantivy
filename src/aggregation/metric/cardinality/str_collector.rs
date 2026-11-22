//! Segment collector for `cardinality` over a `ColumnType::Str` column.
//!
//! Strings are dictionary encoded, and resolving a term ordinal to its bytes
//! is the expensive part. So instead of hashing values during collection, the
//! collector accumulates *term ordinals* per bucket and, at finalization,
//! builds one shared term_ord -> coupon cache for every bucket at once. The
//! coupons are then appended into a [`CardinalityCollector`], which is the
//! same type the numeric collector produces — merging across segments and
//! across column kinds stays in the parent module.

use std::fmt::Debug;
use std::io;

use columnar::ColumnType;
use datasketches::hll::{Coupon, HllSketch, HllType};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use super::term_ord_accumulator::TermOrdAccumulator;
use super::{CardinalityAggReqData, CardinalityCollector, LG_K};
use crate::aggregation::agg_data::AggregationsSegmentCtx;
use crate::aggregation::intermediate_agg_result::{
    IntermediateAggregationResult, IntermediateAggregationResults, IntermediateMetricResult,
};
use crate::aggregation::segment_agg_result::{
    collect_multiple_fallback, is_strictly_increasing, SegmentAggregationCollector,
};
use crate::aggregation::*;

/// A CouponCache is here to cache the mapping term ordinal -> coupon (see above).
/// The idea is that we do not want to fetch terms associated to several term ordinals,
/// several times due to the fact that we have several buckets.
enum CouponCache {
    Dense {
        coupon_map: Vec<Coupon>,
        missing_coupon_opt: Option<Coupon>,
    },
    Sparse {
        coupon_map: FxHashMap<u64, Coupon>,
        missing_coupon_opt: Option<Coupon>,
    },
}

impl CouponCache {
    fn new(
        term_ords: Vec<u64>,
        coupons: Vec<Coupon>,
        missing_coupon_opt: Option<Coupon>,
    ) -> CouponCache {
        let num_terms = term_ords.len();
        assert_eq!(num_terms, coupons.len());
        if term_ords.is_empty() {
            return CouponCache::Dense {
                coupon_map: Vec::new(),
                missing_coupon_opt,
            };
        }
        let highest_term_ord = term_ords.last().copied().unwrap_or(0u64);
        // We prefer the dense implementation, if it is not too wasteful.
        // There are two cases for which we can use it.
        // 1- if the data is small.
        // 2- if the data is not necessarily small, but due to a high occupancy ratio, the RAM usage
        // is not that much bigger than if we had used a HashSet. (occupancy ratio + extra
        // metadata ~ x2.25)
        let should_use_dense =
            highest_term_ord < 1_000_000u64 || highest_term_ord < num_terms as u64 * 3u64;
        if should_use_dense {
            // We don't really care about the value here. We will populate all the values we will
            // read anyway.
            let uninitialized_coupon = Coupon::from_value(0);
            let mut coupon_map: Vec<Coupon> =
                vec![uninitialized_coupon; highest_term_ord as usize + 1];

            for (term_ord, coupon) in term_ords.into_iter().zip(coupons) {
                coupon_map[term_ord as usize] = coupon;
            }
            CouponCache::Dense {
                coupon_map,
                missing_coupon_opt,
            }
        } else {
            let coupon_map: FxHashMap<u64, Coupon> = term_ords.into_iter().zip(coupons).collect();
            CouponCache::Sparse {
                coupon_map,
                missing_coupon_opt,
            }
        }
    }
}

/// Segment collector for `cardinality` over a `ColumnType::Str` column.
///
/// Hidden contract: the column passed at construction must be a str column
/// whose values are term ordinals of the associated dictionary, and
/// `max_term_ord_inclusive` must be `accessor.max_value()`. The missing
/// sentinel `accessor.max_value() + 1` may additionally be inserted when
/// `missing_value_for_accessor` is set, hence accumulators size for
/// `max_term_ord_inclusive + 1`.
pub(crate) struct SegmentStrCardinalityCollector<S: TermOrdAccumulator> {
    /// Buckets are Some(_) until they get consumed by
    /// `add_intermediate_aggregation_result`.
    buckets: Vec<Option<S>>,
    /// Most recent term ordinal inserted per bucket (`u64::MAX` means none yet), skipping
    /// consecutive duplicates before touching the set/bitset.
    last_ords: Vec<u64>,
    /// The request data. Its accessor gives access to the term ordinals.
    req_data: CardinalityAggReqData,
    /// Lazily built at finalization time, shared by every bucket.
    coupon_cache: Option<CouponCache>,
    /// Largest term_ord that may be inserted into a bucket, i.e.
    /// `accessor.max_value()`.
    max_term_ord_inclusive: u64,
}

impl<S: TermOrdAccumulator> Debug for SegmentStrCardinalityCollector<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("SegmentStrCardinalityCollector")
            .field("num_buckets", &self.buckets.len())
            .field(
                "missing_value_for_accessor",
                &self.req_data.missing_value_for_accessor,
            )
            .finish()
    }
}

/// Upper bound on dictionary size for using a contiguous flat bitset union in `build_coupon_cache`
/// (~128 KiB bitset for 1M terms), avoiding `FxHashSet` hashing and `sort_unstable`.
const COUPON_CACHE_FLAT_UNION_MAX_TERMS: u64 = 1_000_000;

/// Builds a coupon cache from the given buckets, dictionary, and optional missing value.
/// Returns a mapping from term_ord to the hash (coupon) of the associated term.
fn build_coupon_cache<S: TermOrdAccumulator>(
    buckets: &[Option<S>],
    dictionary: &dyn ValueSourceDictionary,
    missing_value_opt: Option<&Key>,
) -> io::Result<CouponCache> {
    let missing_coupon_opt: Option<Coupon> = missing_value_opt.map(missing_coupon_for_key);
    let dict_num_terms = dictionary.num_terms();
    if dict_num_terms == 0 {
        return Ok(CouponCache::Dense {
            coupon_map: Vec::new(),
            missing_coupon_opt,
        });
    }

    if dict_num_terms <= COUPON_CACHE_FLAT_UNION_MAX_TERMS {
        let num_words = (dict_num_terms as usize).div_ceil(64);
        let mut union_words = vec![0u64; num_words];
        for bucket in buckets.iter().flatten() {
            bucket.union_into_words(&mut union_words, dict_num_terms);
        }
        let mut num_hit_terms: usize = 0;
        let mut highest_word_idx: Option<usize> = None;
        for (word_idx, &word) in union_words.iter().enumerate() {
            if word != 0 {
                num_hit_terms += word.count_ones() as usize;
                highest_word_idx = Some(word_idx);
            }
        }
        let Some(highest_word_idx) = highest_word_idx else {
            return Ok(CouponCache::Dense {
                coupon_map: Vec::new(),
                missing_coupon_opt,
            });
        };
        let highest_bit = 63 - union_words[highest_word_idx].leading_zeros() as usize;
        let highest_ord = (highest_word_idx << 6) + highest_bit;

        // Dense streaming fast path: when >=25% of dictionary terms are hit across buckets,
        // stream the SSTable once and fill the dense `coupon_map` in-place — skipping `Vec<u64>`
        // materialization, intermediate `Vec<Coupon>`, and second copy.
        if num_hit_terms.saturating_mul(4) >= dict_num_terms as usize {
            let mut coupon_map = vec![Coupon::from_value(0); highest_ord + 1];
            let mut current_ord: usize = 0;
            let limit_ord = highest_ord + 1;
            dictionary.stream_all_terms_cb(&mut |term_bytes| {
                if current_ord < limit_ord {
                    let word = union_words[current_ord >> 6];
                    if (word >> (current_ord & 63)) & 1 != 0 {
                        coupon_map[current_ord] = Coupon::from_value(term_bytes);
                    }
                }
                current_ord += 1;
            })?;
            return Ok(CouponCache::Dense {
                coupon_map,
                missing_coupon_opt,
            });
        }

        // Sparse path: extract sorted ordinals directly from the bitset (no FxHashSet or sort).
        let mut term_ords: Vec<u64> = Vec::with_capacity(num_hit_terms);
        for (word_idx, &word_val) in union_words[..=highest_word_idx].iter().enumerate() {
            let mut word = word_val;
            let base = (word_idx as u64) << 6;
            while word != 0 {
                let bit = word.trailing_zeros() as u64;
                term_ords.push(base + bit);
                word &= word - 1;
            }
        }
        let mut coupon_map = vec![Coupon::from_value(0); highest_ord + 1];
        let mut ord_iter = term_ords.iter().copied();
        let all_term_ords_found: bool =
            dictionary.sorted_ords_to_term_cb(&term_ords, &mut |term_bytes| {
                if let Some(ord) = ord_iter.next() {
                    coupon_map[ord as usize] = Coupon::from_value(term_bytes);
                }
            })?;
        assert!(all_term_ords_found);
        return Ok(CouponCache::Dense {
            coupon_map,
            missing_coupon_opt,
        });
    }

    // Fallback for dictionaries exceeding 1M terms: use an FxHashSet union.
    let mut max_bucket_len = 0usize;
    for bucket in buckets.iter().flatten() {
        max_bucket_len = max_bucket_len.max(bucket.len());
    }
    let mut term_ords_set = FxHashSet::with_capacity_and_hasher(max_bucket_len * 2, FxBuildHasher);
    for bucket in buckets.iter().flatten() {
        term_ords_set.extend(bucket.iter_ords());
    }
    let mut term_ords: Vec<u64> = term_ords_set.into_iter().collect();
    term_ords.sort_unstable();

    term_ords.pop_if(|highest_term_ord| *highest_term_ord >= dict_num_terms);

    let mut coupons: Vec<Coupon> = Vec::with_capacity(term_ords.len());
    let all_term_ords_found: bool =
        dictionary.sorted_ords_to_term_cb(&term_ords, &mut |term_bytes| {
            let coupon: Coupon = Coupon::from_value(term_bytes);
            coupons.push(coupon);
        })?;
    assert!(all_term_ords_found);

    Ok(CouponCache::new(term_ords, coupons, missing_coupon_opt))
}

fn append_to_sketch(
    term_ords: &impl TermOrdAccumulator,
    coupon_cache: &CouponCache,
    sketch: &mut CardinalityCollector,
) {
    match coupon_cache {
        CouponCache::Dense {
            coupon_map,
            missing_coupon_opt,
        } => {
            if let Some(missing_coupon) = missing_coupon_opt {
                for term_ord in term_ords.iter_ords() {
                    let coupon: Coupon = coupon_map
                        .get(term_ord as usize)
                        .copied()
                        .unwrap_or(*missing_coupon);
                    sketch.insert_coupon(coupon);
                }
            } else {
                for term_ord in term_ords.iter_ords() {
                    if let Some(coupon) = coupon_map.get(term_ord as usize).copied() {
                        sketch.insert_coupon(coupon);
                    }
                }
            }
        }
        CouponCache::Sparse {
            coupon_map,
            missing_coupon_opt,
        } => {
            for term_ord in term_ords.iter_ords() {
                if let Some(coupon) = coupon_map.get(&term_ord).copied().or(*missing_coupon_opt) {
                    sketch.insert_coupon(coupon);
                }
            }
        }
    }
}

impl<S: TermOrdAccumulator> SegmentStrCardinalityCollector<S> {
    pub(crate) fn from_req(req_data: CardinalityAggReqData, max_term_ord_inclusive: u64) -> Self {
        Self {
            buckets: Vec::new(),
            last_ords: Vec::new(),
            req_data,
            coupon_cache: None,
            max_term_ord_inclusive,
        }
    }
}

fn missing_coupon_for_key(missing_key: &Key) -> Coupon {
    if let Key::Str(missing_value_str) = missing_key {
        Coupon::from_value(missing_value_str.as_bytes())
    } else {
        // See https://github.com/quickwit-oss/tantivy/issues/2891
        // A missing key with a type different from Str will not work as intended
        // for the moment.
        //
        // Right now this is just a partial workaround.
        Coupon::from_value("__tantivy_missing_non_str__".as_bytes())
    }
}

impl<S: TermOrdAccumulator + 'static> SegmentAggregationCollector
    for SegmentStrCardinalityCollector<S>
{
    fn add_intermediate_aggregation_result(
        &mut self,
        agg_data: &AggregationsSegmentCtx,
        results: &mut IntermediateAggregationResults,
        bucket_id: BucketId,
    ) -> crate::Result<()> {
        self.prepare_max_bucket(bucket_id, agg_data)?;
        let req_data = &self.req_data;
        // `ValueSource::term_dictionary` guarantees a dictionary for every `Str` source.
        let Some(term_dictionary) = req_data.accessor.term_dictionary() else {
            return Err(crate::TantivyError::InternalError(
                "a `Str` value source must provide a term dictionary".to_string(),
            ));
        };
        let name = req_data.name.to_string();

        // Single-bucket fast path (e.g. top-level cardinality over a str column):
        // stream unique ordinals directly into the HLL sketch without building an
        // intermediate FxHashSet, Vec<Coupon>, or CouponCache.
        if self.buckets.len() == 1 && self.coupon_cache.is_none() {
            let Some(term_ords) = self.buckets[bucket_id as usize].take() else {
                return Err(crate::TantivyError::InternalError(
                    "the same bucket should not be finalized twice.".to_string(),
                ));
            };
            let mut sketch =
                HllSketch::new(LG_K, HllType::Hll8).expect("LG_K is within the supported range");
            let dict_num_terms = term_dictionary.num_terms();
            if req_data.missing_value_for_accessor.is_none()
                && term_ords.len() as u64 == dict_num_terms
            {
                // Every term in the dictionary was hit and no missing sentinel can be present:
                // stream the SSTable directly without materializing a Vec<u64> of ordinals.
                term_dictionary.stream_all_terms_cb(&mut |term_bytes| {
                    sketch.update_with_coupon(Coupon::from_value(term_bytes));
                })?;
            } else {
                let mut ords: Vec<u64> = Vec::with_capacity(term_ords.len());
                ords.extend(term_ords.iter_ords());
                if !term_ords.is_iter_sorted() {
                    ords.sort_unstable();
                }
                let had_missing = ords
                    .pop_if(|highest_term_ord| *highest_term_ord >= dict_num_terms)
                    .is_some();
                let all_found =
                    term_dictionary.sorted_ords_to_term_cb(&ords, &mut |term_bytes| {
                        sketch.update_with_coupon(Coupon::from_value(term_bytes));
                    })?;
                assert!(all_found);
                if had_missing {
                    if let Some(missing_key) = req_data.req.missing.as_ref() {
                        sketch.update_with_coupon(missing_coupon_for_key(missing_key));
                    }
                }
            }
            let cardinality = CardinalityCollector::from_sketch(sketch, ColumnType::Str as u8);
            results.push(
                name,
                IntermediateAggregationResult::Metric(IntermediateMetricResult::Cardinality(
                    cardinality,
                )),
            )?;
            return Ok(());
        }

        // Strings are dictionary encoded. Fetching the terms associated to strings
        // is expensive. For this reason, we do that once for all buckets and cache the results
        // here.
        //
        // The cache maps a term_ord to the hash of the associated term. The missing value
        // sentinel will be associated to the hash of the missing value if any.
        if self.coupon_cache.is_none() {
            self.coupon_cache = Some(build_coupon_cache(
                &self.buckets,
                term_dictionary,
                req_data.req.missing.as_ref(),
            )?);
        }
        // take the bucket in buckets and replace it with a new empty one
        let Some(term_ords) = self.buckets[bucket_id as usize].take() else {
            return Err(crate::TantivyError::InternalError(
                "the same bucket should not be finalized twice.".to_string(),
            ));
        };
        let mut cardinality = CardinalityCollector::new(ColumnType::Str as u8);
        if let Some(coupon_cache) = self.coupon_cache.as_ref() {
            append_to_sketch(&term_ords, coupon_cache, &mut cardinality);
        }
        results.push(
            name,
            IntermediateAggregationResult::Metric(IntermediateMetricResult::Cardinality(
                cardinality,
            )),
        )?;

        Ok(())
    }

    fn collect(
        &mut self,
        parent_bucket_id: BucketId,
        docs: &[crate::DocId],
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        agg_data.column_block_accessor.fetch_block_with_missing(
            docs,
            &mut *self.req_data.accessor,
            self.req_data.missing_value_for_accessor,
        );
        let bucket_idx = parent_bucket_id as usize;
        let Some(term_ords) = self.buckets[bucket_idx].as_mut() else {
            return Err(crate::TantivyError::InternalError(
                "collection should not happen after finalization".to_string(),
            ));
        };
        term_ords.maybe_compact();
        let mut last_ord = self.last_ords[bucket_idx];
        term_ords.extend_from_iter(
            agg_data
                .column_block_accessor
                .values()
                .iter()
                .copied()
                .filter(|&ord| {
                    if ord == last_ord {
                        false
                    } else {
                        last_ord = ord;
                        true
                    }
                }),
        );
        self.last_ords[bucket_idx] = last_ord;
        Ok(())
    }

    fn collect_multiple(
        &mut self,
        bucket_ids: &[BucketId],
        docs: &[crate::DocId],
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        debug_assert_eq!(bucket_ids.len(), docs.len());
        if docs.is_empty() {
            return Ok(());
        }
        if !is_strictly_increasing(docs) {
            return collect_multiple_fallback(self, bucket_ids, docs, agg_data);
        }
        agg_data
            .column_block_accessor
            .fetch_block_with_missing_ordered(
                docs,
                &mut *self.req_data.accessor,
                self.req_data.missing_value_for_accessor,
                true,
            );
        let col_block_accessor = &agg_data.column_block_accessor;
        let vals = col_block_accessor.values();
        if col_block_accessor.cardinality().is_full() {
            for (&bucket_id, &ord) in bucket_ids.iter().zip(vals.iter()) {
                let bucket_idx = bucket_id as usize;
                if self.last_ords[bucket_idx] == ord {
                    continue;
                }
                self.last_ords[bucket_idx] = ord;
                let Some(term_ords) = self.buckets[bucket_idx].as_mut() else {
                    return Err(crate::TantivyError::InternalError(
                        "collection should not happen after finalization".to_string(),
                    ));
                };
                term_ords.maybe_compact();
                term_ords.insert(ord);
            }
        } else {
            let fetched_docs = col_block_accessor.docids();
            let mut doc_pos = 0usize;
            for (&doc, &ord) in fetched_docs.iter().zip(vals.iter()) {
                while docs[doc_pos] < doc {
                    doc_pos += 1;
                }
                let bucket_idx = bucket_ids[doc_pos] as usize;
                if self.last_ords[bucket_idx] == ord {
                    continue;
                }
                self.last_ords[bucket_idx] = ord;
                let Some(term_ords) = self.buckets[bucket_idx].as_mut() else {
                    return Err(crate::TantivyError::InternalError(
                        "collection should not happen after finalization".to_string(),
                    ));
                };
                term_ords.maybe_compact();
                term_ords.insert(ord);
            }
        }
        Ok(())
    }

    fn prepare_max_bucket(
        &mut self,
        max_bucket: BucketId,
        _agg_data: &AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        let required = max_bucket as usize + 1;
        if required > self.buckets.len() {
            let max_term_ord_inclusive = self.max_term_ord_inclusive;
            self.buckets
                .resize_with(required, || Some(S::new(max_term_ord_inclusive)));
            self.last_ords.resize(required, u64::MAX);
        }
        Ok(())
    }

    fn compute_metric_value(
        &self,
        bucket_id: BucketId,
        sub_agg_name: &str,
        sub_agg_property: &str,
        _agg_data: &AggregationsSegmentCtx,
    ) -> Option<f64> {
        if self.req_data.name != sub_agg_name || !sub_agg_property.is_empty() {
            return None;
        }
        // The sketch isn't built until finalization; the term_ord set's len is
        // the exact distinct count.
        let term_ords = self.buckets.get(bucket_id as usize)?.as_ref()?;
        Some(term_ords.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use crate::aggregation::agg_req::Aggregations;
    use crate::aggregation::tests::{exec_request, get_test_index_from_terms};
    use crate::schema::{Schema, FAST, STRING};
    use crate::Index;

    /// Build a single-segment string-cardinality index with 32 unique terms.
    /// `column.max_value() = 31` is well below `BITSET_MAX_TERM_ORD`,
    /// so the bucket exercises the `BitSet` path end to end.
    #[test]
    fn cardinality_aggregation_test_str_bitset() -> crate::Result<()> {
        let terms: Vec<String> = (0..32).map(|i| format!("term_{i}")).collect();
        let term_refs: Vec<Vec<&str>> = terms.iter().map(|t| vec![t.as_str()]).collect::<Vec<_>>();
        // single segment so we have a single dictionary of 32 terms.
        let index = get_test_index_from_terms(true, &term_refs)?;

        let agg_req: Aggregations = serde_json::from_value(json!({
            "cardinality": {
                "cardinality": { "field": "string_id" }
            },
        }))
        .unwrap();

        let res = exec_request(agg_req, &index)?;
        assert_eq!(res["cardinality"]["value"], 32.0);
        Ok(())
    }

    /// `BitSet` path with a `missing` parameter: the column-level missing
    /// sentinel (`column.max_value() + 1`) flows into the bitset, the
    /// dict lookup filter at finalization drops it, and the missing
    /// coupon is applied separately.
    #[test]
    fn cardinality_aggregation_test_str_bitset_with_missing() {
        let mut schema_builder = Schema::builder();
        let name_field = schema_builder.add_text_field("name", STRING | FAST);
        let index = Index::create_in_ram(schema_builder.build());
        let mut writer = index.writer_for_tests().unwrap();
        for i in 0..16 {
            let term = format!("t{i:02}");
            writer.add_document(doc!(name_field => term)).unwrap();
        }
        // One empty doc, exercising the missing sentinel.
        writer.add_document(doc!()).unwrap();
        writer.commit().unwrap();

        let agg_req: Aggregations = serde_json::from_value(json!({
            "cardinality": {
                "cardinality": {
                    "field": "name",
                    "missing": "MISSING_SENTINEL_KEY",
                }
            },
        }))
        .unwrap();

        let res = exec_request(agg_req, &index).unwrap();
        // 16 distinct real terms + 1 distinct "missing" value = 17.
        assert_eq!(res["cardinality"]["value"], 17.0);
    }
}
