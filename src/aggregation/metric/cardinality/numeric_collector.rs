//! Segment collector for `cardinality` over any non-str column
//! (numeric, bool, date, IpAddr).
//!
//! Unlike the str case there is no dictionary to resolve, so values go
//! straight into a per-bucket state during collection. Buckets with up to
//! [`INLINE_CARDINALITY_CAP`] distinct values store them inline without heap
//! allocation or MurmurHash3 hashing, promoting to an Apache DataSketches
//! HLL8 sketch on the 5th distinct value. The produced [`CardinalityCollector`]
//! is the same type the str collector produces, so intermediate merging stays
//! in the parent module.

use std::fmt::Debug;
use std::sync::Arc;

use columnar::column_values::CompactSpaceU64Accessor;
use columnar::{Column, ColumnType};

use super::{CardinalityAggReqData, CardinalityCollector};
use crate::aggregation::agg_data::AggregationsSegmentCtx;
use crate::aggregation::intermediate_agg_result::{
    IntermediateAggregationResult, IntermediateAggregationResults, IntermediateMetricResult,
};
use crate::aggregation::segment_agg_result::{
    collect_multiple_fallback, is_strictly_increasing, SegmentAggregationCollector,
};
use crate::aggregation::value_source::ValueSource;
use crate::aggregation::*;
use crate::TantivyError;

/// Number of distinct raw `u64` values stored inline per bucket before allocating an HLL sketch.
///
/// Apache DataSketches HLL starts in exact `List` mode for up to 8 unique entries, so storing up
/// to 4 unique raw values inline and replaying them in first-seen order upon promotion produces a
/// bit-identical HLL sketch while avoiding heap allocation, hashing, and cubic interpolation on
/// the vast majority of small buckets in high-cardinality `terms -> cardinality` queries.
const INLINE_CARDINALITY_CAP: usize = 4;

enum BucketSketch {
    Empty,
    Inline {
        vals: [u64; INLINE_CARDINALITY_CAP],
        len: u8,
    },
    Sketch {
        collector: CardinalityCollector,
        last_val: u64,
    },
    Finalized,
}

impl BucketSketch {
    #[inline]
    fn insert_raw(
        &mut self,
        val: u64,
        salt: u8,
        compact_space_accessor: Option<&CompactSpaceU64Accessor>,
    ) -> crate::Result<()> {
        match self {
            BucketSketch::Empty => {
                *self = BucketSketch::Inline {
                    vals: [val, 0, 0, 0],
                    len: 1,
                };
                Ok(())
            }
            BucketSketch::Inline { vals, len } => {
                let count = *len as usize;
                for i in 0..count {
                    if vals[i] == val {
                        return Ok(());
                    }
                }
                if count < INLINE_CARDINALITY_CAP {
                    vals[count] = val;
                    *len = (count + 1) as u8;
                    return Ok(());
                }
                let mut collector = CardinalityCollector::new(salt);
                if let Some(csa) = compact_space_accessor {
                    for &prev in vals.iter() {
                        collector.insert(csa.compact_to_u128(prev as u32));
                    }
                    collector.insert(csa.compact_to_u128(val as u32));
                } else {
                    for &prev in vals.iter() {
                        collector.insert(prev);
                    }
                    collector.insert(val);
                }
                *self = BucketSketch::Sketch {
                    collector,
                    last_val: val,
                };
                Ok(())
            }
            BucketSketch::Sketch {
                collector,
                last_val,
            } => {
                if *last_val == val {
                    return Ok(());
                }
                *last_val = val;
                if let Some(csa) = compact_space_accessor {
                    collector.insert(csa.compact_to_u128(val as u32));
                } else {
                    collector.insert(val);
                }
                Ok(())
            }
            BucketSketch::Finalized => Err(crate::TantivyError::InternalError(
                "collection should not happen after finalization".to_string(),
            )),
        }
    }

    #[inline]
    fn estimate(&self) -> Option<f64> {
        match self {
            BucketSketch::Empty => Some(0.0),
            BucketSketch::Inline { len, .. } => Some(*len as f64),
            BucketSketch::Sketch { collector, .. } => Some(collector.estimate().trunc()),
            BucketSketch::Finalized => None,
        }
    }

    fn take_collector(
        &mut self,
        salt: u8,
        compact_space_accessor: Option<&CompactSpaceU64Accessor>,
    ) -> crate::Result<CardinalityCollector> {
        match std::mem::replace(self, BucketSketch::Finalized) {
            BucketSketch::Empty => Ok(CardinalityCollector::new(salt)),
            BucketSketch::Inline { vals, len } => {
                let mut collector = CardinalityCollector::new(salt);
                let count = len as usize;
                if let Some(csa) = compact_space_accessor {
                    for &val in &vals[..count] {
                        collector.insert(csa.compact_to_u128(val as u32));
                    }
                } else {
                    for &val in &vals[..count] {
                        collector.insert(val);
                    }
                }
                Ok(collector)
            }
            BucketSketch::Sketch { collector, .. } => Ok(collector),
            BucketSketch::Finalized => Err(crate::TantivyError::InternalError(
                "the same bucket should not be finalized twice.".to_string(),
            )),
        }
    }
}

/// Segment collector for `cardinality` over any non-str column
/// (numeric, bool, date, IpAddr).
///
/// Hidden contract: `column_type` must not be `ColumnType::Str`. Values are
/// inserted into per-bucket inline storage or HLL sketches during collection,
/// so the state of a bucket is already complete when the bucket is finalized.
pub(crate) struct SegmentNumericCardinalityCollector {
    /// Per-bucket cardinality state, replaced with `BucketSketch::Finalized` when consumed.
    buckets: Vec<BucketSketch>,
    /// The name of the aggregation.
    name: String,
    /// The column accessor to access the fast field values.
    accessor: Box<dyn ValueSource>,
    /// Physical column resolved once at construction to avoid per-block virtual dispatch.
    column_opt: Option<(Column<u64>, ColumnType)>,
    /// The column_type of the field.
    column_type: ColumnType,
    /// Set iff `column_type == ColumnType::IpAddr`. Resolved once at
    /// construction: the raw column values are compact-space codes that must
    /// be expanded to their u128 ip representation before hashing.
    compact_space_accessor: Option<Arc<CompactSpaceU64Accessor>>,
    /// The missing value normalized to the internal u64 representation of the field type.
    missing_value_for_accessor: Option<u64>,
}

impl Debug for SegmentNumericCardinalityCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("SegmentNumericCardinalityCollector")
            .field("column_type", &self.column_type)
            .field(
                "missing_value_for_accessor",
                &self.missing_value_for_accessor,
            )
            .finish()
    }
}

impl SegmentNumericCardinalityCollector {
    pub(crate) fn from_req(req_data: CardinalityAggReqData) -> crate::Result<Self> {
        let CardinalityAggReqData {
            accessor,
            missing_value_for_accessor,
            name,
            ..
        } = req_data;
        let column_type = accessor.column_type();
        assert_ne!(column_type, ColumnType::Str);
        let compact_space_accessor = if column_type == ColumnType::IpAddr {
            let compact_space_accessor = accessor
                .as_physical_column()
                .ok_or_else(|| {
                    TantivyError::AggregationError(
                        crate::aggregation::AggregationError::InternalError(
                            "IpAddr cardinality requires a physical column".to_string(),
                        ),
                    )
                })?
                .values
                .clone()
                .downcast_arc::<CompactSpaceU64Accessor>()
                .map_err(|_| {
                    TantivyError::AggregationError(
                        crate::aggregation::AggregationError::InternalError(
                            "Type mismatch: Could not downcast to CompactSpaceU64Accessor"
                                .to_string(),
                        ),
                    )
                })?;
            Some(compact_space_accessor)
        } else {
            None
        };
        let column_opt = accessor
            .as_physical_column()
            .map(|col| (col.clone(), column_type));
        Ok(Self {
            buckets: Vec::new(),
            name,
            accessor,
            column_opt,
            column_type,
            compact_space_accessor,
            missing_value_for_accessor,
        })
    }
}

impl SegmentAggregationCollector for SegmentNumericCardinalityCollector {
    fn add_intermediate_aggregation_result(
        &mut self,
        agg_data: &AggregationsSegmentCtx,
        results: &mut IntermediateAggregationResults,
        bucket_id: BucketId,
    ) -> crate::Result<()> {
        self.prepare_max_bucket(bucket_id, agg_data)?;
        let name = self.name.clone();
        let salt = self.column_type as u8;
        let cardinality = self.buckets[bucket_id as usize]
            .take_collector(salt, self.compact_space_accessor.as_deref())?;
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
        if let Some(column_source) = &mut self.column_opt {
            agg_data.column_block_accessor.fetch_block_with_missing(
                docs,
                column_source,
                self.missing_value_for_accessor,
            );
        } else {
            agg_data.column_block_accessor.fetch_block_with_missing(
                docs,
                &mut *self.accessor,
                self.missing_value_for_accessor,
            );
        }
        let bucket = &mut self.buckets[parent_bucket_id as usize];
        let salt = self.column_type as u8;
        let csa_opt = self.compact_space_accessor.as_deref();
        let vals = agg_data.column_block_accessor.values();

        // Fast loop when the bucket is already promoted to an HLL sketch (e.g. top-level
        // cardinality): keeps `last_val` in a register and avoids matching `BucketSketch` per row.
        if let BucketSketch::Sketch {
            collector,
            last_val,
        } = bucket
        {
            let mut prev = *last_val;
            if let Some(csa) = csa_opt {
                for &val in vals {
                    if prev == val {
                        continue;
                    }
                    prev = val;
                    collector.insert(csa.compact_to_u128(val as u32));
                }
            } else {
                for &val in vals {
                    if prev == val {
                        continue;
                    }
                    prev = val;
                    collector.insert(val);
                }
            }
            *last_val = prev;
            return Ok(());
        }

        for &val in vals {
            bucket.insert_raw(val, salt, csa_opt)?;
        }
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
        if let Some(column_source) = &mut self.column_opt {
            agg_data
                .column_block_accessor
                .fetch_block_with_missing_ordered(
                    docs,
                    column_source,
                    self.missing_value_for_accessor,
                    true,
                );
        } else {
            agg_data
                .column_block_accessor
                .fetch_block_with_missing_ordered(
                    docs,
                    &mut *self.accessor,
                    self.missing_value_for_accessor,
                    true,
                );
        }
        let salt = self.column_type as u8;
        let csa_opt = self.compact_space_accessor.as_deref();
        let col_block_accessor = &agg_data.column_block_accessor;
        let vals = col_block_accessor.values();
        if col_block_accessor.cardinality().is_full() {
            for (&bucket_id, &val) in bucket_ids.iter().zip(vals.iter()) {
                self.buckets[bucket_id as usize].insert_raw(val, salt, csa_opt)?;
            }
        } else {
            let fetched_docs = col_block_accessor.docids();
            let mut doc_pos = 0usize;
            for (&doc, &val) in fetched_docs.iter().zip(vals.iter()) {
                while docs[doc_pos] < doc {
                    doc_pos += 1;
                }
                let bucket_idx = bucket_ids[doc_pos] as usize;
                self.buckets[bucket_idx].insert_raw(val, salt, csa_opt)?;
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
            self.buckets.resize_with(required, || BucketSketch::Empty);
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
        if self.name != sub_agg_name || !sub_agg_property.is_empty() {
            return None;
        }
        self.buckets.get(bucket_id as usize)?.estimate()
    }
}

#[cfg(test)]
mod tests {
    use std::net::IpAddr;
    use std::str::FromStr;

    use crate::aggregation::agg_req::Aggregations;
    use crate::aggregation::tests::exec_request;
    use crate::schema::{IntoIpv6Addr, Schema, FAST};
    use crate::Index;

    #[test]
    fn cardinality_aggregation_u64() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let id_field = schema_builder.add_u64_field("id", FAST);
        let index = Index::create_in_ram(schema_builder.build());
        {
            let mut writer = index.writer_for_tests()?;
            writer.add_document(doc!(id_field => 1u64))?;
            writer.add_document(doc!(id_field => 2u64))?;
            writer.add_document(doc!(id_field => 3u64))?;
            writer.add_document(doc!())?;
            writer.commit()?;
        }

        let agg_req: Aggregations = serde_json::from_value(json!({
            "cardinality": {
                "cardinality": {
                    "field": "id",
                    "missing": 0u64
                },
            }
        }))
        .unwrap();

        let agg_res = exec_request(agg_req, &index)?;
        assert_eq!(agg_res["cardinality"]["value"], 4.0);
        Ok(())
    }

    #[test]
    fn cardinality_aggregation_ip_addr() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let ip_field = schema_builder.add_ip_addr_field("ip", FAST);
        let index = Index::create_in_ram(schema_builder.build());
        {
            let mut writer = index.writer_for_tests()?;
            writer.add_document(
                doc!(ip_field => IpAddr::from_str("127.0.0.1").unwrap().into_ipv6_addr()),
            )?;
            writer.add_document(
                doc!(ip_field => IpAddr::from_str("127.0.0.2").unwrap().into_ipv6_addr()),
            )?;
            writer.add_document(
                doc!(ip_field => IpAddr::from_str("127.0.0.1").unwrap().into_ipv6_addr()),
            )?;
            writer.commit()?;
        }

        let agg_req: Aggregations = serde_json::from_value(json!({
            "cardinality": {
                "cardinality": {
                    "field": "ip"
                },
            }
        }))
        .unwrap();

        let agg_res = exec_request(agg_req, &index)?;
        assert_eq!(agg_res["cardinality"]["value"], 2.0);
        Ok(())
    }
}
