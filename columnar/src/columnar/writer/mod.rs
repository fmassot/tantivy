mod column_operation;
mod column_writers;
mod serializer;
mod value_index;

use std::collections::{HashMap, HashSet};
use std::io;
use std::net::Ipv6Addr;

use column_operation::ColumnOperation;
pub(crate) use column_writers::CompatibleNumericalTypes;
use common::CountingWriter;
use common::json_path_writer::JSON_END_OF_PATH;
pub(crate) use serializer::ColumnarSerializer;
use stacker::{Addr, ArenaHashMap, MemoryArena};

use crate::column_index::{SerializableColumnIndex, SerializableOptionalIndex};
use crate::column_values::{MonotonicallyMappableToU64, MonotonicallyMappableToU128};
use crate::columnar::column_type::ColumnType;
use crate::columnar::writer::column_writers::{
    ColumnWriter, NumericalColumnWriter, StrOrBytesColumnWriter,
};
use crate::columnar::writer::value_index::{IndexBuilder, PreallocatedIndexBuilders};
use crate::dictionary::{DictionaryBuilder, TermIdMapping, UnorderedId};
use crate::value::{Coerce, NumericalType, NumericalValue};
use crate::{Cardinality, RowId};

#[derive(Clone, Copy, Debug)]
enum ColumnHandleInner {
    Arena(Addr),
    Dense(u32),
    DenseStr(u32),
}

/// Opaque direct handle to a column registered in a [`ColumnarWriter`].
///
/// Allows `FastFieldsWriter` to record values for static schema fields in O(1) without hashing
/// the column name string or probing an `ArenaHashMap` on every document.
#[derive(Clone, Copy, Debug)]
pub struct ColumnHandle(ColumnHandleInner);

impl ColumnHandle {
    #[inline(always)]
    fn arena_addr(self) -> Addr {
        match self.0 {
            ColumnHandleInner::Arena(addr) => addr,
            ColumnHandleInner::Dense(_) | ColumnHandleInner::DenseStr(_) => {
                panic!("ColumnHandle kind mismatch: expected arena-backed column handle")
            }
        }
    }
}

/// Flat contiguous column writer for statically-typed numerical and datetime columns
/// (`I64`, `U64`, `F64`, `DateTime`).
///
/// Stores monotonically-mapped `u64` values directly in a `Vec<u64>`, avoiding `ColumnOperation`
/// serialization into `ExpUnrolledLinkedList` and deserialization during `serialize()`.
/// For dense `Full` columns (1 value per doc `0..num_docs`), `doc_ids` stays empty and
/// `serialize()` passes `&self.values[..]` straight to the columnar encoder with zero copies.
struct DenseNumericalColumn {
    name: Box<[u8]>,
    column_type: ColumnType,
    cardinality: Cardinality,
    next_expected_doc: RowId,
    values: Vec<u64>,
    /// Empty while `cardinality == Cardinality::Full` (invariant: `values[i]` belongs to doc `i`).
    /// Lazily materialized on the first missing doc (Optional) or duplicate doc (Multivalued).
    doc_ids: Vec<RowId>,
}

impl DenseNumericalColumn {
    fn new(name: &[u8], column_type: ColumnType) -> Self {
        Self {
            name: name.into(),
            column_type,
            cardinality: Cardinality::Full,
            next_expected_doc: 0,
            values: Vec::new(),
            doc_ids: Vec::new(),
        }
    }

    fn mem_usage(&self) -> usize {
        self.name.len()
            + self.values.capacity() * std::mem::size_of::<u64>()
            + self.doc_ids.capacity() * std::mem::size_of::<RowId>()
    }

    #[inline(always)]
    fn record(&mut self, doc: RowId, val_u64: u64) {
        if doc == self.next_expected_doc && self.cardinality == Cardinality::Full {
            self.next_expected_doc = doc + 1;
            self.values.push(val_u64);
            return;
        }
        self.record_non_dense(doc, val_u64);
    }

    #[inline]
    fn record_non_dense(&mut self, doc: RowId, val_u64: u64) {
        if self.cardinality == Cardinality::Full {
            let num_existing = self.values.len() as RowId;
            self.doc_ids.reserve(self.values.len() + 1);
            self.doc_ids.extend(0..num_existing);
        }
        match doc.cmp(&self.next_expected_doc) {
            std::cmp::Ordering::Less => {
                self.cardinality = Cardinality::Multivalued;
            }
            std::cmp::Ordering::Equal => {
                self.next_expected_doc = doc + 1;
            }
            std::cmp::Ordering::Greater => {
                self.cardinality = self.cardinality.max(Cardinality::Optional);
                self.next_expected_doc = doc + 1;
            }
        }
        self.doc_ids.push(doc);
        self.values.push(val_u64);
    }

    #[inline]
    fn get_cardinality(&self, num_docs: RowId) -> Cardinality {
        if num_docs > self.next_expected_doc {
            self.cardinality.max(Cardinality::Optional)
        } else {
            self.cardinality
        }
    }

    fn sort_order(&self, num_docs: RowId, reversed: bool) -> Vec<u32> {
        let mut doc_sort_keys: Vec<(Option<u64>, RowId)> = Vec::with_capacity(num_docs as usize);
        if self.cardinality == Cardinality::Full {
            for (doc_idx, &val) in self.values.iter().enumerate() {
                doc_sort_keys.push((Some(val), doc_idx as RowId));
            }
            for doc in (self.values.len() as RowId)..num_docs {
                doc_sort_keys.push((None, doc));
            }
        } else {
            let mut next_fill_doc: RowId = 0;
            let mut last_seen_doc: Option<RowId> = None;
            for (&doc, &val) in self.doc_ids.iter().zip(self.values.iter()) {
                if last_seen_doc == Some(doc) {
                    // For multivalued fields, only the first value is used for sorting.
                    continue;
                }
                last_seen_doc = Some(doc);
                for missing_doc in next_fill_doc..doc {
                    doc_sort_keys.push((None, missing_doc));
                }
                doc_sort_keys.push((Some(val), doc));
                next_fill_doc = doc + 1;
            }
            for missing_doc in next_fill_doc..num_docs {
                doc_sort_keys.push((None, missing_doc));
            }
        }
        doc_sort_keys.sort_by(|(left_key, _), (right_key, _)| {
            let cmp = left_key.cmp(right_key);
            if reversed { cmp.reverse() } else { cmp }
        });
        doc_sort_keys.into_iter().map(|(_, doc)| doc).collect()
    }

    fn serialize(
        &self,
        num_docs: RowId,
        old_to_new_row_ids: Option<&[RowId]>,
        buffers: &mut SpareBuffers,
        wrt: &mut impl io::Write,
    ) -> io::Result<()> {
        let effective_cardinality = self.get_cardinality(num_docs);
        if let Some(old_to_new) = old_to_new_row_ids {
            return self.serialize_remapped(
                num_docs,
                effective_cardinality,
                old_to_new,
                buffers,
                wrt,
            );
        }
        match effective_cardinality {
            Cardinality::Full => crate::column::serialize_column_u64_slice(
                SerializableColumnIndex::Full,
                &self.values[..],
                wrt,
            ),
            Cardinality::Optional => {
                if self.cardinality == Cardinality::Full {
                    // Gaps only occur at the tail (`self.values.len()..num_docs`).
                    let optional_index_builder =
                        buffers.value_index_builders.borrow_optional_index_builder();
                    for doc in 0..(self.values.len() as RowId) {
                        optional_index_builder.record_row(doc);
                    }
                    let optional_index = optional_index_builder.finish(num_docs);
                    crate::column::serialize_column_u64_slice(
                        SerializableColumnIndex::Optional(SerializableOptionalIndex {
                            non_null_row_ids: Box::new(optional_index),
                            num_rows: num_docs,
                        }),
                        &self.values[..],
                        wrt,
                    )
                } else {
                    crate::column::serialize_column_u64_slice(
                        SerializableColumnIndex::Optional(SerializableOptionalIndex {
                            non_null_row_ids: Box::new(&self.doc_ids[..]),
                            num_rows: num_docs,
                        }),
                        &self.values[..],
                        wrt,
                    )
                }
            }
            Cardinality::Multivalued => {
                let mv_builder = buffers
                    .value_index_builders
                    .borrow_multivalued_index_builder();
                let mut prev_doc: Option<RowId> = None;
                for &doc in &self.doc_ids {
                    if prev_doc != Some(doc) {
                        mv_builder.record_row(doc);
                        prev_doc = Some(doc);
                    }
                    mv_builder.record_value();
                }
                let mv_index = mv_builder.finish(num_docs);
                crate::column::serialize_column_u64_slice(
                    SerializableColumnIndex::Multivalued(mv_index),
                    &self.values[..],
                    wrt,
                )
            }
        }
    }

    fn serialize_remapped(
        &self,
        num_docs: RowId,
        effective_cardinality: Cardinality,
        old_to_new: &[RowId],
        buffers: &mut SpareBuffers,
        wrt: &mut impl io::Write,
    ) -> io::Result<()> {
        if effective_cardinality == Cardinality::Full {
            // `old_to_new` is a permutation of `0..num_docs`; scatter directly in O(N).
            buffers.u64_values.resize(num_docs as usize, 0);
            for (old_doc, &val) in self.values.iter().enumerate() {
                let new_doc = old_to_new[old_doc] as usize;
                buffers.u64_values[new_doc] = val;
            }
            return crate::column::serialize_column_u64_slice(
                SerializableColumnIndex::Full,
                &buffers.u64_values[..],
                wrt,
            );
        }
        let mut pairs: Vec<(RowId, u64)> = Vec::with_capacity(self.values.len());
        if self.cardinality == Cardinality::Full {
            for (old_doc, &val) in self.values.iter().enumerate() {
                pairs.push((old_to_new[old_doc], val));
            }
        } else {
            for (&old_doc, &val) in self.doc_ids.iter().zip(self.values.iter()) {
                pairs.push((old_to_new[old_doc as usize], val));
            }
        }
        // Stable sort preserves within-doc value ordering for multivalued columns.
        pairs.sort_by_key(|&(new_doc, _)| new_doc);
        buffers.u64_values.clear();
        buffers.u64_values.reserve(pairs.len());
        match effective_cardinality {
            Cardinality::Full => unreachable!(),
            Cardinality::Optional => {
                let optional_index_builder =
                    buffers.value_index_builders.borrow_optional_index_builder();
                for (new_doc, val) in pairs {
                    optional_index_builder.record_row(new_doc);
                    buffers.u64_values.push(val);
                }
                let optional_index = optional_index_builder.finish(num_docs);
                crate::column::serialize_column_u64_slice(
                    SerializableColumnIndex::Optional(SerializableOptionalIndex {
                        non_null_row_ids: Box::new(optional_index),
                        num_rows: num_docs,
                    }),
                    &buffers.u64_values[..],
                    wrt,
                )
            }
            Cardinality::Multivalued => {
                let mv_builder = buffers
                    .value_index_builders
                    .borrow_multivalued_index_builder();
                let mut prev_doc: Option<RowId> = None;
                for (new_doc, val) in pairs {
                    if prev_doc != Some(new_doc) {
                        mv_builder.record_row(new_doc);
                        prev_doc = Some(new_doc);
                    }
                    mv_builder.record_value();
                    buffers.u64_values.push(val);
                }
                let mv_index = mv_builder.finish(num_docs);
                crate::column::serialize_column_u64_slice(
                    SerializableColumnIndex::Multivalued(mv_index),
                    &buffers.u64_values[..],
                    wrt,
                )
            }
        }
    }
}

#[inline(always)]
fn accept_and_convert_numerical(column_type: ColumnType, value: NumericalValue) -> u64 {
    match (column_type, value) {
        (ColumnType::I64, NumericalValue::I64(val)) => val.to_u64(),
        (ColumnType::U64, NumericalValue::U64(val)) => val.to_u64(),
        (ColumnType::F64, NumericalValue::F64(val)) => val.to_u64(),
        _ => panic!(
            "Input type forbidden. This column has been forced to type {column_type:?}, received \
             {value:?}"
        ),
    }
}

/// Flat contiguous column writer for statically-typed `Str` and `Bytes` columns.
///
/// Stores `UnorderedId`s directly in a `Vec<UnorderedId>`, bypassing `ColumnOperation`
/// serialization into `ExpUnrolledLinkedList` and deserialization during `serialize()`.
/// Also caches the `UnorderedId` of the empty slice `b""` to avoid hashing and probing the
/// dictionary on sparse string columns.
struct DenseStrOrBytesColumn {
    name: Box<[u8]>,
    column_type: ColumnType,
    dictionary_id: u32,
    sort_values_within_row: bool,
    cardinality: Cardinality,
    next_expected_doc: RowId,
    empty_bytes_id: Option<UnorderedId>,
    unordered_ids: Vec<UnorderedId>,
    /// Empty while `cardinality == Cardinality::Full` (invariant: `unordered_ids[i]` -> doc `i`).
    doc_ids: Vec<RowId>,
}

impl DenseStrOrBytesColumn {
    fn new(
        name: &[u8],
        column_type: ColumnType,
        dictionary_id: u32,
        sort_values_within_row: bool,
    ) -> Self {
        Self {
            name: name.into(),
            column_type,
            dictionary_id,
            sort_values_within_row,
            cardinality: Cardinality::Full,
            next_expected_doc: 0,
            empty_bytes_id: None,
            unordered_ids: Vec::new(),
            doc_ids: Vec::new(),
        }
    }

    fn mem_usage(&self) -> usize {
        self.name.len()
            + self.unordered_ids.capacity() * std::mem::size_of::<UnorderedId>()
            + self.doc_ids.capacity() * std::mem::size_of::<RowId>()
    }

    #[inline(always)]
    fn record_bytes(
        &mut self,
        doc: RowId,
        bytes: &[u8],
        dictionaries: &mut [DictionaryBuilder],
        arena: &mut MemoryArena,
    ) {
        let unordered_id = if bytes.is_empty() {
            match self.empty_bytes_id {
                Some(id) => id,
                None => {
                    let id =
                        dictionaries[self.dictionary_id as usize].get_or_allocate_id(bytes, arena);
                    self.empty_bytes_id = Some(id);
                    id
                }
            }
        } else {
            dictionaries[self.dictionary_id as usize].get_or_allocate_id(bytes, arena)
        };
        if doc == self.next_expected_doc && self.cardinality == Cardinality::Full {
            self.next_expected_doc = doc + 1;
            self.unordered_ids.push(unordered_id);
            return;
        }
        self.record_non_dense(doc, unordered_id);
    }

    #[inline]
    fn record_non_dense(&mut self, doc: RowId, unordered_id: UnorderedId) {
        if self.cardinality == Cardinality::Full {
            let num_existing = self.unordered_ids.len() as RowId;
            self.doc_ids.reserve(self.unordered_ids.len() + 1);
            self.doc_ids.extend(0..num_existing);
        }
        match doc.cmp(&self.next_expected_doc) {
            std::cmp::Ordering::Less => {
                self.cardinality = Cardinality::Multivalued;
            }
            std::cmp::Ordering::Equal => {
                self.next_expected_doc = doc + 1;
            }
            std::cmp::Ordering::Greater => {
                self.cardinality = self.cardinality.max(Cardinality::Optional);
                self.next_expected_doc = doc + 1;
            }
        }
        self.doc_ids.push(doc);
        self.unordered_ids.push(unordered_id);
    }

    #[inline]
    fn get_cardinality(&self, num_docs: RowId) -> Cardinality {
        if num_docs > self.next_expected_doc {
            self.cardinality.max(Cardinality::Optional)
        } else {
            self.cardinality
        }
    }

    fn sort_order(
        &self,
        num_docs: RowId,
        reversed: bool,
        term_id_mapping: &TermIdMapping,
    ) -> Vec<u32> {
        let mut doc_sort_keys: Vec<(Option<u32>, RowId)> = Vec::with_capacity(num_docs as usize);
        if self.cardinality == Cardinality::Full {
            for (doc_idx, &uid) in self.unordered_ids.iter().enumerate() {
                doc_sort_keys.push((Some(term_id_mapping.to_ord(uid).0), doc_idx as RowId));
            }
            for doc in (self.unordered_ids.len() as RowId)..num_docs {
                doc_sort_keys.push((None, doc));
            }
        } else {
            let mut next_fill_doc: RowId = 0;
            let mut last_seen_doc: Option<RowId> = None;
            for (&doc, &uid) in self.doc_ids.iter().zip(self.unordered_ids.iter()) {
                if last_seen_doc == Some(doc) {
                    continue;
                }
                last_seen_doc = Some(doc);
                for missing_doc in next_fill_doc..doc {
                    doc_sort_keys.push((None, missing_doc));
                }
                doc_sort_keys.push((Some(term_id_mapping.to_ord(uid).0), doc));
                next_fill_doc = doc + 1;
            }
            for missing_doc in next_fill_doc..num_docs {
                doc_sort_keys.push((None, missing_doc));
            }
        }
        doc_sort_keys.sort_by(|(left_key, _), (right_key, _)| {
            let cmp = left_key.cmp(right_key);
            if reversed { cmp.reverse() } else { cmp }
        });
        doc_sort_keys.into_iter().map(|(_, doc)| doc).collect()
    }

    fn serialize(
        &self,
        num_docs: RowId,
        old_to_new_row_ids: Option<&[RowId]>,
        dictionary_builder: &DictionaryBuilder,
        buffers: &mut SpareBuffers,
        arena: &MemoryArena,
        wrt: impl io::Write,
    ) -> io::Result<()> {
        let mut counting_writer = CountingWriter::wrap(wrt);
        let term_id_mapping: TermIdMapping =
            dictionary_builder.serialize(arena, &mut counting_writer)?;
        let dictionary_num_bytes: u32 = counting_writer.written_bytes() as u32;
        let mut wrt = counting_writer.finish();

        let effective_cardinality = self.get_cardinality(num_docs);
        if let Some(old_to_new) = old_to_new_row_ids {
            self.serialize_values_remapped(
                num_docs,
                effective_cardinality,
                old_to_new,
                &term_id_mapping,
                buffers,
                &mut wrt,
            )?;
        } else {
            buffers.u64_values.clear();
            buffers.u64_values.reserve(self.unordered_ids.len());
            for &uid in &self.unordered_ids {
                buffers
                    .u64_values
                    .push(term_id_mapping.to_ord(uid).0 as u64);
            }
            match effective_cardinality {
                Cardinality::Full => {
                    crate::column::serialize_column_u64_slice(
                        SerializableColumnIndex::Full,
                        &buffers.u64_values[..],
                        &mut wrt,
                    )?;
                }
                Cardinality::Optional => {
                    if self.cardinality == Cardinality::Full {
                        let optional_index_builder =
                            buffers.value_index_builders.borrow_optional_index_builder();
                        for doc in 0..(self.unordered_ids.len() as RowId) {
                            optional_index_builder.record_row(doc);
                        }
                        let optional_index = optional_index_builder.finish(num_docs);
                        crate::column::serialize_column_u64_slice(
                            SerializableColumnIndex::Optional(SerializableOptionalIndex {
                                non_null_row_ids: Box::new(optional_index),
                                num_rows: num_docs,
                            }),
                            &buffers.u64_values[..],
                            &mut wrt,
                        )?;
                    } else {
                        crate::column::serialize_column_u64_slice(
                            SerializableColumnIndex::Optional(SerializableOptionalIndex {
                                non_null_row_ids: Box::new(&self.doc_ids[..]),
                                num_rows: num_docs,
                            }),
                            &buffers.u64_values[..],
                            &mut wrt,
                        )?;
                    }
                }
                Cardinality::Multivalued => {
                    let mv_builder = buffers
                        .value_index_builders
                        .borrow_multivalued_index_builder();
                    let mut prev_doc: Option<RowId> = None;
                    for &doc in &self.doc_ids {
                        if prev_doc != Some(doc) {
                            mv_builder.record_row(doc);
                            prev_doc = Some(doc);
                        }
                        mv_builder.record_value();
                    }
                    let mv_index = mv_builder.finish(num_docs);
                    if self.sort_values_within_row {
                        sort_values_within_row_in_place(
                            mv_index.start_offsets.boxed_iter(),
                            &mut buffers.u64_values,
                        );
                    }
                    crate::column::serialize_column_u64_slice(
                        SerializableColumnIndex::Multivalued(mv_index),
                        &buffers.u64_values[..],
                        &mut wrt,
                    )?;
                }
            }
        }
        wrt.write_all(&dictionary_num_bytes.to_le_bytes()[..])?;
        Ok(())
    }

    fn serialize_values_remapped(
        &self,
        num_docs: RowId,
        effective_cardinality: Cardinality,
        old_to_new: &[RowId],
        term_id_mapping: &TermIdMapping,
        buffers: &mut SpareBuffers,
        wrt: &mut impl io::Write,
    ) -> io::Result<()> {
        if effective_cardinality == Cardinality::Full {
            buffers.u64_values.resize(num_docs as usize, 0);
            for (old_doc, &uid) in self.unordered_ids.iter().enumerate() {
                let new_doc = old_to_new[old_doc] as usize;
                buffers.u64_values[new_doc] = term_id_mapping.to_ord(uid).0 as u64;
            }
            return crate::column::serialize_column_u64_slice(
                SerializableColumnIndex::Full,
                &buffers.u64_values[..],
                wrt,
            );
        }
        let mut pairs: Vec<(RowId, u64)> = Vec::with_capacity(self.unordered_ids.len());
        if self.cardinality == Cardinality::Full {
            for (old_doc, &uid) in self.unordered_ids.iter().enumerate() {
                pairs.push((old_to_new[old_doc], term_id_mapping.to_ord(uid).0 as u64));
            }
        } else {
            for (&old_doc, &uid) in self.doc_ids.iter().zip(self.unordered_ids.iter()) {
                pairs.push((
                    old_to_new[old_doc as usize],
                    term_id_mapping.to_ord(uid).0 as u64,
                ));
            }
        }
        pairs.sort_by_key(|&(new_doc, _)| new_doc);
        buffers.u64_values.clear();
        buffers.u64_values.reserve(pairs.len());
        match effective_cardinality {
            Cardinality::Full => unreachable!(),
            Cardinality::Optional => {
                let optional_index_builder =
                    buffers.value_index_builders.borrow_optional_index_builder();
                for (new_doc, val) in pairs {
                    optional_index_builder.record_row(new_doc);
                    buffers.u64_values.push(val);
                }
                let optional_index = optional_index_builder.finish(num_docs);
                crate::column::serialize_column_u64_slice(
                    SerializableColumnIndex::Optional(SerializableOptionalIndex {
                        non_null_row_ids: Box::new(optional_index),
                        num_rows: num_docs,
                    }),
                    &buffers.u64_values[..],
                    wrt,
                )
            }
            Cardinality::Multivalued => {
                let mv_builder = buffers
                    .value_index_builders
                    .borrow_multivalued_index_builder();
                let mut prev_doc: Option<RowId> = None;
                for (new_doc, val) in pairs {
                    if prev_doc != Some(new_doc) {
                        mv_builder.record_row(new_doc);
                        prev_doc = Some(new_doc);
                    }
                    mv_builder.record_value();
                    buffers.u64_values.push(val);
                }
                let mv_index = mv_builder.finish(num_docs);
                if self.sort_values_within_row {
                    sort_values_within_row_in_place(
                        mv_index.start_offsets.boxed_iter(),
                        &mut buffers.u64_values,
                    );
                }
                crate::column::serialize_column_u64_slice(
                    SerializableColumnIndex::Multivalued(mv_index),
                    &buffers.u64_values[..],
                    wrt,
                )
            }
        }
    }
}

#[derive(Clone, Copy)]
enum ColumnSource {
    Arena(Addr),
    Dense(u32),
    DenseStr(u32),
}

/// This is a set of buffers that are used to temporarily write the values into before passing them
/// to the fast field codecs.
#[derive(Default)]
struct SpareBuffers {
    value_index_builders: PreallocatedIndexBuilders,
    u64_values: Vec<u64>,
    ip_addr_values: Vec<Ipv6Addr>,
}

/// Makes it possible to create a new columnar.
///
/// ```rust
/// use tantivy_columnar::ColumnarWriter;
///
/// let mut columnar_writer = ColumnarWriter::default();
/// columnar_writer.record_str(0u32 /* doc id */, "product_name", "Red backpack");
/// columnar_writer.record_numerical(0u32 /* doc id */, "price", 10u64);
/// columnar_writer.record_str(1u32 /* doc id */, "product_name", "Apple");
/// columnar_writer.record_numerical(0u32 /* doc id */, "price", 10.5f64); //< uh oh we ended up mixing integer and floats.
/// let mut wrt: Vec<u8> =  Vec::new();
/// columnar_writer.serialize(2u32, None, &mut wrt).unwrap();
/// ```
#[derive(Default)]
pub struct ColumnarWriter {
    numerical_field_hash_map: ArenaHashMap,
    datetime_field_hash_map: ArenaHashMap,
    bool_field_hash_map: ArenaHashMap,
    ip_addr_field_hash_map: ArenaHashMap,
    bytes_field_hash_map: ArenaHashMap,
    str_field_hash_map: ArenaHashMap,
    /// Statically-typed numerical and datetime columns stored in flat `Vec<u64>` buffers.
    dense_columns: Vec<DenseNumericalColumn>,
    dense_numerical_by_name: HashMap<Box<[u8]>, u32>,
    dense_datetime_by_name: HashMap<Box<[u8]>, u32>,
    /// Statically-typed Str and Bytes columns stored in flat `Vec<UnorderedId>` buffers.
    dense_str_columns: Vec<DenseStrOrBytesColumn>,
    dense_str_by_name: HashMap<Box<[u8]>, u32>,
    dense_bytes_by_name: HashMap<Box<[u8]>, u32>,
    generated_tie_breaker_columns: HashSet<Vec<u8>>,
    arena: MemoryArena,
    // Dictionaries used to store dictionary-encoded values.
    dictionaries: Vec<DictionaryBuilder>,
    buffers: SpareBuffers,
}

impl ColumnarWriter {
    pub fn mem_usage(&self) -> usize {
        self.arena.mem_usage()
            + self.numerical_field_hash_map.mem_usage()
            + self.bool_field_hash_map.mem_usage()
            + self.bytes_field_hash_map.mem_usage()
            + self.str_field_hash_map.mem_usage()
            + self.ip_addr_field_hash_map.mem_usage()
            + self.datetime_field_hash_map.mem_usage()
            + self
                .dense_columns
                .iter()
                .map(DenseNumericalColumn::mem_usage)
                .sum::<usize>()
            + self
                .dense_str_columns
                .iter()
                .map(DenseStrOrBytesColumn::mem_usage)
                .sum::<usize>()
            + self
                .dictionaries
                .iter()
                .map(|dict| dict.mem_usage())
                .sum::<usize>()
    }

    /// Returns the list of doc ids from 0..num_docs sorted by the `sort_field`
    /// column.
    ///
    /// If the column is multivalued, use the first value for scoring.
    /// If no value is associated to a specific row, the document is assigned
    /// the lowest possible score.
    ///
    /// The sort applied is stable.
    ///
    /// The sort_field has to be a schema field (not a json path): we rely on the fact that
    /// all values have the same numerical type in `collect_numerical_sort_order`.
    pub fn sort_order(&self, sort_field: &str, num_docs: RowId, reversed: bool) -> Vec<u32> {
        let sort_field_bytes = sort_field.as_bytes();
        if let Some(&dense_idx) = self.dense_numerical_by_name.get(sort_field_bytes) {
            return self.dense_columns[dense_idx as usize].sort_order(num_docs, reversed);
        }
        let mut symbols_buffer = Vec::new();
        if let Some(numerical_col_writer) = self
            .numerical_field_hash_map
            .get::<NumericalColumnWriter>(sort_field_bytes)
        {
            return collect_numerical_sort_order(
                numerical_col_writer.operation_iterator(&self.arena, None, &mut symbols_buffer),
                num_docs,
                reversed,
            );
        }
        if let Some(&dense_idx) = self.dense_datetime_by_name.get(sort_field_bytes) {
            return self.dense_columns[dense_idx as usize].sort_order(num_docs, reversed);
        }
        // Datetime columns store a plain `ColumnWriter` (not a `NumericalColumnWriter`), with
        // `NumericalValue::I64` values.
        if let Some(datetime_col_writer) = self
            .datetime_field_hash_map
            .get::<ColumnWriter>(sort_field_bytes)
        {
            return collect_numerical_sort_order(
                datetime_col_writer.operation_iterator::<NumericalValue>(
                    &self.arena,
                    None,
                    &mut symbols_buffer,
                ),
                num_docs,
                reversed,
            );
        }
        if let Some(&dense_idx) = self
            .dense_str_by_name
            .get(sort_field_bytes)
            .or_else(|| self.dense_bytes_by_name.get(sort_field_bytes))
        {
            let col = &self.dense_str_columns[dense_idx as usize];
            let dictionary_builder = &self.dictionaries[col.dictionary_id as usize];
            let term_id_mapping = dictionary_builder.build_term_id_mapping(&self.arena);
            return col.sort_order(num_docs, reversed, &term_id_mapping);
        }
        let str_or_bytes_column_opt = self
            .str_field_hash_map
            .get::<StrOrBytesColumnWriter>(sort_field_bytes)
            .or_else(|| {
                self.bytes_field_hash_map
                    .get::<StrOrBytesColumnWriter>(sort_field_bytes)
            });
        let Some(str_or_bytes_column) = str_or_bytes_column_opt else {
            return Vec::new();
        };
        let dictionary_builder = &self.dictionaries[str_or_bytes_column.dictionary_id as usize];
        let term_id_mapping = dictionary_builder.build_term_id_mapping(&self.arena);
        collect_sort_order_from_ops(
            str_or_bytes_column.operation_iterator(&self.arena, None, &mut symbols_buffer),
            num_docs,
            reversed,
            |uid| Some(term_id_mapping.to_ord(uid).0),
            None,
            |a, b| a.cmp(b),
        )
    }

    /// Records a column type. This is useful to bypass the coercion process,
    /// makes sure the empty is present in the resulting columnar, or set
    /// the `sort_values_within_row`.
    ///
    /// `sort_values_within_row` is only allowed for `Bytes` or `Str` columns.
    pub fn record_column_type(
        &mut self,
        column_name: &str,
        column_type: ColumnType,
        sort_values_within_row: bool,
    ) {
        let _ = self.record_column_type_with_handle(
            column_name,
            column_type,
            sort_values_within_row,
        );
    }

    /// Same as [`Self::record_column_type`], returning a [`ColumnHandle`] for direct O(1)
    /// recording via `record_*_by_handle`.
    ///
    /// Contract: the returned handle must only be passed to the `record_*_by_handle` method
    /// matching `column_type` on this exact `ColumnarWriter` instance.
    pub fn record_column_type_with_handle(
        &mut self,
        column_name: &str,
        column_type: ColumnType,
        sort_values_within_row: bool,
    ) -> ColumnHandle {
        if sort_values_within_row {
            assert!(
                column_type == ColumnType::Bytes || column_type == ColumnType::Str,
                "sort_values_within_row is only allowed for Bytes and Str columns",
            );
        }
        let name_bytes = column_name.as_bytes();
        let inner = match column_type {
            ColumnType::Str | ColumnType::Bytes => {
                let is_str = column_type == ColumnType::Str;
                let existing_dense = if is_str {
                    self.dense_str_by_name.get(name_bytes).copied()
                } else {
                    self.dense_bytes_by_name.get(name_bytes).copied()
                };
                if let Some(idx) = existing_dense {
                    self.dense_str_columns[idx as usize].sort_values_within_row =
                        sort_values_within_row;
                    ColumnHandleInner::DenseStr(idx)
                } else {
                    let existing_arena = if is_str {
                        self.str_field_hash_map
                            .get::<StrOrBytesColumnWriter>(name_bytes)
                            .is_some()
                    } else {
                        self.bytes_field_hash_map
                            .get::<StrOrBytesColumnWriter>(name_bytes)
                            .is_some()
                    };
                    if existing_arena {
                        let (hash_map, dictionaries) = (
                            if is_str {
                                &mut self.str_field_hash_map
                            } else {
                                &mut self.bytes_field_hash_map
                            },
                            &mut self.dictionaries,
                        );
                        let addr = hash_map.mutate_or_create_with_addr(
                            name_bytes,
                            |column_opt: Option<StrOrBytesColumnWriter>| {
                                let mut column_writer = if let Some(column_writer) = column_opt {
                                    column_writer
                                } else {
                                    let dictionary_id = dictionaries.len() as u32;
                                    dictionaries.push(DictionaryBuilder::default());
                                    StrOrBytesColumnWriter::with_dictionary_id(dictionary_id)
                                };
                                column_writer.sort_values_within_row = sort_values_within_row;
                                column_writer
                            },
                        );
                        ColumnHandleInner::Arena(addr)
                    } else {
                        let dictionary_id = self.dictionaries.len() as u32;
                        self.dictionaries.push(DictionaryBuilder::default());
                        let idx = self.dense_str_columns.len() as u32;
                        self.dense_str_columns.push(DenseStrOrBytesColumn::new(
                            name_bytes,
                            column_type,
                            dictionary_id,
                            sort_values_within_row,
                        ));
                        if is_str {
                            self.dense_str_by_name.insert(name_bytes.into(), idx);
                        } else {
                            self.dense_bytes_by_name.insert(name_bytes.into(), idx);
                        }
                        ColumnHandleInner::DenseStr(idx)
                    }
                }
            }
            ColumnType::Bool => ColumnHandleInner::Arena(
                self.bool_field_hash_map
                    .mutate_or_create_with_addr(name_bytes, |column_opt: Option<ColumnWriter>| {
                        column_opt.unwrap_or_default()
                    }),
            ),
            ColumnType::DateTime => {
                if let Some(&idx) = self.dense_datetime_by_name.get(name_bytes) {
                    ColumnHandleInner::Dense(idx)
                } else if self
                    .datetime_field_hash_map
                    .get::<ColumnWriter>(name_bytes)
                    .is_some()
                {
                    let addr = self.datetime_field_hash_map.mutate_or_create_with_addr(
                        name_bytes,
                        |column_opt: Option<ColumnWriter>| column_opt.unwrap_or_default(),
                    );
                    ColumnHandleInner::Arena(addr)
                } else {
                    let idx = self.dense_columns.len() as u32;
                    self.dense_columns
                        .push(DenseNumericalColumn::new(name_bytes, ColumnType::DateTime));
                    self.dense_datetime_by_name.insert(name_bytes.into(), idx);
                    ColumnHandleInner::Dense(idx)
                }
            }
            ColumnType::I64 | ColumnType::F64 | ColumnType::U64 => {
                if let Some(&idx) = self.dense_numerical_by_name.get(name_bytes) {
                    assert_eq!(
                        self.dense_columns[idx as usize].column_type, column_type,
                        "Column `{column_name}` already forced to type {:?}",
                        self.dense_columns[idx as usize].column_type
                    );
                    ColumnHandleInner::Dense(idx)
                } else if self
                    .numerical_field_hash_map
                    .get::<NumericalColumnWriter>(name_bytes)
                    .is_some()
                {
                    let numerical_type = column_type.numerical_type().unwrap();
                    let addr = self.numerical_field_hash_map.mutate_or_create_with_addr(
                        name_bytes,
                        |column_opt: Option<NumericalColumnWriter>| {
                            let mut column: NumericalColumnWriter = column_opt.unwrap_or_default();
                            column.force_numerical_type(numerical_type);
                            column
                        },
                    );
                    ColumnHandleInner::Arena(addr)
                } else {
                    let idx = self.dense_columns.len() as u32;
                    self.dense_columns
                        .push(DenseNumericalColumn::new(name_bytes, column_type));
                    self.dense_numerical_by_name.insert(name_bytes.into(), idx);
                    ColumnHandleInner::Dense(idx)
                }
            }
            ColumnType::IpAddr => ColumnHandleInner::Arena(
                self.ip_addr_field_hash_map
                    .mutate_or_create_with_addr(name_bytes, |column_opt: Option<ColumnWriter>| {
                        column_opt.unwrap_or_default()
                    }),
            ),
        };
        ColumnHandle(inner)
    }

    /// Registers a full `i64` column whose values are generated when this writer is serialized.
    pub fn record_tie_breaker_column(&mut self, column_name: &str) {
        self.record_column_type(column_name, ColumnType::I64, false);
        self.generated_tie_breaker_columns
            .insert(column_name.as_bytes().to_vec());
    }

    pub fn record_numerical<T: Into<NumericalValue> + Copy>(
        &mut self,
        doc: RowId,
        column_name: &str,
        numerical_value: T,
    ) {
        if let Some(&idx) = self.dense_numerical_by_name.get(column_name.as_bytes()) {
            let col = &mut self.dense_columns[idx as usize];
            let val_u64 = accept_and_convert_numerical(col.column_type, numerical_value.into());
            col.record(doc, val_u64);
            return;
        }
        let (hash_map, arena) = (&mut self.numerical_field_hash_map, &mut self.arena);
        hash_map.mutate_or_create(
            column_name.as_bytes(),
            |column_opt: Option<NumericalColumnWriter>| {
                let mut column: NumericalColumnWriter = column_opt.unwrap_or_default();
                column.record_numerical_value(doc, numerical_value.into(), arena);
                column
            },
        );
    }

    pub fn record_ip_addr(&mut self, doc: RowId, column_name: &str, ip_addr: Ipv6Addr) {
        let (hash_map, arena) = (&mut self.ip_addr_field_hash_map, &mut self.arena);
        hash_map.mutate_or_create(
            column_name.as_bytes(),
            |column_opt: Option<ColumnWriter>| {
                let mut column: ColumnWriter = column_opt.unwrap_or_default();
                column.record(doc, ip_addr, arena);
                column
            },
        );
    }

    pub fn record_bool(&mut self, doc: RowId, column_name: &str, val: bool) {
        let (hash_map, arena) = (&mut self.bool_field_hash_map, &mut self.arena);
        hash_map.mutate_or_create(
            column_name.as_bytes(),
            |column_opt: Option<ColumnWriter>| {
                let mut column: ColumnWriter = column_opt.unwrap_or_default();
                column.record(doc, val, arena);
                column
            },
        );
    }

    pub fn record_datetime(&mut self, doc: RowId, column_name: &str, datetime: common::DateTime) {
        if let Some(&idx) = self.dense_datetime_by_name.get(column_name.as_bytes()) {
            let col = &mut self.dense_columns[idx as usize];
            col.record(doc, datetime.into_timestamp_nanos().to_u64());
            return;
        }
        let (hash_map, arena) = (&mut self.datetime_field_hash_map, &mut self.arena);
        hash_map.mutate_or_create(
            column_name.as_bytes(),
            |column_opt: Option<ColumnWriter>| {
                let mut column: ColumnWriter = column_opt.unwrap_or_default();
                column.record(
                    doc,
                    NumericalValue::I64(datetime.into_timestamp_nanos()),
                    arena,
                );
                column
            },
        );
    }

    pub fn record_str(&mut self, doc: RowId, column_name: &str, value: &str) {
        if let Some(&idx) = self.dense_str_by_name.get(column_name.as_bytes()) {
            self.dense_str_columns[idx as usize].record_bytes(
                doc,
                value.as_bytes(),
                &mut self.dictionaries,
                &mut self.arena,
            );
            return;
        }
        let (hash_map, arena, dictionaries) = (
            &mut self.str_field_hash_map,
            &mut self.arena,
            &mut self.dictionaries,
        );
        hash_map.mutate_or_create(
            column_name.as_bytes(),
            |column_opt: Option<StrOrBytesColumnWriter>| {
                let mut column: StrOrBytesColumnWriter = column_opt.unwrap_or_else(|| {
                    // Each column has its own dictionary
                    let dictionary_id = dictionaries.len() as u32;
                    dictionaries.push(DictionaryBuilder::default());
                    StrOrBytesColumnWriter::with_dictionary_id(dictionary_id)
                });
                column.record_bytes(doc, value.as_bytes(), dictionaries, arena);
                column
            },
        );
    }

    pub fn record_bytes(&mut self, doc: RowId, column_name: &str, value: &[u8]) {
        if let Some(&idx) = self.dense_bytes_by_name.get(column_name.as_bytes()) {
            self.dense_str_columns[idx as usize].record_bytes(
                doc,
                value,
                &mut self.dictionaries,
                &mut self.arena,
            );
            return;
        }
        let (hash_map, arena, dictionaries) = (
            &mut self.bytes_field_hash_map,
            &mut self.arena,
            &mut self.dictionaries,
        );
        hash_map.mutate_or_create(
            column_name.as_bytes(),
            |column_opt: Option<StrOrBytesColumnWriter>| {
                let mut column: StrOrBytesColumnWriter = column_opt.unwrap_or_else(|| {
                    // Each column has its own dictionary
                    let dictionary_id = dictionaries.len() as u32;
                    dictionaries.push(DictionaryBuilder::default());
                    StrOrBytesColumnWriter::with_dictionary_id(dictionary_id)
                });
                column.record_bytes(doc, value, dictionaries, arena);
                column
            },
        );
    }
    #[inline]
    pub fn record_numerical_by_handle<T: Into<NumericalValue> + Copy>(
        &mut self,
        doc: RowId,
        handle: ColumnHandle,
        numerical_value: T,
    ) {
        match handle.0 {
            ColumnHandleInner::Dense(idx) => {
                let col = &mut self.dense_columns[idx as usize];
                let val_u64 = accept_and_convert_numerical(col.column_type, numerical_value.into());
                col.record(doc, val_u64);
            }
            ColumnHandleInner::Arena(addr) => {
                let arena = &mut self.arena;
                self.numerical_field_hash_map
                    .mutate_at(addr, |mut column: NumericalColumnWriter| {
                        column.record_numerical_value(doc, numerical_value.into(), arena);
                        column
                    });
            }
            ColumnHandleInner::DenseStr(_) => {
                panic!("ColumnHandle kind mismatch: expected numerical handle")
            }
        }
    }

    #[inline]
    pub fn record_ip_addr_by_handle(&mut self, doc: RowId, handle: ColumnHandle, ip_addr: Ipv6Addr) {
        let arena = &mut self.arena;
        self.ip_addr_field_hash_map
            .mutate_at(handle.arena_addr(), |mut column: ColumnWriter| {
                column.record(doc, ip_addr, arena);
                column
            });
    }

    #[inline]
    pub fn record_bool_by_handle(&mut self, doc: RowId, handle: ColumnHandle, val: bool) {
        let arena = &mut self.arena;
        self.bool_field_hash_map
            .mutate_at(handle.arena_addr(), |mut column: ColumnWriter| {
                column.record(doc, val, arena);
                column
            });
    }

    #[inline]
    pub fn record_datetime_by_handle(
        &mut self,
        doc: RowId,
        handle: ColumnHandle,
        datetime: common::DateTime,
    ) {
        match handle.0 {
            ColumnHandleInner::Dense(idx) => {
                let col = &mut self.dense_columns[idx as usize];
                col.record(doc, datetime.into_timestamp_nanos().to_u64());
            }
            ColumnHandleInner::Arena(addr) => {
                let arena = &mut self.arena;
                self.datetime_field_hash_map
                    .mutate_at(addr, |mut column: ColumnWriter| {
                        column.record(
                            doc,
                            NumericalValue::I64(datetime.into_timestamp_nanos()),
                            arena,
                        );
                        column
                    });
            }
            ColumnHandleInner::DenseStr(_) => {
                panic!("ColumnHandle kind mismatch: expected datetime handle")
            }
        }
    }

    #[inline]
    pub fn record_str_by_handle(&mut self, doc: RowId, handle: ColumnHandle, value: &str) {
        match handle.0 {
            ColumnHandleInner::DenseStr(idx) => {
                self.dense_str_columns[idx as usize].record_bytes(
                    doc,
                    value.as_bytes(),
                    &mut self.dictionaries,
                    &mut self.arena,
                );
            }
            ColumnHandleInner::Arena(addr) => {
                let (arena, dictionaries) = (&mut self.arena, &mut self.dictionaries);
                self.str_field_hash_map
                    .mutate_at(addr, |mut column: StrOrBytesColumnWriter| {
                        column.record_bytes(doc, value.as_bytes(), dictionaries, arena);
                        column
                    });
            }
            ColumnHandleInner::Dense(_) => {
                panic!("ColumnHandle kind mismatch: expected str handle")
            }
        }
    }

    #[inline]
    pub fn record_bytes_by_handle(&mut self, doc: RowId, handle: ColumnHandle, value: &[u8]) {
        match handle.0 {
            ColumnHandleInner::DenseStr(idx) => {
                self.dense_str_columns[idx as usize].record_bytes(
                    doc,
                    value,
                    &mut self.dictionaries,
                    &mut self.arena,
                );
            }
            ColumnHandleInner::Arena(addr) => {
                let (arena, dictionaries) = (&mut self.arena, &mut self.dictionaries);
                self.bytes_field_hash_map
                    .mutate_at(addr, |mut column: StrOrBytesColumnWriter| {
                        column.record_bytes(doc, value, dictionaries, arena);
                        column
                    });
            }
            ColumnHandleInner::Dense(_) => {
                panic!("ColumnHandle kind mismatch: expected bytes handle")
            }
        }
    }

    pub fn serialize(
        &mut self,
        num_docs: RowId,
        old_to_new_row_ids: Option<&[RowId]>,
        wrt: &mut dyn io::Write,
    ) -> io::Result<()> {
        let mut serializer = ColumnarSerializer::new(wrt);

        let mut columns: Vec<(&[u8], ColumnType, ColumnSource)> = self
            .numerical_field_hash_map
            .iter()
            .map(|(column_name, addr)| {
                let numerical_column_writer: NumericalColumnWriter =
                    self.numerical_field_hash_map.read(addr);
                let column_type = numerical_column_writer.numerical_type().into();
                (column_name, column_type, ColumnSource::Arena(addr))
            })
            .collect();
        columns.extend(
            self.dense_columns
                .iter()
                .enumerate()
                .map(|(idx, col)| (&col.name[..], col.column_type, ColumnSource::Dense(idx as u32))),
        );
        columns.extend(self.dense_str_columns.iter().enumerate().map(|(idx, col)| {
            (
                &col.name[..],
                col.column_type,
                ColumnSource::DenseStr(idx as u32),
            )
        }));
        columns.extend(
            self.bytes_field_hash_map
                .iter()
                .map(|(column_name, addr)| (column_name, ColumnType::Bytes, ColumnSource::Arena(addr))),
        );
        columns.extend(
            self.str_field_hash_map
                .iter()
                .map(|(column_name, addr)| (column_name, ColumnType::Str, ColumnSource::Arena(addr))),
        );
        columns.extend(
            self.bool_field_hash_map
                .iter()
                .map(|(column_name, addr)| (column_name, ColumnType::Bool, ColumnSource::Arena(addr))),
        );
        columns.extend(
            self.ip_addr_field_hash_map
                .iter()
                .map(|(column_name, addr)| (column_name, ColumnType::IpAddr, ColumnSource::Arena(addr))),
        );
        columns.extend(
            self.datetime_field_hash_map
                .iter()
                .map(|(column_name, addr)| (column_name, ColumnType::DateTime, ColumnSource::Arena(addr))),
        );
        columns.sort_unstable_by_key(|(column_name, col_type, _)| (*column_name, *col_type));
        let (arena, buffers, dictionaries, dense_columns, dense_str_columns) = (
            &self.arena,
            &mut self.buffers,
            &self.dictionaries,
            &self.dense_columns,
            &self.dense_str_columns,
        );
        let mut symbol_byte_buffer: Vec<u8> = Vec::new();
        for (column_name, column_type, source) in columns {
            if column_name.contains(&JSON_END_OF_PATH) {
                // Tantivy uses b'0' as a separator for nested fields in JSON.
                // Column names with a b'0' are not simply ignored by the columnar (and the inverted
                // index).
                continue;
            }
            match column_type {
                ColumnType::Bool => {
                    let ColumnSource::Arena(addr) = source else {
                        unreachable!()
                    };
                    let column_writer: ColumnWriter = self.bool_field_hash_map.read(addr);
                    let cardinality = column_writer.get_cardinality(num_docs);
                    let mut column_serializer =
                        serializer.start_serialize_column(column_name, column_type);
                    serialize_bool_column(
                        cardinality,
                        num_docs,
                        column_writer.operation_iterator(
                            arena,
                            old_to_new_row_ids,
                            &mut symbol_byte_buffer,
                        ),
                        buffers,
                        &mut column_serializer,
                    )?;
                    column_serializer.finalize()?;
                }
                ColumnType::IpAddr => {
                    let ColumnSource::Arena(addr) = source else {
                        unreachable!()
                    };
                    let column_writer: ColumnWriter = self.ip_addr_field_hash_map.read(addr);
                    let cardinality = column_writer.get_cardinality(num_docs);
                    let mut column_serializer =
                        serializer.start_serialize_column(column_name, ColumnType::IpAddr);
                    serialize_ip_addr_column(
                        cardinality,
                        num_docs,
                        column_writer.operation_iterator(
                            arena,
                            old_to_new_row_ids,
                            &mut symbol_byte_buffer,
                        ),
                        buffers,
                        &mut column_serializer,
                    )?;
                    column_serializer.finalize()?;
                }
                ColumnType::Bytes | ColumnType::Str => {
                    let mut column_serializer =
                        serializer.start_serialize_column(column_name, column_type);
                    match source {
                        ColumnSource::DenseStr(idx) => {
                            let col = &dense_str_columns[idx as usize];
                            let dictionary_builder = &dictionaries[col.dictionary_id as usize];
                            col.serialize(
                                num_docs,
                                old_to_new_row_ids,
                                dictionary_builder,
                                buffers,
                                arena,
                                &mut column_serializer,
                            )?;
                        }
                        ColumnSource::Arena(addr) => {
                            let str_or_bytes_column_writer: StrOrBytesColumnWriter =
                                if column_type == ColumnType::Bytes {
                                    self.bytes_field_hash_map.read(addr)
                                } else {
                                    self.str_field_hash_map.read(addr)
                                };
                            let dictionary_builder =
                                &dictionaries[str_or_bytes_column_writer.dictionary_id as usize];
                            let cardinality = str_or_bytes_column_writer
                                .column_writer
                                .get_cardinality(num_docs);
                            serialize_bytes_or_str_column(
                                cardinality,
                                num_docs,
                                str_or_bytes_column_writer.sort_values_within_row,
                                dictionary_builder,
                                str_or_bytes_column_writer.operation_iterator(
                                    arena,
                                    old_to_new_row_ids,
                                    &mut symbol_byte_buffer,
                                ),
                                buffers,
                                &self.arena,
                                &mut column_serializer,
                            )?;
                        }
                        ColumnSource::Dense(_) => unreachable!(),
                    }
                    column_serializer.finalize()?;
                }
                ColumnType::F64 | ColumnType::I64 | ColumnType::U64 => {
                    let mut column_serializer =
                        serializer.start_serialize_column(column_name, column_type);
                    if self.generated_tie_breaker_columns.contains(column_name) {
                        crate::column::serialize_generated_tie_breaker_column(
                            num_docs,
                            &mut column_serializer,
                        )?;
                    } else {
                        match source {
                            ColumnSource::Dense(idx) => {
                                dense_columns[idx as usize].serialize(
                                    num_docs,
                                    old_to_new_row_ids,
                                    buffers,
                                    &mut column_serializer,
                                )?;
                            }
                            ColumnSource::Arena(addr) => {
                                let numerical_column_writer: NumericalColumnWriter =
                                    self.numerical_field_hash_map.read(addr);
                                let cardinality = numerical_column_writer.cardinality(num_docs);
                                let numerical_type = column_type.numerical_type().unwrap();
                                serialize_numerical_column(
                                    cardinality,
                                    num_docs,
                                    numerical_type,
                                    numerical_column_writer.operation_iterator(
                                        arena,
                                        old_to_new_row_ids,
                                        &mut symbol_byte_buffer,
                                    ),
                                    buffers,
                                    &mut column_serializer,
                                )?;
                            }
                            ColumnSource::DenseStr(_) => unreachable!(),
                        }
                    }
                    column_serializer.finalize()?;
                }
                ColumnType::DateTime => {
                    let mut column_serializer =
                        serializer.start_serialize_column(column_name, ColumnType::DateTime);
                    match source {
                        ColumnSource::Dense(idx) => {
                            dense_columns[idx as usize].serialize(
                                num_docs,
                                old_to_new_row_ids,
                                buffers,
                                &mut column_serializer,
                            )?;
                        }
                        ColumnSource::Arena(addr) => {
                            let column_writer: ColumnWriter =
                                self.datetime_field_hash_map.read(addr);
                            let cardinality = column_writer.get_cardinality(num_docs);
                            serialize_numerical_column(
                                cardinality,
                                num_docs,
                                NumericalType::I64,
                                column_writer.operation_iterator(
                                    arena,
                                    old_to_new_row_ids,
                                    &mut symbol_byte_buffer,
                                ),
                                buffers,
                                &mut column_serializer,
                            )?;
                        }
                        ColumnSource::DenseStr(_) => unreachable!(),
                    }
                    column_serializer.finalize()?;
                }
            };
        }
        serializer.finalize(num_docs)?;
        Ok(())
    }
}

/// Returns the sort order of a numerical or datetime column, given its operations.
///
/// # Hidden contract
///
/// For this to make sense, all numerical value emitted have to be of the same type.
fn collect_numerical_sort_order(
    ops: impl Iterator<Item = ColumnOperation<NumericalValue>>,
    num_docs: RowId,
    reversed: bool,
) -> Vec<u32> {
    collect_sort_order_from_ops(
        ops,
        num_docs,
        reversed,
        |numerical_value| {
            Some(match numerical_value {
                NumericalValue::U64(val_u64) => val_u64.to_u64(),
                NumericalValue::I64(val_i64) => val_i64.to_u64(),
                NumericalValue::F64(val_f64) => val_f64.to_u64(),
            })
        },
        // None for missing values. Option<u64> sorts None < Some(_),
        // placing nulls before non-null values.
        None,
        |a, b| a.cmp(b),
    )
}

/// Shared sorting pattern for both numeric and Str/Bytes sort fields.
///
/// Iterates column operations, fills gaps for missing docs with `default_key`, converts each value
/// to a sort key via `value_to_key`, then sorts by the key using `cmp_keys`. Returns the doc ids
/// in sorted order.
fn collect_sort_order_from_ops<V, K: Clone>(
    ops: impl Iterator<Item = ColumnOperation<V>>,
    num_docs: RowId,
    reversed: bool,
    value_to_key: impl Fn(V) -> K,
    default_key: K,
    cmp_keys: impl Fn(&K, &K) -> std::cmp::Ordering,
) -> Vec<u32> {
    let mut doc_sort_keys: Vec<(K, RowId)> = Vec::with_capacity(num_docs as usize);
    let mut start_doc_check_fill: RowId = 0;
    let mut current_doc_opt: Option<RowId> = None;

    for op in ops {
        match op {
            ColumnOperation::NewDoc(doc) => {
                current_doc_opt = Some(doc);
            }
            ColumnOperation::Value(val) => {
                if let Some(current_doc) = current_doc_opt {
                    // Fill gaps since the last doc with the default key.
                    doc_sort_keys.extend(
                        (start_doc_check_fill..current_doc).map(|doc| (default_key.clone(), doc)),
                    );
                    start_doc_check_fill = current_doc + 1;
                    // For multivalued fields, only the first value is used.
                    current_doc_opt = None;

                    doc_sort_keys.push((value_to_key(val), current_doc));
                }
            }
        }
    }
    // Fill remaining docs at the tail.
    doc_sort_keys.extend((start_doc_check_fill..num_docs).map(|doc| (default_key.clone(), doc)));

    doc_sort_keys.sort_by(|(left_key, _), (right_key, _)| {
        let cmp = cmp_keys(left_key, right_key);
        if reversed { cmp.reverse() } else { cmp }
    });
    doc_sort_keys
        .into_iter()
        .map(|(_sort_key, doc)| doc)
        .collect()
}

// Serialize [Dictionary, Column, dictionary num bytes U32::LE]
// Column: [Column Index, Column Values, column index num bytes U32::LE]
#[expect(clippy::too_many_arguments)]
fn serialize_bytes_or_str_column(
    cardinality: Cardinality,
    num_docs: RowId,
    sort_values_within_row: bool,
    dictionary_builder: &DictionaryBuilder,
    operation_it: impl Iterator<Item = ColumnOperation<UnorderedId>>,
    buffers: &mut SpareBuffers,
    arena: &MemoryArena,
    wrt: impl io::Write,
) -> io::Result<()> {
    let SpareBuffers {
        value_index_builders,
        u64_values,
        ..
    } = buffers;
    let mut counting_writer = CountingWriter::wrap(wrt);
    let term_id_mapping: TermIdMapping =
        dictionary_builder.serialize(arena, &mut counting_writer)?;
    let dictionary_num_bytes: u32 = counting_writer.written_bytes() as u32;
    let mut wrt = counting_writer.finish();
    let operation_iterator = operation_it.map(|symbol: ColumnOperation<UnorderedId>| {
        // We map unordered ids to ordered ids.
        match symbol {
            ColumnOperation::Value(unordered_id) => {
                let ordered_id = term_id_mapping.to_ord(unordered_id);
                ColumnOperation::Value(ordered_id.0 as u64)
            }
            ColumnOperation::NewDoc(doc) => ColumnOperation::NewDoc(doc),
        }
    });
    send_to_serialize_column_mappable_to_u64(
        operation_iterator,
        cardinality,
        num_docs,
        sort_values_within_row,
        value_index_builders,
        u64_values,
        &mut wrt,
    )?;
    wrt.write_all(&dictionary_num_bytes.to_le_bytes()[..])?;
    Ok(())
}

fn serialize_numerical_column(
    cardinality: Cardinality,
    num_docs: RowId,
    numerical_type: NumericalType,
    op_iterator: impl Iterator<Item = ColumnOperation<NumericalValue>>,
    buffers: &mut SpareBuffers,
    wrt: &mut impl io::Write,
) -> io::Result<()> {
    let SpareBuffers {
        value_index_builders,
        u64_values,
        ..
    } = buffers;
    match numerical_type {
        NumericalType::I64 => {
            send_to_serialize_column_mappable_to_u64(
                coerce_numerical_symbol::<i64>(op_iterator),
                cardinality,
                num_docs,
                false,
                value_index_builders,
                u64_values,
                wrt,
            )?;
        }
        NumericalType::U64 => {
            send_to_serialize_column_mappable_to_u64(
                coerce_numerical_symbol::<u64>(op_iterator),
                cardinality,
                num_docs,
                false,
                value_index_builders,
                u64_values,
                wrt,
            )?;
        }
        NumericalType::F64 => {
            send_to_serialize_column_mappable_to_u64(
                coerce_numerical_symbol::<f64>(op_iterator),
                cardinality,
                num_docs,
                false,
                value_index_builders,
                u64_values,
                wrt,
            )?;
        }
    };
    Ok(())
}

fn serialize_bool_column(
    cardinality: Cardinality,
    num_docs: RowId,
    column_operations_it: impl Iterator<Item = ColumnOperation<bool>>,
    buffers: &mut SpareBuffers,
    wrt: &mut impl io::Write,
) -> io::Result<()> {
    let SpareBuffers {
        value_index_builders,
        u64_values,
        ..
    } = buffers;
    send_to_serialize_column_mappable_to_u64(
        column_operations_it.map(|bool_column_operation| match bool_column_operation {
            ColumnOperation::NewDoc(doc) => ColumnOperation::NewDoc(doc),
            ColumnOperation::Value(bool_val) => ColumnOperation::Value(bool_val.to_u64()),
        }),
        cardinality,
        num_docs,
        false,
        value_index_builders,
        u64_values,
        wrt,
    )?;
    Ok(())
}

fn serialize_ip_addr_column(
    cardinality: Cardinality,
    num_docs: RowId,
    column_operations_it: impl Iterator<Item = ColumnOperation<Ipv6Addr>>,
    buffers: &mut SpareBuffers,
    wrt: &mut impl io::Write,
) -> io::Result<()> {
    let SpareBuffers {
        value_index_builders,
        ip_addr_values,
        ..
    } = buffers;
    send_to_serialize_column_mappable_to_u128(
        column_operations_it,
        cardinality,
        num_docs,
        value_index_builders,
        ip_addr_values,
        wrt,
    )?;
    Ok(())
}

fn send_to_serialize_column_mappable_to_u128<
    T: Copy + Ord + std::fmt::Debug + Send + Sync + MonotonicallyMappableToU128 + PartialOrd,
>(
    op_iterator: impl Iterator<Item = ColumnOperation<T>>,
    cardinality: Cardinality,
    num_rows: RowId,
    value_index_builders: &mut PreallocatedIndexBuilders,
    values: &mut Vec<T>,
    mut wrt: impl io::Write,
) -> io::Result<()> {
    values.clear();
    // TODO: split index and values
    let serializable_column_index = match cardinality {
        Cardinality::Full => {
            consume_operation_iterator(
                op_iterator,
                value_index_builders.borrow_required_index_builder(),
                values,
            );
            SerializableColumnIndex::Full
        }
        Cardinality::Optional => {
            let optional_index_builder = value_index_builders.borrow_optional_index_builder();
            consume_operation_iterator(op_iterator, optional_index_builder, values);
            let optional_index = optional_index_builder.finish(num_rows);
            SerializableColumnIndex::Optional(SerializableOptionalIndex {
                num_rows,
                non_null_row_ids: Box::new(optional_index),
            })
        }
        Cardinality::Multivalued => {
            let multivalued_index_builder = value_index_builders.borrow_multivalued_index_builder();
            consume_operation_iterator(op_iterator, multivalued_index_builder, values);
            let serializable_multivalued_index = multivalued_index_builder.finish(num_rows);
            SerializableColumnIndex::Multivalued(serializable_multivalued_index)
        }
    };
    crate::column::serialize_column_mappable_to_u128(
        serializable_column_index,
        &&values[..],
        &mut wrt,
    )?;
    Ok(())
}

fn send_to_serialize_column_mappable_to_u64(
    op_iterator: impl Iterator<Item = ColumnOperation<u64>>,
    cardinality: Cardinality,
    num_rows: RowId,
    sort_values_within_row: bool,
    value_index_builders: &mut PreallocatedIndexBuilders,
    values: &mut Vec<u64>,
    mut wrt: impl io::Write,
) -> io::Result<()> {
    values.clear();
    let serializable_column_index = match cardinality {
        Cardinality::Full => {
            consume_operation_iterator(
                op_iterator,
                value_index_builders.borrow_required_index_builder(),
                values,
            );
            SerializableColumnIndex::Full
        }
        Cardinality::Optional => {
            let optional_index_builder = value_index_builders.borrow_optional_index_builder();
            consume_operation_iterator(op_iterator, optional_index_builder, values);
            let optional_index = optional_index_builder.finish(num_rows);
            SerializableColumnIndex::Optional(SerializableOptionalIndex {
                non_null_row_ids: Box::new(optional_index),
                num_rows,
            })
        }
        Cardinality::Multivalued => {
            let multivalued_index_builder = value_index_builders.borrow_multivalued_index_builder();
            consume_operation_iterator(op_iterator, multivalued_index_builder, values);
            let serializable_multivalued_index = multivalued_index_builder.finish(num_rows);
            if sort_values_within_row {
                sort_values_within_row_in_place(
                    serializable_multivalued_index.start_offsets.boxed_iter(),
                    values,
                );
            }
            SerializableColumnIndex::Multivalued(serializable_multivalued_index)
        }
    };
    crate::column::serialize_column_u64_slice(
        serializable_column_index,
        &values[..],
        &mut wrt,
    )?;
    Ok(())
}

fn sort_values_within_row_in_place(
    multivalued_index: impl Iterator<Item = RowId>,
    values: &mut [u64],
) {
    let mut start_index: usize = 0;
    for end_index in multivalued_index {
        let end_index = end_index as usize;
        values[start_index..end_index].sort_unstable();
        start_index = end_index;
    }
}

fn coerce_numerical_symbol<T>(
    operation_iterator: impl Iterator<Item = ColumnOperation<NumericalValue>>,
) -> impl Iterator<Item = ColumnOperation<u64>>
where T: Coerce + MonotonicallyMappableToU64 {
    operation_iterator.map(|symbol| match symbol {
        ColumnOperation::NewDoc(doc) => ColumnOperation::NewDoc(doc),
        ColumnOperation::Value(numerical_value) => {
            ColumnOperation::Value(T::coerce(numerical_value).to_u64())
        }
    })
}

fn consume_operation_iterator<T: Ord, TIndexBuilder: IndexBuilder>(
    operation_iterator: impl Iterator<Item = ColumnOperation<T>>,
    index_builder: &mut TIndexBuilder,
    values: &mut Vec<T>,
) {
    for symbol in operation_iterator {
        match symbol {
            ColumnOperation::NewDoc(doc) => {
                index_builder.record_row(doc);
            }
            ColumnOperation::Value(value) => {
                index_builder.record_value();
                values.push(value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use stacker::MemoryArena;

    use crate::columnar::writer::column_operation::ColumnOperation;
    use crate::{Cardinality, NumericalValue};

    #[test]
    fn test_column_writer_required_simple() {
        let mut arena = MemoryArena::default();
        let mut column_writer = super::ColumnWriter::default();
        column_writer.record(0u32, NumericalValue::from(14i64), &mut arena);
        column_writer.record(1u32, NumericalValue::from(15i64), &mut arena);
        column_writer.record(2u32, NumericalValue::from(-16i64), &mut arena);
        assert_eq!(column_writer.get_cardinality(3), Cardinality::Full);
        let mut buffer = Vec::new();
        let symbols: Vec<ColumnOperation<NumericalValue>> = column_writer
            .operation_iterator(&arena, None, &mut buffer)
            .collect();
        assert_eq!(symbols.len(), 6);
        assert!(matches!(symbols[0], ColumnOperation::NewDoc(0u32)));
        assert!(matches!(
            symbols[1],
            ColumnOperation::Value(NumericalValue::I64(14i64))
        ));
        assert!(matches!(symbols[2], ColumnOperation::NewDoc(1u32)));
        assert!(matches!(
            symbols[3],
            ColumnOperation::Value(NumericalValue::I64(15i64))
        ));
        assert!(matches!(symbols[4], ColumnOperation::NewDoc(2u32)));
        assert!(matches!(
            symbols[5],
            ColumnOperation::Value(NumericalValue::I64(-16i64))
        ));
    }

    #[test]
    fn test_column_writer_optional_cardinality_missing_first() {
        let mut arena = MemoryArena::default();
        let mut column_writer = super::ColumnWriter::default();
        column_writer.record(1u32, NumericalValue::from(15i64), &mut arena);
        column_writer.record(2u32, NumericalValue::from(-16i64), &mut arena);
        assert_eq!(column_writer.get_cardinality(3), Cardinality::Optional);
        let mut buffer = Vec::new();
        let symbols: Vec<ColumnOperation<NumericalValue>> = column_writer
            .operation_iterator(&arena, None, &mut buffer)
            .collect();
        assert_eq!(symbols.len(), 4);
        assert!(matches!(symbols[0], ColumnOperation::NewDoc(1u32)));
        assert!(matches!(
            symbols[1],
            ColumnOperation::Value(NumericalValue::I64(15i64))
        ));
        assert!(matches!(symbols[2], ColumnOperation::NewDoc(2u32)));
        assert!(matches!(
            symbols[3],
            ColumnOperation::Value(NumericalValue::I64(-16i64))
        ));
    }

    #[test]
    fn test_column_writer_optional_cardinality_missing_last() {
        let mut arena = MemoryArena::default();
        let mut column_writer = super::ColumnWriter::default();
        column_writer.record(0u32, NumericalValue::from(15i64), &mut arena);
        assert_eq!(column_writer.get_cardinality(2), Cardinality::Optional);
        let mut buffer = Vec::new();
        let symbols: Vec<ColumnOperation<NumericalValue>> = column_writer
            .operation_iterator(&arena, None, &mut buffer)
            .collect();
        assert_eq!(symbols.len(), 2);
        assert!(matches!(symbols[0], ColumnOperation::NewDoc(0u32)));
        assert!(matches!(
            symbols[1],
            ColumnOperation::Value(NumericalValue::I64(15i64))
        ));
    }

    #[test]
    fn test_column_writer_multivalued() {
        let mut arena = MemoryArena::default();
        let mut column_writer = super::ColumnWriter::default();
        column_writer.record(0u32, NumericalValue::from(16i64), &mut arena);
        column_writer.record(0u32, NumericalValue::from(17i64), &mut arena);
        assert_eq!(column_writer.get_cardinality(1), Cardinality::Multivalued);
        let mut buffer = Vec::new();
        let symbols: Vec<ColumnOperation<NumericalValue>> = column_writer
            .operation_iterator(&arena, None, &mut buffer)
            .collect();
        assert_eq!(symbols.len(), 3);
        assert!(matches!(symbols[0], ColumnOperation::NewDoc(0u32)));
        assert!(matches!(
            symbols[1],
            ColumnOperation::Value(NumericalValue::I64(16i64))
        ));
        assert!(matches!(
            symbols[2],
            ColumnOperation::Value(NumericalValue::I64(17i64))
        ));
    }

    #[test]
    fn test_dense_numerical_column_full_optional_multivalued_and_datetime() {
        let mut writer = super::ColumnarWriter::default();
        let h_full = writer.record_column_type_with_handle("full_i64", super::ColumnType::I64, false);
        let h_opt_mid =
            writer.record_column_type_with_handle("opt_mid_u64", super::ColumnType::U64, false);
        let h_opt_tail =
            writer.record_column_type_with_handle("opt_tail_f64", super::ColumnType::F64, false);
        let h_empty =
            writer.record_column_type_with_handle("empty_i64", super::ColumnType::I64, false);
        let h_mv = writer.record_column_type_with_handle("mv_i64", super::ColumnType::I64, false);
        let h_dt =
            writer.record_column_type_with_handle("ts", super::ColumnType::DateTime, false);
        let _ = h_empty;

        // 4 docs: 0, 1, 2, 3
        // full_i64: mix handle + name-based recording
        writer.record_numerical_by_handle(0, h_full, -10i64);
        writer.record_numerical(1, "full_i64", 20i64);
        writer.record_numerical_by_handle(2, h_full, -5i64);
        writer.record_numerical_by_handle(3, h_full, 0i64);

        // opt_mid_u64: missing doc 0 and doc 2
        writer.record_numerical_by_handle(1, h_opt_mid, 100u64);
        writer.record_numerical_by_handle(3, h_opt_mid, 300u64);

        // opt_tail_f64: docs 0, 1 present; docs 2, 3 missing at tail
        writer.record_numerical_by_handle(0, h_opt_tail, 1.5f64);
        writer.record_numerical_by_handle(1, h_opt_tail, 2.5f64);

        // mv_i64: doc 0 has [10], doc 2 has [30, 20]
        writer.record_numerical_by_handle(0, h_mv, 10i64);
        writer.record_numerical_by_handle(2, h_mv, 30i64);
        writer.record_numerical_by_handle(2, h_mv, 20i64);

        // datetime: docs 0..4
        for doc in 0..4u32 {
            writer.record_datetime_by_handle(
                doc,
                h_dt,
                common::DateTime::from_timestamp_secs(1000 + doc as i64),
            );
        }

        // Check sort_order before serialize
        assert_eq!(writer.sort_order("full_i64", 4, false), vec![0, 2, 3, 1]);
        assert_eq!(writer.sort_order("full_i64", 4, true), vec![1, 3, 2, 0]);
        assert_eq!(writer.sort_order("opt_mid_u64", 4, false), vec![0, 2, 1, 3]);
        assert_eq!(writer.sort_order("opt_tail_f64", 4, false), vec![2, 3, 0, 1]);
        assert_eq!(writer.sort_order("mv_i64", 4, false), vec![1, 3, 0, 2]);
        assert_eq!(writer.sort_order("ts", 4, true), vec![3, 2, 1, 0]);
        assert!(writer.mem_usage() > 0);

        let mut buf = Vec::new();
        writer.serialize(4, None, &mut buf).unwrap();
        let reader = crate::ColumnarReader::open(buf).unwrap();

        let crate::DynamicColumn::I64(full_col) =
            reader.read_columns("full_i64").unwrap()[0].open().unwrap()
        else {
            panic!("expected I64 column");
        };
        assert_eq!(full_col.get_cardinality(), Cardinality::Full);
        assert_eq!(
            (0..4).map(|d| full_col.first(d)).collect::<Vec<_>>(),
            vec![Some(-10), Some(20), Some(-5), Some(0)]
        );

        let crate::DynamicColumn::U64(opt_mid) =
            reader.read_columns("opt_mid_u64").unwrap()[0].open().unwrap()
        else {
            panic!("expected U64 column");
        };
        assert_eq!(opt_mid.get_cardinality(), Cardinality::Optional);
        assert_eq!(
            (0..4).map(|d| opt_mid.first(d)).collect::<Vec<_>>(),
            vec![None, Some(100), None, Some(300)]
        );

        let crate::DynamicColumn::F64(opt_tail) =
            reader.read_columns("opt_tail_f64").unwrap()[0].open().unwrap()
        else {
            panic!("expected F64 column");
        };
        assert_eq!(opt_tail.get_cardinality(), Cardinality::Optional);
        assert_eq!(
            (0..4).map(|d| opt_tail.first(d)).collect::<Vec<_>>(),
            vec![Some(1.5), Some(2.5), None, None]
        );

        let crate::DynamicColumn::I64(empty_col) =
            reader.read_columns("empty_i64").unwrap()[0].open().unwrap()
        else {
            panic!("expected I64 column");
        };
        assert_eq!(empty_col.get_cardinality(), Cardinality::Optional);
        assert_eq!(
            (0..4).map(|d| empty_col.first(d)).collect::<Vec<_>>(),
            vec![None, None, None, None]
        );

        let crate::DynamicColumn::I64(mv_col) =
            reader.read_columns("mv_i64").unwrap()[0].open().unwrap()
        else {
            panic!("expected I64 column");
        };
        assert_eq!(mv_col.get_cardinality(), Cardinality::Multivalued);
        assert_eq!(mv_col.values_for_doc(0).collect::<Vec<_>>(), vec![10]);
        assert!(mv_col.values_for_doc(1).next().is_none());
        assert_eq!(mv_col.values_for_doc(2).collect::<Vec<_>>(), vec![30, 20]);
        assert!(mv_col.values_for_doc(3).next().is_none());

        let crate::DynamicColumn::DateTime(dt_col) =
            reader.read_columns("ts").unwrap()[0].open().unwrap()
        else {
            panic!("expected DateTime column");
        };
        assert_eq!(dt_col.get_cardinality(), Cardinality::Full);
        assert_eq!(
            (0..4)
                .map(|d| dt_col.first(d).unwrap().into_timestamp_secs())
                .collect::<Vec<_>>(),
            vec![1000, 1001, 1002, 1003]
        );
    }

    #[test]
    fn test_dense_numerical_column_remapped_row_ids() {
        let mut writer = super::ColumnarWriter::default();
        let h_full = writer.record_column_type_with_handle("full", super::ColumnType::I64, false);
        let h_opt = writer.record_column_type_with_handle("opt", super::ColumnType::I64, false);
        let h_tail = writer.record_column_type_with_handle("tail", super::ColumnType::I64, false);
        let h_mv = writer.record_column_type_with_handle("mv", super::ColumnType::I64, false);

        // Old docs: 0, 1, 2, 3 -> New docs: [3, 1, 0, 2]
        let old_to_new = [3u32, 1, 0, 2];
        for d in 0..4u32 {
            writer.record_numerical_by_handle(d, h_full, (d as i64 + 1) * 10);
        }
        writer.record_numerical_by_handle(1, h_opt, 111i64);
        writer.record_numerical_by_handle(3, h_opt, 333i64);

        writer.record_numerical_by_handle(0, h_tail, 10i64);
        writer.record_numerical_by_handle(1, h_tail, 20i64);

        writer.record_numerical_by_handle(0, h_mv, 1i64);
        writer.record_numerical_by_handle(0, h_mv, 2i64);
        writer.record_numerical_by_handle(2, h_mv, 9i64);

        let mut buf = Vec::new();
        writer.serialize(4, Some(&old_to_new), &mut buf).unwrap();
        let reader = crate::ColumnarReader::open(buf).unwrap();

        let crate::DynamicColumn::I64(full) =
            reader.read_columns("full").unwrap()[0].open().unwrap()
        else {
            panic!("expected I64 column");
        };
        assert_eq!(
            (0..4).map(|d| full.first(d)).collect::<Vec<_>>(),
            vec![Some(30), Some(20), Some(40), Some(10)]
        );

        let crate::DynamicColumn::I64(opt) =
            reader.read_columns("opt").unwrap()[0].open().unwrap()
        else {
            panic!("expected I64 column");
        };
        assert_eq!(
            (0..4).map(|d| opt.first(d)).collect::<Vec<_>>(),
            vec![None, Some(111), Some(333), None]
        );

        let crate::DynamicColumn::I64(tail) =
            reader.read_columns("tail").unwrap()[0].open().unwrap()
        else {
            panic!("expected I64 column");
        };
        assert_eq!(
            (0..4).map(|d| tail.first(d)).collect::<Vec<_>>(),
            vec![None, Some(20), None, Some(10)]
        );

        let crate::DynamicColumn::I64(mv) =
            reader.read_columns("mv").unwrap()[0].open().unwrap()
        else {
            panic!("expected I64 column");
        };
        assert_eq!(mv.values_for_doc(0).collect::<Vec<_>>(), vec![9]);
        assert!(mv.values_for_doc(1).next().is_none());
        assert!(mv.values_for_doc(2).next().is_none());
        assert_eq!(mv.values_for_doc(3).collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn test_dense_str_and_bytes_columns() {
        let mut writer = super::ColumnarWriter::default();
        let h_full = writer.record_column_type_with_handle("full_str", super::ColumnType::Str, false);
        let h_opt = writer.record_column_type_with_handle("opt_str", super::ColumnType::Str, false);
        let h_mv = writer.record_column_type_with_handle("mv_sorted", super::ColumnType::Str, true);
        let h_bytes =
            writer.record_column_type_with_handle("raw_bytes", super::ColumnType::Bytes, false);

        // Full str with repeated empty strings (exercises empty_bytes_id cache)
        writer.record_str_by_handle(0, h_full, "");
        writer.record_str_by_handle(1, h_full, "beta");
        writer.record_str(2, "full_str", "");
        writer.record_str_by_handle(3, h_full, "alpha");

        // Optional str (missing doc 0 and doc 2)
        writer.record_str_by_handle(1, h_opt, "z");
        writer.record_str_by_handle(3, h_opt, "a");

        // Multivalued with sort_values_within_row=true
        writer.record_str_by_handle(0, h_mv, "c");
        writer.record_str_by_handle(0, h_mv, "a");
        writer.record_str_by_handle(0, h_mv, "b");
        writer.record_str_by_handle(2, h_mv, "y");
        writer.record_str_by_handle(2, h_mv, "x");

        // Bytes column (tail missing doc 2, 3)
        writer.record_bytes_by_handle(0, h_bytes, b"");
        writer.record_bytes_by_handle(1, h_bytes, b"\x01\x02");

        assert_eq!(writer.sort_order("full_str", 4, false), vec![0, 2, 3, 1]);
        assert_eq!(writer.sort_order("opt_str", 4, false), vec![0, 2, 3, 1]);

        // Test both unmapped and remapped serialization
        let mut buf = Vec::new();
        writer.serialize(4, None, &mut buf).unwrap();
        let reader = crate::ColumnarReader::open(buf).unwrap();

        let crate::DynamicColumn::Str(full_str) =
            reader.read_columns("full_str").unwrap()[0].open().unwrap()
        else {
            panic!("expected Str column");
        };
        let get_strs = |col: &crate::StrColumn, doc: u32| -> Vec<String> {
            col.term_ord_column
                .values_for_doc(doc)
                .map(|ord| {
                    let mut s = String::new();
                    col.ord_to_str(ord, &mut s).unwrap();
                    s
                })
                .collect()
        };
        assert_eq!(full_str.term_ord_column.get_cardinality(), Cardinality::Full);
        assert_eq!(get_strs(&full_str, 0), vec![""]);
        assert_eq!(get_strs(&full_str, 1), vec!["beta"]);
        assert_eq!(get_strs(&full_str, 2), vec![""]);
        assert_eq!(get_strs(&full_str, 3), vec!["alpha"]);

        let crate::DynamicColumn::Str(mv_col) =
            reader.read_columns("mv_sorted").unwrap()[0].open().unwrap()
        else {
            panic!("expected Str column");
        };
        assert_eq!(get_strs(&mv_col, 0), vec!["a", "b", "c"]);
        assert!(get_strs(&mv_col, 1).is_empty());
        assert_eq!(get_strs(&mv_col, 2), vec!["x", "y"]);

        // Now serialize with doc id remapping [2, 0, 3, 1]
        let old_to_new = [2u32, 0, 3, 1];
        let mut buf_remap = Vec::new();
        writer.serialize(4, Some(&old_to_new), &mut buf_remap).unwrap();
        let reader_r = crate::ColumnarReader::open(buf_remap).unwrap();
        let crate::DynamicColumn::Str(full_r) =
            reader_r.read_columns("full_str").unwrap()[0].open().unwrap()
        else {
            panic!("expected Str column");
        };
        assert_eq!(get_strs(&full_r, 0), vec!["beta"]);
        assert_eq!(get_strs(&full_r, 1), vec!["alpha"]);
        assert_eq!(get_strs(&full_r, 2), vec![""]);
        assert_eq!(get_strs(&full_r, 3), vec![""]);

        let crate::DynamicColumn::Str(mv_r) =
            reader_r.read_columns("mv_sorted").unwrap()[0].open().unwrap()
        else {
            panic!("expected Str column");
        };
        assert_eq!(get_strs(&mv_r, 2), vec!["a", "b", "c"]);
        assert_eq!(get_strs(&mv_r, 3), vec!["x", "y"]);
    }
}
