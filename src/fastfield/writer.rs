use std::io;

use columnar::{ColumnHandle, ColumnType, ColumnarWriter, NumericalValue};
use common::{DateTimePrecision, JsonPathWriter};
use tokenizer_api::Token;

use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::schema::document::{Document, ReferenceValue, ReferenceValueLeaf, Value};
use crate::schema::{value_type_to_column_type, Field, FieldType, Schema, Type};
use crate::tokenizer::{TextAnalyzer, TokenizerManager};
use crate::{DocId, TantivyError};

/// Only index JSON down to a depth of 20.
/// This is mostly to guard us from a stack overflow triggered by malicious input.
const JSON_DEPTH_LIMIT: usize = 20;

/// Returns the analyzer applied to the values of a text fast field, or `None` when values are
/// recorded as is.
///
/// `tokenizer_name` is `None` when no tokenizer was configured (`FAST` flag). A configured
/// tokenizer is always looked up, including the raw tokenizer.
fn resolve_fast_field_tokenizer(
    tokenizer_manager: &TokenizerManager,
    tokenizer_name: Option<&str>,
) -> crate::Result<Option<TextAnalyzer>> {
    let Some(tokenizer_name) = tokenizer_name else {
        return Ok(None);
    };
    let text_analyzer = tokenizer_manager.get(tokenizer_name).ok_or_else(|| {
        TantivyError::InvalidArgument(format!("Tokenizer `{tokenizer_name}` not found"))
    })?;
    Ok(Some(text_analyzer))
}

/// Direct handle into one of [`ColumnarWriter`]'s typed field maps.
///
/// Storing the column kind alongside the handle guarantees that mismatched runtime value types
/// (e.g. writing an `IpAddr` into a `u64` schema field) safely fall back to the name-based map
/// instead of accessing the wrong arena.
#[derive(Clone, Copy, Debug)]
enum TypedColumnHandle {
    Numerical(ColumnHandle),
    Str(ColumnHandle),
    Bytes(ColumnHandle),
    DateTime(ColumnHandle),
    Bool(ColumnHandle),
    IpAddr(ColumnHandle),
}

impl TypedColumnHandle {
    #[inline]
    fn from_type_and_handle(column_type: ColumnType, handle: ColumnHandle) -> Self {
        match column_type {
            ColumnType::I64 | ColumnType::U64 | ColumnType::F64 => Self::Numerical(handle),
            ColumnType::Str => Self::Str(handle),
            ColumnType::Bytes => Self::Bytes(handle),
            ColumnType::DateTime => Self::DateTime(handle),
            ColumnType::Bool => Self::Bool(handle),
            ColumnType::IpAddr => Self::IpAddr(handle),
        }
    }
}

/// The `FastFieldsWriter` groups all of the fast field writers.
pub struct FastFieldsWriter {
    columnar_writer: ColumnarWriter,
    fast_field_names: Vec<Option<String>>,
    /// Direct typed handles into `columnar_writer` for static schema fields, avoiding per-cell
    /// field-name hashing and `ArenaHashMap` probing.
    column_handles: Vec<Option<TypedColumnHandle>>,
    per_field_tokenizer: Vec<Option<TextAnalyzer>>,
    /// When `Some(limit)`, the field's fast-field tokenizer is `RawTokenizer` (optionally wrapped
    /// in `RemoveLongFilter::limit(limit)`), allowing zero-copy recording without `Token` copying.
    per_field_raw_limit: Vec<Option<usize>>,
    date_precisions: Vec<DateTimePrecision>,
    expand_dots: Vec<bool>,
    num_docs: DocId,
    // Buffer that we recycle to avoid allocation.
    json_path_buffer: JsonPathWriter,
}

impl FastFieldsWriter {
    /// Create all `FastFieldWriter` required by the schema.
    #[cfg(test)]
    pub fn from_schema(schema: &Schema) -> crate::Result<FastFieldsWriter> {
        FastFieldsWriter::from_schema_and_tokenizer_manager(schema, TokenizerManager::new())
    }

    /// Create all `FastFieldWriter` required by the schema.
    pub fn from_schema_and_tokenizer_manager(
        schema: &Schema,
        tokenizer_manager: TokenizerManager,
    ) -> crate::Result<FastFieldsWriter> {
        let mut columnar_writer = ColumnarWriter::default();

        let mut fast_field_names: Vec<Option<String>> = vec![None; schema.num_fields()];
        let mut column_handles: Vec<Option<TypedColumnHandle>> = vec![None; schema.num_fields()];
        let mut date_precisions: Vec<DateTimePrecision> =
            std::iter::repeat_with(DateTimePrecision::default)
                .take(schema.num_fields())
                .collect();
        let mut expand_dots = vec![false; schema.num_fields()];
        let mut per_field_tokenizer: Vec<Option<TextAnalyzer>> = vec![None; schema.num_fields()];
        let mut per_field_raw_limit: Vec<Option<usize>> = vec![None; schema.num_fields()];
        // TODO see other types
        for (field_id, field_entry) in schema.fields() {
            if !field_entry.field_type().is_fast() {
                continue;
            }
            if matches!(field_entry.field_type(), FieldType::TieBreaker) {
                columnar_writer.record_tie_breaker_column(field_entry.name());
                continue;
            }
            let idx = field_id.field_id() as usize;
            fast_field_names[idx] = Some(field_entry.name().to_string());
            let value_type = field_entry.field_type().value_type();
            if let FieldType::Date(date_options) = field_entry.field_type() {
                date_precisions[idx] = date_options.get_precision();
            }
            if let FieldType::JsonObject(json_object_options) = field_entry.field_type() {
                per_field_tokenizer[idx] = resolve_fast_field_tokenizer(
                    &tokenizer_manager,
                    json_object_options.get_fast_field_tokenizer_name(),
                )?;
                expand_dots[idx] = json_object_options.is_expand_dots_enabled();
            }
            if let FieldType::Str(text_options) = field_entry.field_type() {
                let tokenizer_opt = resolve_fast_field_tokenizer(
                    &tokenizer_manager,
                    text_options.get_fast_field_tokenizer_name(),
                )?;
                per_field_raw_limit[idx] = tokenizer_opt
                    .as_ref()
                    .and_then(TextAnalyzer::raw_pass_through_limit);
                per_field_tokenizer[idx] = tokenizer_opt;
            }

            let sort_values_within_row = value_type == Type::Facet;
            if let Some(column_type) = value_type_to_column_type(value_type) {
                let handle = columnar_writer.record_column_type_with_handle(
                    field_entry.name(),
                    column_type,
                    sort_values_within_row,
                );
                column_handles[idx] =
                    Some(TypedColumnHandle::from_type_and_handle(column_type, handle));
            }
        }
        Ok(FastFieldsWriter {
            columnar_writer,
            fast_field_names,
            column_handles,
            per_field_tokenizer,
            per_field_raw_limit,
            num_docs: 0u32,
            date_precisions,
            expand_dots,
            json_path_buffer: JsonPathWriter::default(),
        })
    }

    /// The memory used (inclusive childs)
    pub fn mem_usage(&self) -> usize {
        self.columnar_writer.mem_usage()
    }

    pub(crate) fn sort_order(
        &self,
        sort_field: &str,
        num_docs: DocId,
        reversed: bool,
    ) -> Vec<DocId> {
        self.columnar_writer
            .sort_order(sort_field, num_docs, reversed)
    }

    /// Indexes all of the fastfields of a new document.
    pub fn add_document<D: Document>(&mut self, doc: &D) -> crate::Result<()> {
        let doc_id = self.num_docs;
        for (field, value) in doc.iter_fields_and_values() {
            let value_access = value as D::Value<'_>;

            self.add_doc_value(doc_id, field, value_access)?;
        }
        self.num_docs += 1;
        Ok(())
    }

    #[inline]
    fn add_doc_value<'a, V: Value<'a>>(
        &mut self,
        doc_id: DocId,
        field: Field,
        value: V,
    ) -> crate::Result<()> {
        let field_idx = field.field_id() as usize;
        let field_name = match &self.fast_field_names[field_idx] {
            None => return Ok(()),
            Some(name) => name,
        };
        let handle_opt = self.column_handles[field_idx];

        match value.as_value() {
            ReferenceValue::Leaf(leaf) => match leaf {
                ReferenceValueLeaf::Null => {}
                ReferenceValueLeaf::Str(val) => {
                    if let Some(limit) = self.per_field_raw_limit[field_idx] {
                        if val.len() < limit {
                            if let Some(TypedColumnHandle::Str(handle)) = handle_opt {
                                self.columnar_writer.record_str_by_handle(doc_id, handle, val);
                            } else {
                                self.columnar_writer.record_str(doc_id, field_name, val);
                            }
                        }
                    } else if let Some(tokenizer) = &mut self.per_field_tokenizer[field_idx] {
                        let mut token_stream = tokenizer.token_stream(val);
                        if let Some(TypedColumnHandle::Str(handle)) = handle_opt {
                            token_stream.process(&mut |token: &Token| {
                                self.columnar_writer
                                    .record_str_by_handle(doc_id, handle, &token.text);
                            });
                        } else {
                            token_stream.process(&mut |token: &Token| {
                                self.columnar_writer
                                    .record_str(doc_id, field_name, &token.text);
                            });
                        }
                    } else if let Some(TypedColumnHandle::Str(handle)) = handle_opt {
                        self.columnar_writer.record_str_by_handle(doc_id, handle, val);
                    } else {
                        self.columnar_writer.record_str(doc_id, field_name, val);
                    }
                }
                ReferenceValueLeaf::U64(val) => {
                    if let Some(TypedColumnHandle::Numerical(handle)) = handle_opt {
                        self.columnar_writer.record_numerical_by_handle(
                            doc_id,
                            handle,
                            NumericalValue::from(val),
                        );
                    } else {
                        self.columnar_writer.record_numerical(
                            doc_id,
                            field_name,
                            NumericalValue::from(val),
                        );
                    }
                }
                ReferenceValueLeaf::I64(val) => {
                    if let Some(TypedColumnHandle::Numerical(handle)) = handle_opt {
                        self.columnar_writer.record_numerical_by_handle(
                            doc_id,
                            handle,
                            NumericalValue::from(val),
                        );
                    } else {
                        self.columnar_writer.record_numerical(
                            doc_id,
                            field_name,
                            NumericalValue::from(val),
                        );
                    }
                }
                ReferenceValueLeaf::F64(val) => {
                    if let Some(TypedColumnHandle::Numerical(handle)) = handle_opt {
                        self.columnar_writer.record_numerical_by_handle(
                            doc_id,
                            handle,
                            NumericalValue::from(val),
                        );
                    } else {
                        self.columnar_writer.record_numerical(
                            doc_id,
                            field_name,
                            NumericalValue::from(val),
                        );
                    }
                }
                ReferenceValueLeaf::Date(val) => {
                    let date_precision = self.date_precisions[field_idx];
                    let truncated_datetime = val.truncate(date_precision);
                    if let Some(TypedColumnHandle::DateTime(handle)) = handle_opt {
                        self.columnar_writer.record_datetime_by_handle(
                            doc_id,
                            handle,
                            truncated_datetime,
                        );
                    } else {
                        self.columnar_writer
                            .record_datetime(doc_id, field_name, truncated_datetime);
                    }
                }
                ReferenceValueLeaf::Facet(val) => {
                    if let Some(TypedColumnHandle::Str(handle)) = handle_opt {
                        self.columnar_writer.record_str_by_handle(doc_id, handle, val);
                    } else {
                        self.columnar_writer.record_str(doc_id, field_name, val);
                    }
                }
                ReferenceValueLeaf::Bytes(val) => {
                    if let Some(TypedColumnHandle::Bytes(handle)) = handle_opt {
                        self.columnar_writer
                            .record_bytes_by_handle(doc_id, handle, val);
                    } else {
                        self.columnar_writer.record_bytes(doc_id, field_name, val);
                    }
                }
                ReferenceValueLeaf::IpAddr(val) => {
                    if let Some(TypedColumnHandle::IpAddr(handle)) = handle_opt {
                        self.columnar_writer
                            .record_ip_addr_by_handle(doc_id, handle, val);
                    } else {
                        self.columnar_writer.record_ip_addr(doc_id, field_name, val);
                    }
                }
                ReferenceValueLeaf::Bool(val) => {
                    if let Some(TypedColumnHandle::Bool(handle)) = handle_opt {
                        self.columnar_writer.record_bool_by_handle(doc_id, handle, val);
                    } else {
                        self.columnar_writer.record_bool(doc_id, field_name, val);
                    }
                }
                ReferenceValueLeaf::PreTokStr(val) => {
                    if let Some(TypedColumnHandle::Str(handle)) = handle_opt {
                        for token in &val.tokens {
                            self.columnar_writer
                                .record_str_by_handle(doc_id, handle, &token.text);
                        }
                    } else {
                        for token in &val.tokens {
                            self.columnar_writer
                                .record_str(doc_id, field_name, &token.text);
                        }
                    }
                }
                // Custom fields are never fast (`is_fast() == false`), so they are not registered
                // in `fast_field_names` and `add_doc_value` returns early above.
                ReferenceValueLeaf::Custom(_) => {
                    unreachable!("the fast field writer does not support custom field types")
                }
            },
            ReferenceValue::Array(val) => {
                // TODO: Check this is the correct behaviour we want.
                for value in val {
                    self.add_doc_value(doc_id, field, value)?;
                }
            }
            ReferenceValue::Object(val) => {
                let expand_dots = self.expand_dots[field_idx];
                self.json_path_buffer.clear();
                // First field should not be expanded.
                self.json_path_buffer.set_expand_dots(false);
                self.json_path_buffer.push(field_name);
                self.json_path_buffer.set_expand_dots(expand_dots);

                let text_analyzer = &mut self.per_field_tokenizer[field_idx];

                record_json_obj_to_columnar_writer::<V>(
                    doc_id,
                    val,
                    JSON_DEPTH_LIMIT,
                    &mut self.json_path_buffer,
                    &mut self.columnar_writer,
                    text_analyzer,
                );
            }
        }

        Ok(())
    }

    /// Serializes all of the `FastFieldWriter`s by pushing them in
    /// order to the fast field serializer.
    pub fn serialize(
        &mut self,
        wrt: &mut dyn io::Write,
        doc_id_map_opt: Option<&DocIdMapping>,
    ) -> io::Result<()> {
        let num_docs = self.num_docs;
        let old_to_new_row_ids =
            doc_id_map_opt.map(|doc_id_mapping| doc_id_mapping.old_to_new_ids());
        self.columnar_writer
            .serialize(num_docs, old_to_new_row_ids, wrt)?;
        Ok(())
    }
}

fn record_json_obj_to_columnar_writer<'a, V: Value<'a>>(
    doc: DocId,
    json_visitor: V::ObjectIter,
    remaining_depth_limit: usize,
    json_path_buffer: &mut JsonPathWriter,
    columnar_writer: &mut columnar::ColumnarWriter,
    tokenizer: &mut Option<TextAnalyzer>,
) {
    for (key, child) in json_visitor {
        json_path_buffer.push(key);
        record_json_value_to_columnar_writer(
            doc,
            child,
            remaining_depth_limit,
            json_path_buffer,
            columnar_writer,
            tokenizer,
        );
        json_path_buffer.pop();
    }
}

fn record_json_value_to_columnar_writer<'a, V: Value<'a>>(
    doc: DocId,
    json_val: V,
    mut remaining_depth_limit: usize,
    json_path_writer: &mut JsonPathWriter,
    columnar_writer: &mut columnar::ColumnarWriter,
    tokenizer: &mut Option<TextAnalyzer>,
) {
    if remaining_depth_limit == 0 {
        return;
    }
    remaining_depth_limit -= 1;

    match json_val.as_value() {
        ReferenceValue::Leaf(leaf) => match leaf {
            ReferenceValueLeaf::Null => {} // TODO: Handle null
            ReferenceValueLeaf::Str(val) => {
                if let Some(text_analyzer) = tokenizer.as_mut() {
                    let mut token_stream = text_analyzer.token_stream(val);
                    token_stream.process(&mut |token| {
                        columnar_writer.record_str(doc, json_path_writer.as_str(), &token.text);
                    })
                } else {
                    columnar_writer.record_str(doc, json_path_writer.as_str(), val);
                }
            }
            ReferenceValueLeaf::U64(val) => {
                columnar_writer.record_numerical(
                    doc,
                    json_path_writer.as_str(),
                    NumericalValue::from(val),
                );
            }
            ReferenceValueLeaf::I64(val) => {
                columnar_writer.record_numerical(
                    doc,
                    json_path_writer.as_str(),
                    NumericalValue::from(val),
                );
            }
            ReferenceValueLeaf::F64(val) => {
                columnar_writer.record_numerical(
                    doc,
                    json_path_writer.as_str(),
                    NumericalValue::from(val),
                );
            }
            ReferenceValueLeaf::Bool(val) => {
                columnar_writer.record_bool(doc, json_path_writer.as_str(), val);
            }
            ReferenceValueLeaf::Date(val) => {
                columnar_writer.record_datetime(doc, json_path_writer.as_str(), val);
            }
            ReferenceValueLeaf::Facet(_) => {
                unimplemented!("Facet support in dynamic fields is not yet implemented")
            }
            ReferenceValueLeaf::Bytes(_) => {
                // TODO: This can be re added once it is added to the JSON Utils section as well.
                // columnar_writer.record_bytes(doc, json_path_writer.as_str(), val);
                unimplemented!("Bytes support in dynamic fields is not yet implemented")
            }
            ReferenceValueLeaf::IpAddr(_) => {
                unimplemented!("IP address support in dynamic fields is not yet implemented")
            }
            ReferenceValueLeaf::PreTokStr(_) => {
                unimplemented!(
                    "Pre-tokenized string support in dynamic fields is not yet implemented"
                )
            }
            ReferenceValueLeaf::Custom(_) => {
                unimplemented!("the JSON field does not support custom field types")
            }
        },
        ReferenceValue::Array(elements) => {
            for el in elements {
                record_json_value_to_columnar_writer(
                    doc,
                    el,
                    remaining_depth_limit,
                    json_path_writer,
                    columnar_writer,
                    tokenizer,
                );
            }
        }
        ReferenceValue::Object(object) => {
            record_json_obj_to_columnar_writer::<V>(
                doc,
                object,
                remaining_depth_limit,
                json_path_writer,
                columnar_writer,
                tokenizer,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use columnar::{Column, ColumnarReader, ColumnarWriter, StrColumn};
    use common::JsonPathWriter;

    use super::{record_json_value_to_columnar_writer, FastFieldsWriter};
    use crate::fastfield::writer::JSON_DEPTH_LIMIT;
    use crate::schema::{Schema, FAST};
    use crate::DocId;

    #[test]
    fn test_raw_fast_fields_bypass_tokenizer() {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", FAST);
        let json_field = schema_builder.add_json_field("json", FAST);
        let schema = schema_builder.build();

        // `from_schema` uses an empty tokenizer manager, so this also verifies that no
        // tokenizer is looked up for the `FAST` flag.
        let fast_fields_writer = FastFieldsWriter::from_schema(&schema).unwrap();
        assert!(fast_fields_writer.per_field_tokenizer[text_field.field_id() as usize].is_none());
        assert!(fast_fields_writer.per_field_tokenizer[json_field.field_id() as usize].is_none());
    }

    fn test_columnar_from_jsons_aux(
        json_docs: &[serde_json::Value],
        expand_dots: bool,
    ) -> ColumnarReader {
        let mut columnar_writer = ColumnarWriter::default();
        let mut json_path = JsonPathWriter::default();
        json_path.set_expand_dots(expand_dots);
        for (doc, json_doc) in json_docs.iter().enumerate() {
            record_json_value_to_columnar_writer(
                doc as u32,
                json_doc,
                JSON_DEPTH_LIMIT,
                &mut json_path,
                &mut columnar_writer,
                &mut None,
            );
        }
        let mut buffer = Vec::new();
        columnar_writer
            .serialize(json_docs.len() as DocId, None, &mut buffer)
            .unwrap();
        ColumnarReader::open(buffer).unwrap()
    }

    #[test]
    fn test_json_fastfield_record_simple() {
        let json_doc = serde_json::json!({
            "float": 1.02,
            "text": "hello happy tax payer",
            "nested": {"child": 3, "child2": 5},
            "arr": ["hello", "happy", "tax", "payer"]
        });
        let columnar_reader = test_columnar_from_jsons_aux(&[json_doc], false);
        let columns = columnar_reader.list_columns().unwrap();
        {
            assert_eq!(columns[0].0, "arr");
            let column_arr_opt: Option<StrColumn> = columns[0].1.open().unwrap().into();
            assert!(column_arr_opt
                .unwrap()
                .term_ords(0)
                .eq([1, 0, 3, 2].into_iter()));
        }
        {
            assert_eq!(columns[1].0, "float");
            let column_float_opt: Option<Column<f64>> = columns[1].1.open().unwrap().into();
            assert!(column_float_opt
                .unwrap()
                .values_for_doc(0)
                .eq([1.02f64].into_iter()));
        }
        {
            assert_eq!(columns[2].0, "nested\u{1}child");
            let column_nest_child_opt: Option<Column<i64>> = columns[2].1.open().unwrap().into();
            assert!(column_nest_child_opt
                .unwrap()
                .values_for_doc(0)
                .eq([3].into_iter()));
        }
        {
            assert_eq!(columns[3].0, "nested\u{1}child2");
            let column_nest_child2_opt: Option<Column<i64>> = columns[3].1.open().unwrap().into();
            assert!(column_nest_child2_opt
                .unwrap()
                .values_for_doc(0)
                .eq([5].into_iter()));
        }
        {
            assert_eq!(columns[4].0, "text");
            let column_text_opt: Option<StrColumn> = columns[4].1.open().unwrap().into();
            assert!(column_text_opt.unwrap().term_ords(0).eq([0].into_iter()));
        }
    }

    #[test]
    fn test_json_fastfield_deep_obj() {
        let json_doc = serde_json::json!(
            {"a": {"a": {"a": {"a": {"a":
            {"a": {"a": {"a": {"a": {"a":
            {"a": {"a": {"a": {"a": {"a":
            {"a": {"a": {"a": {"depth_accepted": 19, "a": {  "depth_truncated": 20}
        }}}}}}}}}}}}}}}}}}});
        let columnar_reader = test_columnar_from_jsons_aux(&[json_doc], false);
        let columns = columnar_reader.list_columns().unwrap();
        assert_eq!(columns.len(), 1);
        assert!(columns[0].0.ends_with("a\u{1}a\u{1}a\u{1}depth_accepted"));
    }

    #[test]
    fn test_json_fastfield_deep_arr() {
        let json_doc = json!(
        {"obj":
        [[[[[,
        [[[[[,
        [[[[[,
        [[18, [19, //< within limits
        [20]]]]]]]]]]]]]]]]]]]});
        let columnar_reader = test_columnar_from_jsons_aux(&[json_doc], false);
        let columns = columnar_reader.list_columns().unwrap();
        assert_eq!(columns.len(), 1);
        assert_eq!(columns[0].0, "obj");
        let dynamic_column = columns[0].1.open().unwrap();
        let col: Option<Column<i64>> = dynamic_column.into();
        let vals: Vec<i64> = col.unwrap().values_for_doc(0).collect();
        assert_eq!(&vals, &[18, 19])
    }

    #[test]
    fn test_json_fast_field_do_not_expand_dots() {
        let json_doc = json!({"field.with.dots": {"child.with.dot": "hello"}});
        let columnar_reader = test_columnar_from_jsons_aux(&[json_doc], false);
        let columns = columnar_reader.list_columns().unwrap();
        assert_eq!(columns.len(), 1);
        assert_eq!(columns[0].0, "field.with.dots\u{1}child.with.dot");
    }

    #[test]
    fn test_json_fast_field_expand_dots() {
        let json_doc = json!({"field.with.dots": {"child.with.dot": "hello"}});
        let columnar_reader = test_columnar_from_jsons_aux(&[json_doc], true);
        let columns = columnar_reader.list_columns().unwrap();
        assert_eq!(columns.len(), 1);
        assert_eq!(
            columns[0].0,
            "field\u{1}with\u{1}dots\u{1}child\u{1}with\u{1}dot"
        );
    }
}
