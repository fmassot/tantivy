use std::borrow::Cow;
use std::io;
use std::io::Write;

use columnar::MonotonicallyMappableToU128;
use common::{f64_to_u64, BinarySerializable, VInt};

#[cfg(test)]
use super::OwnedValue;
use super::ReferenceValueLeaf;
use crate::schema::document::{type_codes, Document, ReferenceValue, Value};
use crate::schema::Schema;

/// A serializer writing documents which implement [`Document`] to a provided writer.
#[cfg(test)]
pub struct BinaryDocumentSerializer<'se, W> {
    writer: &'se mut W,
    schema: &'se Schema,
}

#[cfg(test)]
impl<'se, W> BinaryDocumentSerializer<'se, W>
where W: Write
{
    /// Creates a new serializer with a provided writer.
    pub(crate) fn new(writer: &'se mut W, schema: &'se Schema) -> Self {
        Self { writer, schema }
    }

    /// Attempts to serialize a given document and write the output
    /// to the writer.
    #[inline]
    pub(crate) fn serialize_doc<D>(&mut self, doc: &D) -> io::Result<()>
    where D: Document {
        let stored_field_values = || {
            doc.iter_fields_and_values()
                .filter(|(field, _)| self.schema.get_field_entry(*field).is_stored())
        };
        let num_field_values = stored_field_values().count();
        let mut actual_length = 0;

        VInt(num_field_values as u64).serialize(self.writer)?;
        for (field, value_access) in stored_field_values() {
            field.serialize(self.writer)?;

            let mut serializer = BinaryValueSerializer::new(self.writer);
            match value_access.as_value() {
                ReferenceValue::Leaf(ReferenceValueLeaf::PreTokStr(pre_tokenized_text)) => {
                    serializer.serialize_value(ReferenceValue::Leaf::<&'_ OwnedValue>(
                        ReferenceValueLeaf::Str(&pre_tokenized_text.text),
                    ))?;
                }
                _ => {
                    serializer.serialize_value(value_access.as_value())?;
                }
            }

            actual_length += 1;
        }

        if num_field_values != actual_length {
            return Err(io::Error::other(format!(
                "Unexpected number of entries written to serializer, expected {num_field_values} \
                 entries, got {actual_length} entries",
            )));
        }

        Ok(())
    }
}

#[inline(always)]
fn push_field_and_u64(out: &mut Vec<u8>, field_id: u32, code: u8, val: u64) {
    let f = field_id.to_le_bytes();
    let b = val.to_le_bytes();
    out.extend_from_slice(&[
        f[0], f[1], f[2], f[3], code, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]);
}

#[inline(always)]
fn push_field_and_bytes(out: &mut Vec<u8>, field_id: u32, code: u8, bytes: &[u8]) {
    let f = field_id.to_le_bytes();
    let len = bytes.len();
    if len < 128 {
        out.reserve(6 + len);
        out.extend_from_slice(&[f[0], f[1], f[2], f[3], code, (len as u8) | 0x80]);
        out.extend_from_slice(bytes);
    } else {
        let mut vint_buf = [0u8; 8];
        let len_bytes = common::serialize_vint_u32(len as u32, &mut vint_buf);
        out.reserve(5 + len_bytes.len() + len);
        out.extend_from_slice(&[f[0], f[1], f[2], f[3], code]);
        out.extend_from_slice(len_bytes);
        out.extend_from_slice(bytes);
    }
}

/// Fast single-pass document serializer directly into a `&mut Vec<u8>`.
///
/// Avoids the double iteration over `doc.iter_fields_and_values()` and per-write `io::Result`
/// error plumbing when writing into an in-memory buffer (as used by `StoreWriter` and
/// `TempStoreWriter`).
#[inline]
pub(crate) fn serialize_doc_to_vec<D: Document>(
    doc: &D,
    schema: &Schema,
    out: &mut Vec<u8>,
) -> io::Result<()> {
    let iter = doc.iter_fields_and_values();
    let (size_hint, _) = iter.size_hint();
    if size_hint > 0 {
        out.reserve(1 + size_hint.saturating_mul(14));
    }
    let count_pos = out.len();
    // Reserve 1 byte for the common case (< 128 stored field values).
    out.push(0);
    let mut num_field_values: usize = 0;

    for (field, value_access) in iter {
        if !schema.get_field_entry(field).is_stored() {
            continue;
        }
        let field_id = field.field_id();
        match value_access.as_value() {
            ReferenceValue::Leaf(leaf) => match leaf {
                ReferenceValueLeaf::Str(val) => {
                    push_field_and_bytes(out, field_id, type_codes::TEXT_CODE, val.as_bytes());
                }
                ReferenceValueLeaf::I64(val) => {
                    push_field_and_u64(out, field_id, type_codes::I64_CODE, val as u64);
                }
                ReferenceValueLeaf::U64(val) => {
                    push_field_and_u64(out, field_id, type_codes::U64_CODE, val);
                }
                ReferenceValueLeaf::F64(val) => {
                    push_field_and_u64(out, field_id, type_codes::F64_CODE, f64_to_u64(val));
                }
                ReferenceValueLeaf::Date(val) => {
                    push_field_and_u64(
                        out,
                        field_id,
                        type_codes::DATE_CODE,
                        val.into_timestamp_nanos() as u64,
                    );
                }
                ReferenceValueLeaf::Bool(val) => {
                    let f = field_id.to_le_bytes();
                    out.extend_from_slice(&[f[0], f[1], f[2], f[3], type_codes::BOOL_CODE, val as u8]);
                }
                ReferenceValueLeaf::Null => {
                    let f = field_id.to_le_bytes();
                    out.extend_from_slice(&[f[0], f[1], f[2], f[3], type_codes::NULL_CODE]);
                }
                ReferenceValueLeaf::Bytes(val) => {
                    push_field_and_bytes(out, field_id, type_codes::BYTES_CODE, val);
                }
                ReferenceValueLeaf::Facet(val) => {
                    push_field_and_bytes(
                        out,
                        field_id,
                        type_codes::HIERARCHICAL_FACET_CODE,
                        val.as_bytes(),
                    );
                }
                ReferenceValueLeaf::PreTokStr(pre_tokenized_text) => {
                    push_field_and_bytes(
                        out,
                        field_id,
                        type_codes::TEXT_CODE,
                        pre_tokenized_text.text.as_bytes(),
                    );
                }
                ReferenceValueLeaf::IpAddr(val) => {
                    let f = field_id.to_le_bytes();
                    out.extend_from_slice(&[f[0], f[1], f[2], f[3], type_codes::IP_CODE]);
                    out.extend_from_slice(&val.to_u128().to_le_bytes());
                }
                ReferenceValueLeaf::Custom(_) => {
                    return Err(io::Error::other(
                        "the store does not support custom field types",
                    ));
                }
            },
            val => {
                out.extend_from_slice(&field_id.to_le_bytes());
                serialize_value_to_vec(out, val)?;
            }
        }
        num_field_values += 1;
    }

    if num_field_values < 128 {
        out[count_pos] = (num_field_values as u8) | 0x80;
    } else {
        let mut vint_buf = [0u8; 8];
        let len_bytes = common::serialize_vint_u32(num_field_values as u32, &mut vint_buf);
        out.splice(count_pos..count_pos + 1, len_bytes.iter().copied());
    }
    Ok(())
}

#[inline(always)]
fn push_collection_header(out: &mut Vec<u8>, code: u8, len: usize) {
    if len < 128 {
        out.extend_from_slice(&[code, (len as u8) | 0x80]);
    } else {
        let mut vint_buf = [0u8; 8];
        let len_bytes = common::serialize_vint_u32(len as u32, &mut vint_buf);
        out.push(code);
        out.extend_from_slice(len_bytes);
    }
}

#[inline(always)]
fn push_bytes_with_type_code(out: &mut Vec<u8>, code: u8, bytes: &[u8]) {
    let len = bytes.len();
    if len < 128 {
        out.reserve(2 + len);
        out.extend_from_slice(&[code, (len as u8) | 0x80]);
        out.extend_from_slice(bytes);
    } else {
        let mut vint_buf = [0u8; 8];
        let len_bytes = common::serialize_vint_u32(len as u32, &mut vint_buf);
        out.reserve(1 + len_bytes.len() + len);
        out.push(code);
        out.extend_from_slice(len_bytes);
        out.extend_from_slice(bytes);
    }
}

#[inline(always)]
fn push_u64_with_type_code(out: &mut Vec<u8>, code: u8, val: u64) {
    let b = val.to_le_bytes();
    out.extend_from_slice(&[code, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
}

#[inline]
fn serialize_value_to_vec<'a, V: Value<'a>>(
    out: &mut Vec<u8>,
    value: ReferenceValue<'a, V>,
) -> io::Result<()> {
    match value {
        ReferenceValue::Leaf(leaf) => match leaf {
            ReferenceValueLeaf::Null => out.push(type_codes::NULL_CODE),
            ReferenceValueLeaf::Str(val) => {
                push_bytes_with_type_code(out, type_codes::TEXT_CODE, val.as_bytes());
            }
            ReferenceValueLeaf::U64(val) => {
                push_u64_with_type_code(out, type_codes::U64_CODE, val);
            }
            ReferenceValueLeaf::I64(val) => {
                push_u64_with_type_code(out, type_codes::I64_CODE, val as u64);
            }
            ReferenceValueLeaf::F64(val) => {
                push_u64_with_type_code(out, type_codes::F64_CODE, f64_to_u64(val));
            }
            ReferenceValueLeaf::Date(val) => {
                push_u64_with_type_code(
                    out,
                    type_codes::DATE_CODE,
                    val.into_timestamp_nanos() as u64,
                );
            }
            ReferenceValueLeaf::Facet(val) => {
                push_bytes_with_type_code(out, type_codes::HIERARCHICAL_FACET_CODE, val.as_bytes());
            }
            ReferenceValueLeaf::Bytes(val) => {
                push_bytes_with_type_code(out, type_codes::BYTES_CODE, val);
            }
            ReferenceValueLeaf::IpAddr(val) => {
                out.push(type_codes::IP_CODE);
                out.extend_from_slice(&val.to_u128().to_le_bytes());
            }
            ReferenceValueLeaf::Bool(val) => {
                out.extend_from_slice(&[type_codes::BOOL_CODE, val as u8]);
            }
            ReferenceValueLeaf::PreTokStr(val) => {
                out.push(type_codes::EXT_CODE);
                out.push(type_codes::TOK_STR_EXT_CODE);
                BinarySerializable::serialize(&*val, out)?;
            }
            ReferenceValueLeaf::Custom(_) => {
                return Err(io::Error::other(
                    "the store does not support custom field types",
                ));
            }
        },
        ReferenceValue::Array(elements) => {
            let (lower, upper) = elements.size_hint();
            if upper == Some(lower) {
                push_collection_header(out, type_codes::ARRAY_CODE, lower);
                let mut actual_length = 0;
                for elem in elements {
                    serialize_value_to_vec(out, elem.as_value())?;
                    actual_length += 1;
                }
                if lower != actual_length {
                    return Err(io::Error::other(format!(
                        "Unexpected number of entries written to serializer, expected {lower} \
                         entries, got {actual_length} entries",
                    )));
                }
            } else {
                let elements: Vec<V> = elements.collect();
                push_collection_header(out, type_codes::ARRAY_CODE, elements.len());
                for elem in elements {
                    serialize_value_to_vec(out, elem.as_value())?;
                }
            }
        }
        ReferenceValue::Object(object) => {
            let (lower, upper) = object.size_hint();
            if upper == Some(lower) {
                push_collection_header(out, type_codes::OBJECT_CODE, lower * 2);
                let mut actual_length = 0;
                for (key, val) in object {
                    push_bytes_with_type_code(out, type_codes::TEXT_CODE, key.as_bytes());
                    serialize_value_to_vec(out, val.as_value())?;
                    actual_length += 1;
                }
                if lower != actual_length {
                    return Err(io::Error::other(format!(
                        "Unexpected number of entries written to serializer, expected {lower} \
                         entries, got {actual_length} entries",
                    )));
                }
            } else {
                let entries: Vec<(&str, V)> = object.collect();
                push_collection_header(out, type_codes::OBJECT_CODE, entries.len() * 2);
                for (key, val) in entries {
                    push_bytes_with_type_code(out, type_codes::TEXT_CODE, key.as_bytes());
                    serialize_value_to_vec(out, val.as_value())?;
                }
            }
        }
    }
    Ok(())
}

/// A serializer for a single value.
pub struct BinaryValueSerializer<'se, W> {
    writer: &'se mut W,
}

impl<'se, W> BinaryValueSerializer<'se, W>
where W: Write
{
    /// Creates a new serializer with a provided writer.
    pub(crate) fn new(writer: &'se mut W) -> Self {
        Self { writer }
    }

    fn serialize_with_type_code<T: BinarySerializable>(
        &mut self,
        code: u8,
        val: &T,
    ) -> io::Result<()> {
        self.write_type_code(code)?;
        BinarySerializable::serialize(val, self.writer)
    }

    #[inline]
    fn serialize_u64_with_type_code(&mut self, code: u8, val: u64) -> io::Result<()> {
        let b = val.to_le_bytes();
        self.writer
            .write_all(&[code, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }

    #[inline]
    fn serialize_collection_header(&mut self, code: u8, len: usize) -> io::Result<()> {
        if len < 128 {
            // Tantivy VInt sets the high bit (0x80) on the final byte.
            self.writer.write_all(&[code, (len as u8) | 0x80])
        } else {
            let mut vint_buf = [0u8; 8];
            let len_bytes = common::serialize_vint_u32(len as u32, &mut vint_buf);
            let mut header = [0u8; 9];
            header[0] = code;
            header[1..1 + len_bytes.len()].copy_from_slice(len_bytes);
            self.writer.write_all(&header[..1 + len_bytes.len()])
        }
    }

    #[inline]
    fn serialize_bytes_with_type_code(&mut self, code: u8, bytes: &[u8]) -> io::Result<()> {
        self.serialize_collection_header(code, bytes.len())?;
        self.writer.write_all(bytes)
    }

    /// Attempts to serialize a given value and write the output
    /// to the writer.
    #[inline]
    pub(crate) fn serialize_value<'a, V>(
        &mut self,
        value: ReferenceValue<'a, V>,
    ) -> io::Result<()>
    where
        V: Value<'a>,
    {
        match value {
            ReferenceValue::Leaf(leaf) => match leaf {
                ReferenceValueLeaf::Null => self.write_type_code(type_codes::NULL_CODE),
                ReferenceValueLeaf::Str(val) => {
                    self.serialize_bytes_with_type_code(type_codes::TEXT_CODE, val.as_bytes())
                }
                ReferenceValueLeaf::U64(val) => {
                    self.serialize_u64_with_type_code(type_codes::U64_CODE, val)
                }
                ReferenceValueLeaf::I64(val) => {
                    self.serialize_u64_with_type_code(type_codes::I64_CODE, val as u64)
                }
                ReferenceValueLeaf::F64(val) => {
                    self.serialize_u64_with_type_code(type_codes::F64_CODE, f64_to_u64(val))
                }
                ReferenceValueLeaf::Date(val) => {
                    let timestamp_nanos: i64 = val.into_timestamp_nanos();
                    self.serialize_u64_with_type_code(type_codes::DATE_CODE, timestamp_nanos as u64)
                }
                ReferenceValueLeaf::Facet(val) => self.serialize_with_type_code(
                    type_codes::HIERARCHICAL_FACET_CODE,
                    &Cow::Borrowed(val),
                ),
                ReferenceValueLeaf::Bytes(val) => {
                    self.serialize_bytes_with_type_code(type_codes::BYTES_CODE, val)
                }
                ReferenceValueLeaf::IpAddr(val) => {
                    self.serialize_with_type_code(type_codes::IP_CODE, &val.to_u128())
                }
                ReferenceValueLeaf::Bool(val) => {
                    self.writer.write_all(&[type_codes::BOOL_CODE, val as u8])
                }
                ReferenceValueLeaf::PreTokStr(val) => {
                    self.write_type_code(type_codes::EXT_CODE)?;
                    self.serialize_with_type_code(type_codes::TOK_STR_EXT_CODE, &*val)
                }
                // Custom fields are never stored (`is_stored() == false`), so the document
                // serializer never reaches this arm. Guard it explicitly rather than inventing an
                // on-disk type code for an opaque, plugin-owned payload.
                ReferenceValueLeaf::Custom(_) => Err(io::Error::other(
                    "the store does not support custom field types",
                )),
            },
            ReferenceValue::Array(elements) => {
                let (lower, upper) = elements.size_hint();
                if upper == Some(lower) {
                    self.serialize_collection_header(type_codes::ARRAY_CODE, lower)?;
                    let mut actual_length = 0;
                    for value in elements {
                        self.serialize_value(value.as_value())?;
                        actual_length += 1;
                    }
                    if lower != actual_length {
                        return Err(io::Error::other(format!(
                            "Unexpected number of entries written to serializer, expected {lower} \
                             entries, got {actual_length} entries",
                        )));
                    }
                    Ok(())
                } else {
                    let elements: Vec<V> = elements.collect();
                    self.serialize_collection_header(type_codes::ARRAY_CODE, elements.len())?;
                    for value in elements {
                        self.serialize_value(value.as_value())?;
                    }
                    Ok(())
                }
            }
            ReferenceValue::Object(object) => {
                let (lower, upper) = object.size_hint();
                if upper == Some(lower) {
                    // Keys and values are counted separately (`length * 2`).
                    self.serialize_collection_header(type_codes::OBJECT_CODE, lower * 2)?;
                    let mut actual_length = 0;
                    for (key, value) in object {
                        self.serialize_bytes_with_type_code(type_codes::TEXT_CODE, key.as_bytes())?;
                        self.serialize_value(value.as_value())?;
                        actual_length += 1;
                    }
                    if lower != actual_length {
                        return Err(io::Error::other(format!(
                            "Unexpected number of entries written to serializer, expected {lower} \
                             entries, got {actual_length} entries",
                        )));
                    }
                    Ok(())
                } else {
                    let entries: Vec<(&str, V)> = object.collect();
                    self.serialize_collection_header(type_codes::OBJECT_CODE, entries.len() * 2)?;
                    for (key, value) in entries {
                        self.serialize_bytes_with_type_code(type_codes::TEXT_CODE, key.as_bytes())?;
                        self.serialize_value(value.as_value())?;
                    }
                    Ok(())
                }
            }
        }
    }

    fn write_type_code(&mut self, code: u8) -> io::Result<()> {
        code.serialize(self.writer)
    }
}

/// A serializer for writing a sequence of values to a writer.
pub struct BinaryArraySerializer<'se, W> {
    writer: &'se mut W,
    expected_length: usize,
    actual_length: usize,
}

impl<'se, W> BinaryArraySerializer<'se, W>
where W: Write
{
    /// Creates a new array serializer and writes the length of the array to the writer.
    pub(crate) fn begin(length: usize, writer: &'se mut W) -> io::Result<Self> {
        VInt(length as u64).serialize(writer)?;

        Ok(Self {
            writer,
            expected_length: length,
            actual_length: 0,
        })
    }

    /// Attempts to serialize a given value and write the output
    /// to the writer.
    pub(crate) fn serialize_value<'a, V>(
        &mut self,
        value: ReferenceValue<'a, V>,
    ) -> io::Result<()>
    where
        V: Value<'a>,
    {
        let mut serializer = BinaryValueSerializer::new(self.writer);
        serializer.serialize_value(value)?;

        self.actual_length += 1;
        Ok(())
    }

    /// Finishes writing the array to the writer and validates it.
    pub(crate) fn end(self) -> io::Result<()> {
        if self.expected_length != self.actual_length {
            return Err(io::Error::other(format!(
                "Unexpected number of entries written to serializer, expected {} entries, got {} \
                 entries",
                self.expected_length, self.actual_length,
            )));
        }
        Ok(())
    }
}

/// A serializer for writing a set of key-value pairs to a writer.
pub struct BinaryObjectSerializer<'se, W> {
    inner: BinaryArraySerializer<'se, W>,
    expected_length: usize,
    actual_length: usize,
}

impl<'se, W> BinaryObjectSerializer<'se, W>
where W: Write
{
    /// Creates a new object serializer and writes the length of the object to the writer.
    pub(crate) fn begin(length: usize, writer: &'se mut W) -> io::Result<Self> {
        // We mul by 2 here to count the keys and values separately:
        // [("a", 1), ("b", 2)] is actually stored as ["a", 1, "b", 2]
        let inner = BinaryArraySerializer::begin(length * 2, writer)?;

        Ok(Self {
            inner,
            expected_length: length,
            actual_length: 0,
        })
    }

    /// Attempts to serialize a given value and write the output
    /// to the writer.
    pub(crate) fn serialize_entry<'a, V>(
        &mut self,
        key: &'a str,
        value: ReferenceValue<'a, V>,
    ) -> io::Result<()>
    where
        V: Value<'a>,
    {
        // Keys and values are stored inline with one another.
        let mut key_serializer = BinaryValueSerializer::new(self.inner.writer);
        key_serializer.serialize_bytes_with_type_code(type_codes::TEXT_CODE, key.as_bytes())?;
        self.inner.actual_length += 1;
        self.inner.serialize_value(value)?;

        self.actual_length += 1;
        Ok(())
    }

    /// Finishes writing the array to the writer and validates it.
    pub(crate) fn end(self) -> io::Result<()> {
        if self.expected_length != self.actual_length {
            return Err(io::Error::other(format!(
                "Unexpected number of entries written to serializer, expected {} entries, got {} \
                 entries",
                self.expected_length, self.actual_length,
            )));
        }

        // This should never fail if the above statement is valid.
        self.inner.end()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::Number;
    use tokenizer_api::Token;

    use super::*;
    use crate::schema::document::existing_type_impls::JsonObjectIter;
    use crate::schema::{Facet, Field, FAST, STORED, TEXT};
    use crate::tokenizer::PreTokenizedString;

    fn serialize_value<'a>(value: ReferenceValue<'a, &'a serde_json::Value>) -> Vec<u8> {
        let mut writer = Vec::new();

        let mut serializer = BinaryValueSerializer::new(&mut writer);
        serializer.serialize_value(value).expect("Serialize value");

        writer
    }

    /// A macro for defining the expected binary representation
    /// of the serialized values in a somewhat human readable way.
    macro_rules! binary_repr {
        ($( $type_code:expr $(, $ext_code:expr)? => $value:expr $(,)?)*) => {{
            let mut writer = Vec::new();

            $(
                $type_code.serialize(&mut writer).unwrap();

                $(
                    $ext_code.serialize(&mut writer).unwrap();
                )?

                BinarySerializable::serialize(
                    &$value,
                    &mut writer,
                ).unwrap();
            )*

            writer
        }};
        (collection $code:expr, length $len:expr, $( $type_code:expr $(, $ext_code:expr)? => $value:expr $(,)?)*) => {{
            let mut writer = Vec::new();

            $code.serialize(&mut writer).unwrap();
            VInt($len as u64).serialize(&mut writer).unwrap();

            $(
                $type_code.serialize(&mut writer).unwrap();

                $(
                    $ext_code.serialize(&mut writer).unwrap();
                )?

                BinarySerializable::serialize(
                    &$value,
                    &mut writer,
                ).unwrap();
            )*

            writer
        }};
    }

    #[test]
    fn test_simple_value_serialize() {
        let result = serialize_value(ReferenceValueLeaf::Null.into());
        let expected = binary_repr!(
            type_codes::NULL_CODE => (),
        );
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );

        let result = serialize_value(ReferenceValueLeaf::Str("hello, world").into());
        let expected = binary_repr!(
            type_codes::TEXT_CODE => String::from("hello, world"),
        );
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );

        let result = serialize_value(ReferenceValueLeaf::U64(123).into());
        let expected = binary_repr!(
            type_codes::U64_CODE => 123u64,
        );
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );

        let result = serialize_value(ReferenceValueLeaf::I64(-123).into());
        let expected = binary_repr!(
            type_codes::I64_CODE => -123i64,
        );
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );

        let result = serialize_value(ReferenceValueLeaf::F64(123.3845f64).into());
        let expected = binary_repr!(
            type_codes::F64_CODE => f64_to_u64(123.3845f64),
        );
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );

        let result = serialize_value(ReferenceValueLeaf::Bool(false).into());
        let expected = binary_repr!(
            type_codes::BOOL_CODE => false,
        );
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );

        let facet = Facet::from_text("/hello/world").unwrap();
        let result = serialize_value(ReferenceValueLeaf::Facet(facet.encoded_str()).into());
        let expected = binary_repr!(
            type_codes::HIERARCHICAL_FACET_CODE => Facet::from_text("/hello/world").unwrap(),
        );
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );

        let pre_tok_str = PreTokenizedString {
            text: "hello, world".to_string(),
            tokens: vec![Token::default(), Token::default()],
        };
        let result =
            serialize_value(ReferenceValueLeaf::PreTokStr(pre_tok_str.clone().into()).into());
        let expected = binary_repr!(
            type_codes::EXT_CODE, type_codes::TOK_STR_EXT_CODE => pre_tok_str,
        );
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );
    }

    #[test]
    fn test_array_serialize() {
        let elements = [serde_json::Value::Null, serde_json::Value::Null];
        let result = serialize_value(ReferenceValue::Array(elements.iter()));
        let expected = binary_repr!(
            collection type_codes::ARRAY_CODE,
            length elements.len(),
            type_codes::NULL_CODE => (),
            type_codes::NULL_CODE => (),
        );
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );

        let elements = [
            serde_json::Value::String("Hello, world".into()),
            serde_json::Value::String("Some demo".into()),
        ];
        let result = serialize_value(ReferenceValue::Array(elements.iter()));
        let expected = binary_repr!(
            collection type_codes::ARRAY_CODE,
            length elements.len(),
            type_codes::TEXT_CODE => String::from("Hello, world"),
            type_codes::TEXT_CODE => String::from("Some demo"),
        );
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );

        let elements = [];
        let result = serialize_value(ReferenceValue::Array(elements.iter()));
        let expected = binary_repr!(
            collection type_codes::ARRAY_CODE,
            length elements.len(),
        );
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );

        let elements = [
            serde_json::Value::Null,
            serde_json::Value::String("Hello, world".into()),
            serde_json::Value::Number(12345.into()),
        ];
        let result = serialize_value(ReferenceValue::Array(elements.iter()));
        let expected = binary_repr!(
            collection type_codes::ARRAY_CODE,
            length elements.len(),
            type_codes::NULL_CODE => (),
            type_codes::TEXT_CODE => String::from("Hello, world"),
            type_codes::I64_CODE => 12345i64,
        );
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );
    }

    #[test]
    fn test_object_serialize() {
        let mut object = serde_json::Map::new();
        object.insert(
            "my-first-key".into(),
            serde_json::Value::String("Hello".into()),
        );
        object.insert("my-second-key".into(), serde_json::Value::Null);
        object.insert(
            "my-third-key".into(),
            serde_json::Value::Number(Number::from_f64(123.0).unwrap()),
        );
        let result = serialize_value(ReferenceValue::Object(JsonObjectIter(object.iter())));
        let expected = binary_repr!(
            collection type_codes::OBJECT_CODE,
            length object.len() * 2,  // To account for keys counting towards the length
            type_codes::TEXT_CODE => String::from("my-first-key"),
            type_codes::TEXT_CODE => String::from("Hello"),
            type_codes::TEXT_CODE => String::from("my-second-key"),
            type_codes::NULL_CODE => (),
            type_codes::TEXT_CODE => String::from("my-third-key"),
            type_codes::F64_CODE => f64_to_u64(123.0),
        );
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );

        let object = serde_json::Map::new();
        let result = serialize_value(ReferenceValue::Object(JsonObjectIter(object.iter())));
        let expected = binary_repr!(
            collection type_codes::OBJECT_CODE,
            length object.len(),
        );
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );

        let mut object = serde_json::Map::new();
        object.insert("my-first-key".into(), serde_json::Value::Null);
        object.insert("my-second-key".into(), serde_json::Value::Null);
        object.insert("my-third-key".into(), serde_json::Value::Null);
        let result = serialize_value(ReferenceValue::Object(JsonObjectIter(object.iter())));
        let expected = binary_repr!(
            collection type_codes::OBJECT_CODE,
            length object.len() * 2, // To account for keys counting towards the length
            type_codes::TEXT_CODE => String::from("my-first-key"),
            type_codes::NULL_CODE => (),
            type_codes::TEXT_CODE => String::from("my-second-key"),
            type_codes::NULL_CODE => (),
            type_codes::TEXT_CODE => String::from("my-third-key"),
            type_codes::NULL_CODE => (),
        );
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );
    }

    #[test]
    fn test_nested_serialize() {
        let mut object = serde_json::Map::new();
        object.insert(
            "my-array".into(),
            serde_json::Value::Array(vec![
                serde_json::Value::Null,
                serde_json::Value::String(String::from("bobby of the sea")),
            ]),
        );
        object.insert(
            "my-object".into(),
            serde_json::Value::Object(
                vec![
                    (
                        "inner-1".to_string(),
                        serde_json::Value::Number((-123i64).into()),
                    ),
                    (
                        "inner-2".to_string(),
                        serde_json::Value::String(String::from("bobby of the sea 2")),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
        );
        let result = serialize_value(ReferenceValue::Object(JsonObjectIter(object.iter())));

        let mut expected = Vec::new();
        let header = binary_repr!(
            collection type_codes::OBJECT_CODE,
            length object.len() * 2,
        );
        expected.extend_from_slice(&header);
        expected
            .extend_from_slice(&binary_repr!(type_codes::TEXT_CODE => String::from("my-array")));
        let nested_array = binary_repr!(
            collection type_codes::ARRAY_CODE,
            length 2,
            type_codes::NULL_CODE => (),
            type_codes::TEXT_CODE => String::from("bobby of the sea"),
        );
        expected.extend_from_slice(&nested_array);
        expected
            .extend_from_slice(&binary_repr!(type_codes::TEXT_CODE => String::from("my-object")));
        let nested_object = binary_repr!(
            collection type_codes::OBJECT_CODE,
            length 4,   // 2 keys, 2 values
            type_codes::TEXT_CODE => String::from("inner-1"),
            type_codes::I64_CODE => -123i64,
            type_codes::TEXT_CODE => String::from("inner-2"),
            type_codes::TEXT_CODE => String::from("bobby of the sea 2"),
        );
        expected.extend_from_slice(&nested_object);
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );

        // Some more extreme nesting that might behave weirdly
        let mut object = serde_json::Map::new();
        object.insert(
            "my-array".into(),
            serde_json::Value::Array(vec![serde_json::Value::Array(vec![
                serde_json::Value::Array(vec![]),
                serde_json::Value::Array(vec![serde_json::Value::Null]),
            ])]),
        );
        let result = serialize_value(ReferenceValue::Object(JsonObjectIter(object.iter())));

        let mut expected = Vec::new();
        let header = binary_repr!(
            collection type_codes::OBJECT_CODE,
            length object.len() * 2,
        );
        expected.extend_from_slice(&header);
        expected
            .extend_from_slice(&binary_repr!(type_codes::TEXT_CODE => String::from("my-array")));
        let nested_array = binary_repr!(
            collection type_codes::ARRAY_CODE,
            length 1,
        );
        expected.extend_from_slice(&nested_array);
        let nested_array = binary_repr!(
            collection type_codes::ARRAY_CODE,
            length 2,
        );
        expected.extend_from_slice(&nested_array);
        let nested_array = binary_repr!(
            collection type_codes::ARRAY_CODE,
            length 0,
        );
        expected.extend_from_slice(&nested_array);
        let nested_array = binary_repr!(
            collection type_codes::ARRAY_CODE,
            length 1,
            type_codes::NULL_CODE => (),
        );
        expected.extend_from_slice(&nested_array);
        assert_eq!(
            result, expected,
            "Expected serialized value to match the binary representation"
        );
    }

    #[inline]
    fn serialize_doc<D: Document>(doc: &D, schema: &Schema) -> Vec<u8> {
        let mut writer = Vec::new();

        let mut serializer = BinaryDocumentSerializer::new(&mut writer, schema);
        serializer.serialize_doc(doc).expect("Serialize value");

        let mut fast_writer = Vec::new();
        serialize_doc_to_vec(doc, schema, &mut fast_writer).expect("Fast serialize doc");
        assert_eq!(
            fast_writer, writer,
            "serialize_doc_to_vec must produce byte-for-byte identical output to \
             BinaryDocumentSerializer"
        );

        writer
    }

    /// A helper macro for generating the expected binary representation of the document.
    macro_rules! expected_doc_data {
        (length $len:expr) => {{
            let mut writer = Vec::new();
            VInt($len as u64).serialize(&mut writer).unwrap();
            writer
        }};
        (length $len:expr, $( $field_id:expr => $value:expr $(,)?)*) => {{
            let mut writer = Vec::new();

            VInt($len as u64).serialize(&mut writer).unwrap();
            $(
                $field_id.serialize(&mut writer).unwrap();
                $value.serialize(&mut writer).unwrap();
            )*

            writer
        }};
    }

    #[test]
    fn test_document_serialize() {
        let mut builder = Schema::builder();
        let name = builder.add_text_field("name", TEXT | STORED);
        let age = builder.add_u64_field("age", FAST | STORED);
        let schema = builder.build();

        let mut document = BTreeMap::new();
        document.insert(name, crate::schema::OwnedValue::Str("ChillFish8".into()));
        document.insert(age, crate::schema::OwnedValue::U64(20));

        let result = serialize_doc(&document, &schema);
        let mut expected = expected_doc_data!(length document.len());
        name.serialize(&mut expected).unwrap();
        expected
            .extend_from_slice(&binary_repr!(type_codes::TEXT_CODE => String::from("ChillFish8")));
        age.serialize(&mut expected).unwrap();
        expected.extend_from_slice(&binary_repr!(type_codes::U64_CODE => 20u64));
        assert_eq!(
            result, expected,
            "Expected serialized document to match the binary representation"
        );

        let mut builder = Schema::builder();
        let name = builder.add_text_field("name", TEXT | STORED);
        // This should be skipped when serializing.
        let age = builder.add_u64_field("age", FAST);
        let schema = builder.build();

        let mut document = BTreeMap::new();
        document.insert(name, crate::schema::OwnedValue::Str("ChillFish8".into()));
        document.insert(age, crate::schema::OwnedValue::U64(20));

        let result = serialize_doc(&document, &schema);
        let mut expected = expected_doc_data!(length 1);
        name.serialize(&mut expected).unwrap();
        expected
            .extend_from_slice(&binary_repr!(type_codes::TEXT_CODE => String::from("ChillFish8")));
        assert_eq!(
            result, expected,
            "Expected serialized document to match the binary representation"
        );

        let builder = Schema::builder();
        let schema = builder.build();
        let document = BTreeMap::<Field, crate::schema::OwnedValue>::new();
        let result = serialize_doc(&document, &schema);
        let expected = expected_doc_data!(length document.len());
        assert_eq!(
            result, expected,
            "Expected serialized document to match the binary representation"
        );
    }
}
