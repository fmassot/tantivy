use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{self, Read, Write};
use std::net::Ipv6Addr;

use columnar::MonotonicallyMappableToU128;
use common::{read_u32_vint_no_advance, serialize_vint_u32, BinarySerializable, DateTime, VInt};
use serde_json::Map;
use smallvec::SmallVec;
pub use CompactDoc as TantivyDocument;

use super::{ReferenceValue, ReferenceValueLeaf, Value};
use crate::schema::document::{
    DeserializeError, Document, DocumentDeserialize, DocumentDeserializer,
};
use crate::schema::field_type::ValueParsingError;
use crate::schema::{Facet, Field, FieldType, NamedFieldDocument, OwnedValue, Schema};
use crate::tokenizer::PreTokenizedString;

#[repr(C, packed)]
#[derive(Debug, Clone)]
/// A field value pair in the compact tantivy document
struct FieldValueAddr {
    pub field: u16,
    pub value_addr: ValueAddr,
}

#[derive(Debug, Clone)]
/// The default document in tantivy. It encodes data in a compact form.
pub struct CompactDoc {
    /// `node_data` is a vec of bytes, where each value is serialized into bytes and stored. It
    /// includes all the data of the document and also metadata like where the nodes are located
    /// in an object or array.
    ///
    /// Invariant: string values are only written from `&str`, so their bytes are valid UTF-8.
    /// `extract_str` relies on this. The field must not be exposed outside of the crate.
    pub(crate) node_data: Vec<u8>,
    /// The root (Field, Value) pairs
    field_values: Vec<FieldValueAddr>,
}

impl Default for CompactDoc {
    fn default() -> Self {
        Self::new()
    }
}

impl CompactDoc {
    /// Creates a new, empty document object
    /// The reserved capacity is for the total serialized data
    pub fn with_capacity(bytes: usize) -> CompactDoc {
        Self::with_capacity_and_num_fields(bytes, 12)
    }

    /// Creates a new, empty document object with pre-reserved data and field-entry capacities.
    pub fn with_capacity_and_num_fields(bytes: usize, num_fields: usize) -> CompactDoc {
        CompactDoc {
            node_data: Vec::with_capacity(bytes),
            field_values: Vec::with_capacity(num_fields),
        }
    }

    /// Creates a new, empty document object
    pub fn new() -> CompactDoc {
        CompactDoc::with_capacity(1024)
    }

    /// Skrinks the capacity of the document to fit the data
    pub fn shrink_to_fit(&mut self) {
        self.node_data.shrink_to_fit();
        self.field_values.shrink_to_fit();
    }

    /// Returns the length of the document.
    pub fn len(&self) -> usize {
        self.field_values.len()
    }

    /// Adding a facet to the document.
    pub fn add_facet<F>(&mut self, field: Field, path: F)
    where Facet: From<F> {
        let facet = Facet::from(path);
        self.add_leaf_field_value(field, ReferenceValueLeaf::Facet(facet.encoded_str()));
    }

    /// Add a text field.
    pub fn add_text<S: AsRef<str>>(&mut self, field: Field, text: S) {
        self.add_leaf_field_value(field, ReferenceValueLeaf::Str(text.as_ref()));
    }

    /// Add a pre-tokenized text field.
    pub fn add_pre_tokenized_text(&mut self, field: Field, pre_tokenized_text: PreTokenizedString) {
        self.add_leaf_field_value(field, pre_tokenized_text);
    }

    /// Add a u64 field
    pub fn add_u64(&mut self, field: Field, value: u64) {
        self.add_leaf_field_value(field, value);
    }

    /// Add a IP address field. Internally only Ipv6Addr is used.
    pub fn add_ip_addr(&mut self, field: Field, value: Ipv6Addr) {
        self.add_leaf_field_value(field, value);
    }

    /// Add a i64 field
    pub fn add_i64(&mut self, field: Field, value: i64) {
        self.add_leaf_field_value(field, value);
    }

    /// Add a f64 field
    pub fn add_f64(&mut self, field: Field, value: f64) {
        self.add_leaf_field_value(field, value);
    }

    /// Add a bool field
    pub fn add_bool(&mut self, field: Field, value: bool) {
        self.add_leaf_field_value(field, value);
    }

    /// Add a date field with unspecified time zone offset
    pub fn add_date(&mut self, field: Field, value: DateTime) {
        self.add_leaf_field_value(field, value);
    }

    /// Add a bytes field
    pub fn add_bytes(&mut self, field: Field, value: &[u8]) {
        self.add_leaf_field_value(field, value);
    }

    /// Add a plugin-defined custom field value.
    ///
    /// The bytes are an opaque payload owned by the plugin that consumes this field's custom
    /// type; tantivy never inspects, indexes, or stores them.
    pub fn add_custom(&mut self, field: Field, value: &[u8]) {
        self.add_leaf_field_value(field, ReferenceValueLeaf::Custom(value));
    }

    /// Add a dynamic object field
    pub fn add_object(&mut self, field: Field, object: BTreeMap<String, OwnedValue>) {
        self.add_field_value(field, &OwnedValue::from(object));
    }

    /// Add a (field, value) to the document.
    ///
    /// `OwnedValue` implements Value, which should be easiest to use, but is not the most
    /// performant.
    pub fn add_field_value<'a, V: Value<'a>>(&mut self, field: Field, value: V) {
        let field_value = FieldValueAddr {
            field: field
                .field_id()
                .try_into()
                .expect("support only up to u16::MAX field ids"),
            value_addr: self.add_value(value),
        };
        self.field_values.push(field_value);
    }

    /// Add a (field, leaf value) to the document.
    /// Leaf values don't have nested values.
    pub fn add_leaf_field_value<'a, T: Into<ReferenceValueLeaf<'a>>>(
        &mut self,
        field: Field,
        typed_val: T,
    ) {
        let value = typed_val.into();
        let field_value = FieldValueAddr {
            field: field
                .field_id()
                .try_into()
                .expect("support only up to u16::MAX field ids"),
            value_addr: self.add_value_leaf(value),
        };
        self.field_values.push(field_value);
    }

    /// field_values accessor
    pub fn field_values(&self) -> impl Iterator<Item = (Field, CompactDocValue<'_>)> {
        self.field_values.iter().map(|field_val| {
            let field = Field::from_field_id(field_val.field as u32);
            let val = self.get_compact_doc_value(field_val.value_addr);
            (field, val)
        })
    }

    /// Returns all of the `ReferenceValue`s associated the given field
    pub fn get_all(&self, field: Field) -> impl Iterator<Item = CompactDocValue<'_>> + '_ {
        self.field_values
            .iter()
            .filter(move |field_value| Field::from_field_id(field_value.field as u32) == field)
            .map(|val| self.get_compact_doc_value(val.value_addr))
    }

    /// Returns the first `ReferenceValue` associated the given field
    pub fn get_first(&self, field: Field) -> Option<CompactDocValue<'_>> {
        self.get_all(field).next()
    }

    /// Create document from a named doc.
    pub fn convert_named_doc(
        schema: &Schema,
        named_doc: NamedFieldDocument,
    ) -> Result<Self, DocParsingError> {
        let mut document = Self::new();
        for (field_name, values) in named_doc.0 {
            if let Ok(field) = schema.get_field(&field_name) {
                for value in values {
                    document.add_field_value(field, &value);
                }
            }
        }
        Ok(document)
    }

    /// Build a document object from a json-object.
    pub fn parse_json(schema: &Schema, doc_json: &str) -> Result<Self, DocParsingError> {
        let json_obj: Map<String, serde_json::Value> =
            serde_json::from_str(doc_json).map_err(|_| DocParsingError::invalid_json(doc_json))?;
        Self::from_json_object(schema, json_obj)
    }

    /// Build a document object from a json-object.
    pub fn from_json_object(
        schema: &Schema,
        json_obj: Map<String, serde_json::Value>,
    ) -> Result<Self, DocParsingError> {
        let mut doc = Self::default();
        for (field_name, json_value) in json_obj {
            if let Ok(field) = schema.get_field(&field_name) {
                let field_entry = schema.get_field_entry(field);
                let field_type = field_entry.field_type();
                if matches!(field_type, FieldType::TieBreaker) {
                    continue;
                }
                match json_value {
                    serde_json::Value::Array(json_items) => {
                        for json_item in json_items {
                            let value = field_type
                                .value_from_json(json_item)
                                .map_err(|e| DocParsingError::ValueError(field_name.clone(), e))?;
                            doc.add_field_value(field, &value);
                        }
                    }
                    _ => {
                        let value = field_type
                            .value_from_json(json_value)
                            .map_err(|e| DocParsingError::ValueError(field_name.clone(), e))?;
                        doc.add_field_value(field, &value);
                    }
                }
            }
        }
        Ok(doc)
    }

    #[inline]
    fn add_value_leaf(&mut self, leaf: ReferenceValueLeaf) -> ValueAddr {
        let type_id = ValueType::from(&leaf);
        // Write into `node_data` and return u32 position as its address
        // Null and bool are inlined into the address
        let val_addr = match leaf {
            ReferenceValueLeaf::Null => 0,
            ReferenceValueLeaf::Str(bytes) => {
                write_bytes_into(&mut self.node_data, bytes.as_bytes())
            }
            ReferenceValueLeaf::Facet(bytes) => {
                write_bytes_into(&mut self.node_data, bytes.as_bytes())
            }
            ReferenceValueLeaf::Bytes(bytes) => write_bytes_into(&mut self.node_data, bytes),
            ReferenceValueLeaf::Custom(bytes) => write_bytes_into(&mut self.node_data, bytes),
            ReferenceValueLeaf::U64(num) => {
                let pos = self.node_data.len() as u32;
                self.node_data.extend_from_slice(&num.to_le_bytes());
                pos
            }
            ReferenceValueLeaf::I64(num) => {
                let pos = self.node_data.len() as u32;
                self.node_data.extend_from_slice(&num.to_le_bytes());
                pos
            }
            ReferenceValueLeaf::F64(num) => write_into(&mut self.node_data, num),
            ReferenceValueLeaf::Bool(b) => b as u32,
            ReferenceValueLeaf::Date(date) => {
                let pos = self.node_data.len() as u32;
                self.node_data
                    .extend_from_slice(&date.into_timestamp_nanos().to_le_bytes());
                pos
            }
            ReferenceValueLeaf::IpAddr(num) => {
                let pos = self.node_data.len() as u32;
                self.node_data
                    .extend_from_slice(&num.to_u128().to_le_bytes());
                pos
            }
            ReferenceValueLeaf::PreTokStr(pre_tok) => write_into(&mut self.node_data, *pre_tok),
        };
        ValueAddr { type_id, val_addr }
    }

    /// Returns a streaming value writer that encodes nested arrays and objects directly into
    /// `node_data` using a caller-owned reusable `scratch` buffer without materializing an
    /// intermediate DOM.
    #[inline]
    pub fn value_writer_with_scratch<'a, 's>(
        &'a mut self,
        scratch: &'s mut Vec<u8>,
    ) -> CompactDocValueWriter<'a, 's> {
        scratch.clear();
        CompactDocValueWriter { doc: self, scratch }
    }

    /// Adds a value and returns its address in `node_data`.
    fn add_value<'a, V: Value<'a>>(&mut self, value: V) -> ValueAddr {
        let value = value.as_value();
        match value {
            ReferenceValue::Leaf(leaf) => self.add_value_leaf(leaf),
            other => {
                let mut scratch: SmallVec<[u8; 512]> = SmallVec::new();
                self.add_ref_value_with_scratch(other, &mut scratch)
            }
        }
    }

    /// Recursively encodes a value into `node_data`, using `scratch` as a LIFO buffer for child
    /// address tables so nested arrays/objects do not allocate per node.
    fn add_ref_value_with_scratch<'a, V: Value<'a>>(
        &mut self,
        value: ReferenceValue<'a, V>,
        scratch: &mut SmallVec<[u8; 512]>,
    ) -> ValueAddr {
        let type_id = ValueType::from(&value);
        match value {
            ReferenceValue::Leaf(leaf) => self.add_value_leaf(leaf),
            ReferenceValue::Array(elements) => {
                let start = scratch.len();
                for elem in elements {
                    let value_addr = self.add_ref_value_with_scratch(elem.as_value(), scratch);
                    write_fixed_value_addr_smallvec(scratch, value_addr);
                }
                let val_addr = write_bytes_into(&mut self.node_data, &scratch[start..]);
                scratch.truncate(start);
                ValueAddr { type_id, val_addr }
            }
            ReferenceValue::Object(entries) => {
                let start = scratch.len();
                for (key, val) in entries {
                    let key_val_addr = write_bytes_into(&mut self.node_data, key.as_bytes());
                    let value_addr = self.add_ref_value_with_scratch(val.as_value(), scratch);
                    let entry = encode_object_entry(key_val_addr, value_addr);
                    scratch.extend_from_slice(&entry);
                }
                let val_addr = write_bytes_into(&mut self.node_data, &scratch[start..]);
                scratch.truncate(start);
                ValueAddr { type_id, val_addr }
            }
        }
    }

    /// Get CompactDocValue for address
    #[inline(always)]
    fn get_compact_doc_value(&self, value_addr: ValueAddr) -> CompactDocValue<'_> {
        CompactDocValue {
            container: self,
            value_addr,
        }
    }

    /// get &[u8] reference from node_data
    #[inline(always)]
    fn extract_bytes(&self, addr: Addr) -> &[u8] {
        binary_deserialize_bytes(self.get_slice(addr))
    }

    /// get &str reference from node_data
    #[inline(always)]
    fn extract_str(&self, addr: Addr) -> &str {
        let data = self.extract_bytes(addr);
        // Utf-8 checks would have a noticeable performance overhead here.
        // SAFETY: `addr` points to a string value, which `add_value_leaf` wrote from a `&str`.
        // `node_data` is not writable outside of the crate, so the bytes are still valid UTF-8.
        unsafe { std::str::from_utf8_unchecked(data) }
    }

    /// deserialized owned value from node_data
    #[inline]
    fn read_from<T: BinarySerializable>(&self, addr: Addr) -> io::Result<T> {
        let data_slice = &self.node_data[addr as usize..];
        let mut cursor = std::io::Cursor::new(data_slice);
        T::deserialize(&mut cursor)
    }

    /// get slice from address. The returned slice is open ended
    #[inline(always)]
    fn get_slice(&self, addr: Addr) -> &[u8] {
        &self.node_data[addr as usize..]
    }
}

/// BinarySerializable alternative to read references
#[inline(always)]
fn binary_deserialize_bytes(data: &[u8]) -> &[u8] {
    let first = data[0];
    if first >= 0x80 {
        let len = (first & 0x7f) as usize;
        &data[1..1 + len]
    } else {
        let (len, bytes_read) = read_u32_vint_no_advance(data);
        &data[bytes_read..bytes_read + len as usize]
    }
}

/// Fixed encoded byte width of a [`ValueAddr`] inside `CompactDoc`'s in-memory `node_data`.
const VALUE_ADDR_ENCODED_SIZE: usize = 5;
/// Fixed encoded byte width of an object `(key_addr: u32, value_addr: ValueAddr)` entry.
const OBJECT_ENTRY_ENCODED_SIZE: usize = 4 + VALUE_ADDR_ENCODED_SIZE;

#[inline(always)]
fn encode_value_addr(addr: ValueAddr) -> [u8; VALUE_ADDR_ENCODED_SIZE] {
    let bytes = addr.val_addr.to_le_bytes();
    [addr.type_id as u8, bytes[0], bytes[1], bytes[2], bytes[3]]
}

#[inline(always)]
fn encode_object_entry(key_addr: u32, addr: ValueAddr) -> [u8; OBJECT_ENTRY_ENCODED_SIZE] {
    let key_bytes = key_addr.to_le_bytes();
    let val_bytes = addr.val_addr.to_le_bytes();
    [
        key_bytes[0],
        key_bytes[1],
        key_bytes[2],
        key_bytes[3],
        addr.type_id as u8,
        val_bytes[0],
        val_bytes[1],
        val_bytes[2],
        val_bytes[3],
    ]
}

#[inline(always)]
fn write_fixed_value_addr_smallvec(buf: &mut SmallVec<[u8; 512]>, addr: ValueAddr) {
    buf.extend_from_slice(&encode_value_addr(addr));
}

#[inline(always)]
fn read_fixed_value_addr(bytes: &[u8]) -> Option<ValueAddr> {
    if bytes.len() < VALUE_ADDR_ENCODED_SIZE {
        return None;
    }
    let type_byte = bytes[0];
    if type_byte > 13 {
        return None;
    }
    // SAFETY: `type_byte` is checked to be in `0..=13`, matching `ValueType`'s repr(u8) variants.
    let type_id = unsafe { std::mem::transmute::<u8, ValueType>(type_byte) };
    let val_addr = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
    Some(ValueAddr { type_id, val_addr })
}

/// Write bytes and return the position of the written data.
///
/// BinarySerializable alternative to write references
#[inline]
fn write_bytes_into(vec: &mut Vec<u8>, data: &[u8]) -> u32 {
    let pos = vec.len() as u32;
    let len = data.len();
    if len < 128 {
        vec.reserve(1 + len);
        // Tantivy VInt sets the high bit (0x80) on the final byte.
        vec.push((len as u8) | 0x80);
    } else {
        let mut buf = [0u8; 8];
        let len_vint_bytes = serialize_vint_u32(len as u32, &mut buf);
        vec.reserve(len_vint_bytes.len() + len);
        vec.extend_from_slice(len_vint_bytes);
    }
    vec.extend_from_slice(data);
    pos
}

/// Serialize and return the position
fn write_into<T: BinarySerializable>(vec: &mut Vec<u8>, value: T) -> u32 {
    let pos = vec.len() as u32;
    value.serialize(vec).unwrap();
    pos
}

impl PartialEq for CompactDoc {
    fn eq(&self, other: &Self) -> bool {
        // super slow, but only here for tests
        let convert_to_comparable_map = |doc: &CompactDoc| {
            let mut field_value_set: HashMap<Field, HashSet<String>> = Default::default();
            for field_value in doc.field_values.iter() {
                let value: OwnedValue = doc.get_compact_doc_value(field_value.value_addr).into();
                let value = serde_json::to_string(&value).unwrap();
                field_value_set
                    .entry(Field::from_field_id(field_value.field as u32))
                    .or_default()
                    .insert(value);
            }
            field_value_set
        };
        let self_field_values: HashMap<Field, HashSet<String>> = convert_to_comparable_map(self);
        let other_field_values: HashMap<Field, HashSet<String>> = convert_to_comparable_map(other);
        self_field_values.eq(&other_field_values)
    }
}

impl Eq for CompactDoc {}

impl DocumentDeserialize for CompactDoc {
    fn deserialize<'de, D>(mut deserializer: D) -> Result<Self, DeserializeError>
    where D: DocumentDeserializer<'de> {
        let mut doc = CompactDoc::default();
        // TODO: Deserializing into OwnedValue is wasteful. The deserializer should be able to work
        // on slices and referenced data.
        while let Some((field, value)) = deserializer.next_field::<OwnedValue>()? {
            doc.add_field_value(field, &value);
        }
        Ok(doc)
    }
}

/// A value of Compact Doc needs a reference to the container to extract its payload
#[derive(Debug, Clone, Copy)]
pub struct CompactDocValue<'a> {
    container: &'a CompactDoc,
    value_addr: ValueAddr,
}
impl PartialEq for CompactDocValue<'_> {
    fn eq(&self, other: &Self) -> bool {
        let value1: OwnedValue = (*self).into();
        let value2: OwnedValue = (*other).into();
        value1 == value2
    }
}
impl From<CompactDocValue<'_>> for OwnedValue {
    fn from(value: CompactDocValue) -> Self {
        value.as_value().into()
    }
}
impl<'a> Value<'a> for CompactDocValue<'a> {
    type ArrayIter = CompactDocArrayIter<'a>;

    type ObjectIter = CompactDocObjectIter<'a>;

    #[inline]
    fn as_value(&self) -> ReferenceValue<'a, Self> {
        self.get_ref_value().unwrap()
    }
}
impl<'a> CompactDocValue<'a> {
    #[inline]
    fn get_ref_value(&self) -> io::Result<ReferenceValue<'a, CompactDocValue<'a>>> {
        let addr = self.value_addr.val_addr;
        match self.value_addr.type_id {
            ValueType::Null => Ok(ReferenceValueLeaf::Null.into()),
            ValueType::Str => {
                let str_ref = self.container.extract_str(addr);
                Ok(ReferenceValueLeaf::Str(str_ref).into())
            }
            ValueType::Facet => {
                let str_ref = self.container.extract_str(addr);
                Ok(ReferenceValueLeaf::Facet(str_ref).into())
            }
            ValueType::Bytes => {
                let data = self.container.extract_bytes(addr);
                Ok(ReferenceValueLeaf::Bytes(data).into())
            }
            ValueType::Custom => {
                let data = self.container.extract_bytes(addr);
                Ok(ReferenceValueLeaf::Custom(data).into())
            }
            ValueType::U64 => self
                .container
                .read_from::<u64>(addr)
                .map(ReferenceValueLeaf::U64)
                .map(Into::into),
            ValueType::I64 => self
                .container
                .read_from::<i64>(addr)
                .map(ReferenceValueLeaf::I64)
                .map(Into::into),
            ValueType::F64 => self
                .container
                .read_from::<f64>(addr)
                .map(ReferenceValueLeaf::F64)
                .map(Into::into),
            ValueType::Bool => Ok(ReferenceValueLeaf::Bool(addr != 0).into()),
            ValueType::Date => self
                .container
                .read_from::<i64>(addr)
                .map(|ts| ReferenceValueLeaf::Date(DateTime::from_timestamp_nanos(ts)))
                .map(Into::into),
            ValueType::IpAddr => self
                .container
                .read_from::<u128>(addr)
                .map(|num| ReferenceValueLeaf::IpAddr(Ipv6Addr::from_u128(num)))
                .map(Into::into),
            ValueType::PreTokStr => self
                .container
                .read_from::<PreTokenizedString>(addr)
                .map(Into::into)
                .map(ReferenceValueLeaf::PreTokStr)
                .map(Into::into),
            ValueType::Object => Ok(ReferenceValue::Object(CompactDocObjectIter::new(
                self.container,
                addr,
            )?)),
            ValueType::Array => Ok(ReferenceValue::Array(CompactDocArrayIter::new(
                self.container,
                addr,
            )?)),
        }
    }
}

/// Opaque handle to a value encoded inside a [`CompactDoc`]'s `node_data` buffer.
#[derive(Clone, Copy, Debug)]
pub struct CompactDocValueAddr(ValueAddr);

/// Streaming encoder that writes nested JSON/document values straight into a [`CompactDoc`]
/// using a caller-owned LIFO `Vec<u8>` scratch buffer, avoiding any intermediate DOM allocation
/// or per-document stack/heap scratch setup.
pub struct CompactDocValueWriter<'a, 's> {
    doc: &'a mut CompactDoc,
    scratch: &'s mut Vec<u8>,
}

impl CompactDocValueWriter<'_, '_> {
    /// Returns a mutable reference to the underlying [`CompactDoc`] for adding top-level
    /// leaf fields while a dynamic object is being accumulated in `scratch`.
    #[inline(always)]
    pub fn doc_mut(&mut self) -> &mut CompactDoc {
        self.doc
    }

    /// Encodes a leaf value into `node_data` and returns its address.
    #[inline(always)]
    pub fn write_leaf(&mut self, leaf: ReferenceValueLeaf<'_>) -> CompactDocValueAddr {
        CompactDocValueAddr(self.doc.add_value_leaf(leaf))
    }

    /// Encodes an object key string into `node_data` and returns its byte offset.
    #[inline(always)]
    pub fn write_key(&mut self, key: &str) -> u32 {
        write_bytes_into(&mut self.doc.node_data, key.as_bytes())
    }

    /// Starts encoding an object into the LIFO scratch buffer and returns its frame marker.
    #[inline(always)]
    pub fn begin_object(&self) -> usize {
        self.scratch.len()
    }

    /// Appends a `(key_addr, val_addr)` entry to the currently open object frame.
    #[inline(always)]
    pub fn push_object_entry(&mut self, key_addr: u32, val_addr: CompactDocValueAddr) {
        let entry = encode_object_entry(key_addr, val_addr.0);
        self.scratch.extend_from_slice(&entry);
    }

    /// Finalizes the object frame started at `start` and returns its encoded address.
    #[inline(always)]
    pub fn end_object(&mut self, start: usize) -> CompactDocValueAddr {
        let val_addr = write_bytes_into(&mut self.doc.node_data, &self.scratch[start..]);
        self.scratch.truncate(start);
        CompactDocValueAddr(ValueAddr {
            type_id: ValueType::Object,
            val_addr,
        })
    }

    /// Returns whether any entries have been written into the object/array frame starting at
    /// `start`.
    #[inline(always)]
    pub fn frame_is_empty(&self, start: usize) -> bool {
        self.scratch.len() == start
    }

    /// Starts encoding an array into the LIFO scratch buffer and returns its frame marker.
    #[inline(always)]
    pub fn begin_array(&self) -> usize {
        self.scratch.len()
    }

    /// Appends an element address to the currently open array frame.
    #[inline(always)]
    pub fn push_array_elem(&mut self, val_addr: CompactDocValueAddr) {
        let entry = encode_value_addr(val_addr.0);
        self.scratch.extend_from_slice(&entry);
    }

    /// Finalizes the array frame started at `start` and returns its encoded address.
    #[inline(always)]
    pub fn end_array(&mut self, start: usize) -> CompactDocValueAddr {
        let val_addr = write_bytes_into(&mut self.doc.node_data, &self.scratch[start..]);
        self.scratch.truncate(start);
        CompactDocValueAddr(ValueAddr {
            type_id: ValueType::Array,
            val_addr,
        })
    }

    /// Attaches an encoded value address as a top-level field value on the document.
    #[inline(always)]
    pub fn finish_field(&mut self, field: Field, addr: CompactDocValueAddr) {
        self.doc.field_values.push(FieldValueAddr {
            field: field
                .field_id()
                .try_into()
                .expect("support only up to u16::MAX field ids"),
            value_addr: addr.0,
        });
    }
}

/// The address in the vec
type Addr = u32;

#[derive(Clone, Copy, Default)]
#[repr(C, packed)]
/// The value type and the address to its payload in the container.
struct ValueAddr {
    type_id: ValueType,
    /// This is the address to the value in the vec, except for bool and null, which are inlined
    val_addr: Addr,
}
impl BinarySerializable for ValueAddr {
    fn serialize<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        self.type_id.serialize(writer)?;
        VInt(self.val_addr as u64).serialize(writer)
    }

    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let type_id = ValueType::deserialize(reader)?;
        let val_addr = VInt::deserialize(reader)?.0 as u32;
        Ok(ValueAddr { type_id, val_addr })
    }
}
impl std::fmt::Debug for ValueAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let val_addr = self.val_addr;
        f.write_fmt(format_args!("{:?} at {:?}", self.type_id, val_addr))
    }
}

/// A enum representing a value for tantivy to index.
///
/// ** Any changes need to be reflected in `BinarySerializable` for `ValueType` **
///
/// We can't use [schema::Type] or [columnar::ColumnType] here, because they are missing
/// some items like Array and PreTokStr.
#[derive(Default, Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum ValueType {
    /// A null value.
    #[default]
    Null = 0,
    /// The str type is used for any text information.
    Str = 1,
    /// Unsigned 64-bits Integer `u64`
    U64 = 2,
    /// Signed 64-bits Integer `i64`
    I64 = 3,
    /// 64-bits Float `f64`
    F64 = 4,
    /// Date/time with nanoseconds precision
    Date = 5,
    /// Facet
    Facet = 6,
    /// Arbitrarily sized byte array
    Bytes = 7,
    /// IpV6 Address. Internally there is no IpV4, it needs to be converted to `Ipv6Addr`.
    IpAddr = 8,
    /// Bool value
    Bool = 9,
    /// Pre-tokenized str type,
    PreTokStr = 10,
    /// Object
    Object = 11,
    /// Pre-tokenized str type,
    Array = 12,
    /// Opaque payload of a plugin-defined custom field.
    Custom = 13,
}

impl BinarySerializable for ValueType {
    fn serialize<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        (*self as u8).serialize(writer)?;
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let num = u8::deserialize(reader)?;
        let type_id = if (0..=13).contains(&num) {
            unsafe { std::mem::transmute::<u8, ValueType>(num) }
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid value type id: {num}"),
            ));
        };
        Ok(type_id)
    }
}

impl<'a, V: Value<'a>> From<&ReferenceValue<'a, V>> for ValueType {
    fn from(value: &ReferenceValue<'a, V>) -> Self {
        match value {
            ReferenceValue::Leaf(leaf) => leaf.into(),
            ReferenceValue::Array(_) => ValueType::Array,
            ReferenceValue::Object(_) => ValueType::Object,
        }
    }
}
impl<'a> From<&ReferenceValueLeaf<'a>> for ValueType {
    fn from(value: &ReferenceValueLeaf<'a>) -> Self {
        match value {
            ReferenceValueLeaf::Null => ValueType::Null,
            ReferenceValueLeaf::Str(_) => ValueType::Str,
            ReferenceValueLeaf::U64(_) => ValueType::U64,
            ReferenceValueLeaf::I64(_) => ValueType::I64,
            ReferenceValueLeaf::F64(_) => ValueType::F64,
            ReferenceValueLeaf::Bool(_) => ValueType::Bool,
            ReferenceValueLeaf::Date(_) => ValueType::Date,
            ReferenceValueLeaf::IpAddr(_) => ValueType::IpAddr,
            ReferenceValueLeaf::PreTokStr(_) => ValueType::PreTokStr,
            ReferenceValueLeaf::Facet(_) => ValueType::Facet,
            ReferenceValueLeaf::Bytes(_) => ValueType::Bytes,
            ReferenceValueLeaf::Custom(_) => ValueType::Custom,
        }
    }
}

#[derive(Debug, Clone)]
/// The Iterator for the object values in the compact document
pub struct CompactDocObjectIter<'a> {
    container: &'a CompactDoc,
    node_addresses_slice: &'a [u8],
}

impl<'a> CompactDocObjectIter<'a> {
    fn new(container: &'a CompactDoc, addr: Addr) -> io::Result<Self> {
        // Objects are `&[ValueAddr]` serialized into bytes
        let node_addresses_slice = container.extract_bytes(addr);
        Ok(Self {
            container,
            node_addresses_slice,
        })
    }
}

impl<'a> Iterator for CompactDocObjectIter<'a> {
    type Item = (&'a str, CompactDocValue<'a>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.node_addresses_slice.len() < OBJECT_ENTRY_ENCODED_SIZE {
            return None;
        }
        let (entry, rest) = self
            .node_addresses_slice
            .split_at(OBJECT_ENTRY_ENCODED_SIZE);
        self.node_addresses_slice = rest;
        let key_addr = u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]);
        let key = self.container.extract_str(key_addr);
        let value_addr = read_fixed_value_addr(&entry[4..])?;
        let value = CompactDocValue {
            container: self.container,
            value_addr,
        };
        Some((key, value))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.node_addresses_slice.len() / OBJECT_ENTRY_ENCODED_SIZE;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for CompactDocObjectIter<'_> {}

#[derive(Debug, Clone)]
/// The Iterator for the array values in the compact document
pub struct CompactDocArrayIter<'a> {
    container: &'a CompactDoc,
    node_addresses_slice: &'a [u8],
}

impl<'a> CompactDocArrayIter<'a> {
    fn new(container: &'a CompactDoc, addr: Addr) -> io::Result<Self> {
        // Arrays are fixed-width `ValueAddr` entries serialized into bytes
        let node_addresses_slice = container.extract_bytes(addr);
        Ok(Self {
            container,
            node_addresses_slice,
        })
    }
}

impl<'a> Iterator for CompactDocArrayIter<'a> {
    type Item = CompactDocValue<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.node_addresses_slice.len() < VALUE_ADDR_ENCODED_SIZE {
            return None;
        }
        let (entry, rest) = self.node_addresses_slice.split_at(VALUE_ADDR_ENCODED_SIZE);
        self.node_addresses_slice = rest;
        let value_addr = read_fixed_value_addr(entry)?;
        let value = CompactDocValue {
            container: self.container,
            value_addr,
        };
        Some(value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.node_addresses_slice.len() / VALUE_ADDR_ENCODED_SIZE;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for CompactDocArrayIter<'_> {}

impl Document for CompactDoc {
    type Value<'a> = CompactDocValue<'a>;
    type FieldsValuesIter<'a> = FieldValueIterRef<'a>;

    fn iter_fields_and_values(&self) -> Self::FieldsValuesIter<'_> {
        FieldValueIterRef {
            slice: self.field_values.iter(),
            container: self,
        }
    }

    fn into_tantivy_document(self) -> TantivyDocument {
        self
    }
}

/// A helper wrapper for creating an iterator over the field values
pub struct FieldValueIterRef<'a> {
    slice: std::slice::Iter<'a, FieldValueAddr>,
    container: &'a CompactDoc,
}

impl<'a> Iterator for FieldValueIterRef<'a> {
    type Item = (Field, CompactDocValue<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        self.slice.next().map(|field_value| {
            (
                Field::from_field_id(field_value.field as u32),
                CompactDocValue::<'a> {
                    container: self.container,
                    value_addr: field_value.value_addr,
                },
            )
        })
    }
}

/// Error that may happen when deserializing
/// a document from JSON.
#[derive(Debug, Error, PartialEq)]
pub enum DocParsingError {
    /// The payload given is not valid JSON.
    #[error("The provided string is not valid JSON")]
    InvalidJson(String),
    /// One of the value node could not be parsed.
    #[error("The field '{0:?}' could not be parsed: {1:?}")]
    ValueError(String, ValueParsingError),
}

impl DocParsingError {
    /// Builds a NotJson DocParsingError
    fn invalid_json(invalid_json: &str) -> Self {
        let sample = invalid_json.chars().take(20).collect();
        DocParsingError::InvalidJson(sample)
    }
}

#[cfg(test)]
mod tests {
    use crate::schema::*;

    #[test]
    fn test_doc() {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("title", TEXT);
        let mut doc = TantivyDocument::default();
        doc.add_text(text_field, "My title");
        assert_eq!(doc.field_values().count(), 1);

        let schema = schema_builder.build();
        let _val = doc.get_first(text_field).unwrap();
        let _json = doc.to_named_doc(&schema);
    }

    #[test]
    fn test_parse_json_ignores_tie_breaker_values() {
        let mut schema_builder = Schema::builder();
        let title = schema_builder.add_text_field("title", TEXT);
        let tie = schema_builder.add_tie_breaker_field("tie");
        let schema = schema_builder.build();
        let doc =
            TantivyDocument::parse_json(&schema, r#"{"title": "hello", "tie": [1, "two", null]}"#)
                .unwrap();
        assert!(doc.get_first(title).is_some());
        assert!(doc.get_first(tie).is_none());
    }

    #[test]
    fn test_json_value() {
        let json_str = r#"{
            "toto": "titi",
            "float": -0.2,
            "bool": true,
            "unsigned": 1,
            "signed": -2,
            "complexobject": {
                "field.with.dot": 1
            },
            "date": "1985-04-12T23:20:50.52Z",
            "my_arr": [2, 3, {"my_key": "two tokens"}, 4, {"nested_array": [2, 5, 6, [7, 8, {"a": [{"d": {"e":[99]}}, 9000]}, 9, 10], [5, 5]]}]
        }"#;
        let json_val: std::collections::BTreeMap<String, OwnedValue> =
            serde_json::from_str(json_str).unwrap();

        let mut schema_builder = Schema::builder();
        let json_field = schema_builder.add_json_field("json", TEXT);
        let mut doc = TantivyDocument::default();
        doc.add_object(json_field, json_val);

        let schema = schema_builder.build();
        let json = doc.to_json(&schema);
        let actual_json: serde_json::Value = serde_json::from_str(&json).unwrap();
        let expected_json: serde_json::Value = serde_json::from_str(json_str).unwrap();
        assert_eq!(actual_json["json"][0], expected_json);
    }

    // TODO: Should this be re-added with the serialize method
    //       technically this is no longer useful since the doc types
    //       do not implement BinarySerializable due to orphan rules.
    // #[test]
    // fn test_doc_serialization_issue() {
    //     let mut doc = Document::default();
    //     doc.add_json_object(
    //         Field::from_field_id(0),
    //         serde_json::json!({"key": 2u64})
    //             .as_object()
    //             .unwrap()
    //             .clone(),
    //     );
    //     doc.add_text(Field::from_field_id(1), "hello");
    //     assert_eq!(doc.field_values().len(), 2);
    //     let mut payload: Vec<u8> = Vec::new();
    //     doc_binary_wrappers::serialize(&doc, &mut payload).unwrap();
    //     assert_eq!(payload.len(), 26);
    //     doc_binary_wrappers::deserialize::<Document, _>(&mut &payload[..]).unwrap();
    // }
}
