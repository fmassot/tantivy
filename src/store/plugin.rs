//! Store as a [`SegmentPlugin`] implementation.
//!
//! This wraps the existing `StoreWriter` and `StoreReader` types behind the plugin
//! interface so that the document store participates in the unified plugin lifecycle.

use std::any::Any;
use std::collections::BTreeMap;
use std::io::Write;

use common::TerminatingWrite;
use measure_time::debug_time;

use crate::directory::{Directory, WritePtr};
use crate::index::{SegmentComponent, SegmentReader};
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::plugin::{PluginMergeContext, PluginWriter, PluginWriterContext, SegmentPlugin};
use crate::schema::document::{serialize_doc_to_vec, Document};
use crate::schema::Schema;
use crate::space_usage::{ComponentSpaceUsage, STORE};
use crate::store::StoreWriter;
use crate::Segment;

/// Built-in segment plugin that stores and merges stored documents.
pub struct StorePlugin;

impl SegmentPlugin for StorePlugin {
    fn extensions(&self) -> &[&str] {
        &["store"]
    }

    fn create_writer(&self, _ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        unimplemented!("StorePlugin is a built-in; use StorePluginWriter::new")
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        debug_time!("write-storable-fields");
        debug!("write-storable-fields");

        let path = ctx.target_segment.relative_path(SegmentComponent::Store);
        let store_write = ctx.target_segment.index().directory().open_write(&path)?;
        let settings = ctx.settings;
        let mut store_writer = StoreWriter::new(
            store_write,
            settings.docstore_compression,
            settings.docstore_blocksize,
            settings.docstore_compress_dedicated_thread,
        )?;

        if !ctx.doc_id_mapping.is_trivial() {
            debug!("non-trivial-doc-id-mapping");

            let store_readers: Vec<_> = ctx
                .readers
                .iter()
                .map(|reader| reader.get_store_reader(50))
                .collect::<Result<_, _>>()?;

            let mut document_iterators: Vec<_> = store_readers
                .iter()
                .enumerate()
                .map(|(i, store)| store.iter_raw(ctx.readers[i].alive_bitset()))
                .collect();

            for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
                let doc_bytes_it = &mut document_iterators[old_doc_addr.segment_ord as usize];
                if let Some(doc_bytes_res) = doc_bytes_it.next() {
                    let doc_bytes = doc_bytes_res?;
                    store_writer.store_bytes(&doc_bytes)?;
                } else {
                    return Err(crate::error::DataCorruption::comment_only(format!(
                        "unexpected missing document in docstore on merge, doc address \
                         {old_doc_addr:?}",
                    ))
                    .into());
                }
            }
        } else {
            debug!("trivial-doc-id-mapping");
            for reader in ctx.readers {
                let store_reader = reader.get_store_reader(1)?;
                if reader.has_deletes()
                    // If there is not enough data in the store, we avoid stacking in order to
                    // avoid creating many small blocks in the doc store. Once we have 5 full blocks,
                    // we start stacking. In the worst case 2/7 of the blocks would be very small.
                    || store_reader.block_checkpoints().take(7).count() < 6
                    || store_reader.decompressor() != store_writer.compressor().into()
                {
                    for doc_bytes_res in store_reader.iter_raw(reader.alive_bitset()) {
                        let doc_bytes = doc_bytes_res?;
                        store_writer.store_bytes(&doc_bytes)?;
                    }
                } else {
                    store_writer.stack(store_reader)?;
                }
            }
        }
        store_writer.close()?;
        Ok(())
    }

    fn space_usage(
        &self,
        segment_reader: &SegmentReader,
    ) -> crate::Result<BTreeMap<String, ComponentSpaceUsage>> {
        let store = segment_reader.get_store_reader(0)?;
        Ok(BTreeMap::from([(
            STORE.to_string(),
            ComponentSpaceUsage::Store(store.space_usage()),
        )]))
    }
}

/// Flat sequential writer for the ephemeral `SegmentComponent::TempStore` file used when
/// documents must be reordered by a [`DocIdMapping`] before writing the final compressed store.
///
/// Storing raw serialized document bytes back-to-back with an in-memory `doc_offsets` index
/// avoids block framing, skip-index construction/lookup, and LRU block-cache locking during
/// random-order resorting.
struct TempStoreWriter {
    writer: WritePtr,
    doc_buf: Vec<u8>,
    /// Byte offsets into `TempStore`: document `i` occupies `doc_offsets[i]..doc_offsets[i + 1]`.
    doc_offsets: Vec<u64>,
    bytes_written: u64,
}

impl TempStoreWriter {
    fn new(writer: WritePtr) -> Self {
        let mut doc_offsets = Vec::with_capacity(1024);
        doc_offsets.push(0);
        Self {
            writer,
            doc_buf: Vec::with_capacity(2048),
            doc_offsets,
            bytes_written: 0,
        }
    }

    fn store<D: Document>(&mut self, document: &D, schema: &Schema) -> std::io::Result<()> {
        self.doc_buf.clear();
        serialize_doc_to_vec(document, schema, &mut self.doc_buf)?;
        self.writer.write_all(&self.doc_buf)?;
        self.bytes_written += self.doc_buf.len() as u64;
        self.doc_offsets.push(self.bytes_written);
        Ok(())
    }

    fn store_bytes(&mut self, serialized_document: &[u8]) -> std::io::Result<()> {
        self.writer.write_all(serialized_document)?;
        self.bytes_written += serialized_document.len() as u64;
        self.doc_offsets.push(self.bytes_written);
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        self.doc_buf.capacity() + self.doc_offsets.capacity() * std::mem::size_of::<u64>()
    }
}

enum ActiveStoreWriter {
    Direct(StoreWriter),
    Temp(TempStoreWriter),
}

/// Accumulates and serializes stored documents for a segment.
pub struct StorePluginWriter {
    store_writer: Option<ActiveStoreWriter>,
}

impl StorePluginWriter {
    pub(crate) fn new(ctx: &PluginWriterContext) -> crate::Result<Self> {
        let settings = ctx.segment.index().settings();
        let directory = ctx.segment.index().directory();
        let remapping_required = settings.sort_by_field.is_some() || settings.manual_doc_id_mapping;

        let active_writer = if remapping_required {
            let path = ctx.segment.relative_path(SegmentComponent::TempStore);
            let store_write = directory.open_write(&path)?;
            ActiveStoreWriter::Temp(TempStoreWriter::new(store_write))
        } else {
            let path = ctx.segment.relative_path(SegmentComponent::Store);
            let store_write = directory.open_write(&path)?;
            ActiveStoreWriter::Direct(StoreWriter::new(
                store_write,
                settings.docstore_compression,
                settings.docstore_blocksize,
                settings.docstore_compress_dedicated_thread,
            )?)
        };

        Ok(StorePluginWriter {
            store_writer: Some(active_writer),
        })
    }

    /// Serializes and stores a document according to the supplied schema.
    pub fn store<D: Document>(&mut self, document: &D, schema: &Schema) -> crate::Result<()> {
        match self.store_writer.as_mut() {
            Some(ActiveStoreWriter::Direct(writer)) => writer
                .store(document, schema)
                .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?,
            Some(ActiveStoreWriter::Temp(writer)) => writer
                .store(document, schema)
                .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?,
            None => {}
        }
        Ok(())
    }

    /// Stores an already serialized document.
    pub fn store_bytes(&mut self, serialized_document: &[u8]) -> crate::Result<()> {
        match self.store_writer.as_mut() {
            Some(ActiveStoreWriter::Direct(writer)) => writer
                .store_bytes(serialized_document)
                .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?,
            Some(ActiveStoreWriter::Temp(writer)) => writer
                .store_bytes(serialized_document)
                .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?,
            None => {}
        }
        Ok(())
    }
}

impl PluginWriter for StorePluginWriter {
    fn serialize(
        mut self: Box<Self>,
        segment: &Segment,
        doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        match self.store_writer.take() {
            Some(ActiveStoreWriter::Temp(temp_writer)) => {
                temp_writer
                    .writer
                    .terminate()
                    .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;

                if let Some(doc_id_map) = doc_id_map {
                    debug!("resort-docstore");
                    let settings = segment.index().settings().clone();
                    let store_write = segment.open_write(SegmentComponent::Store)?;
                    let mut store_writer = StoreWriter::new(
                        store_write,
                        settings.docstore_compression,
                        settings.docstore_blocksize,
                        settings.docstore_compress_dedicated_thread,
                    )
                    .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;

                    let temp_file = segment.open_read(SegmentComponent::TempStore)?;
                    let temp_bytes = temp_file
                        .read_bytes()
                        .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
                    let raw_slice = temp_bytes.as_slice();
                    let doc_offsets = &temp_writer.doc_offsets;

                    for old_doc_id in doc_id_map.iter_old_doc_ids() {
                        let doc_idx = old_doc_id as usize;
                        let start = doc_offsets[doc_idx] as usize;
                        let end = doc_offsets[doc_idx + 1] as usize;
                        store_writer
                            .store_bytes(&raw_slice[start..end])
                            .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
                    }
                    store_writer
                        .close()
                        .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
                }
            }
            Some(ActiveStoreWriter::Direct(writer)) => {
                writer
                    .close()
                    .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
            }
            None => {}
        }
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        match self.store_writer.as_ref() {
            Some(ActiveStoreWriter::Direct(writer)) => writer.mem_usage(),
            Some(ActiveStoreWriter::Temp(writer)) => writer.mem_usage(),
            None => 0,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
