#![allow(clippy::needless_borrows_for_generic_args)]

use std::cmp::Ordering;
use std::io;
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

use common::bounds::{TransformBound, transform_bound_inner_res};
use common::file_slice::FileSlice;
use common::{BinarySerializable, ByteCount, HasLen, OwnedBytes};
use futures_util::{StreamExt, TryStreamExt, stream};
use itertools::Itertools;
use tantivy_fst::Automaton;
use tantivy_fst::automaton::AlwaysMatch;

use crate::delta::DeltaKeyComparator;
use crate::streamer::{Streamer, StreamerBuilder};
use crate::{BlockAddr, DeltaReader, Reader, SSTable, SSTableIndex, TermOrdinal, VoidSSTable};

/// An SSTable is a sorted map that associates sorted `&[u8]` keys
/// to any kind of typed values.
///
/// The SSTable is organized in blocks.
/// In each block, keys and values are encoded separately.
///
/// The keys are encoded using incremental encoding.
/// The values on the other hand, are encoded according to a value-specific
/// codec defined in the TSSTable generic argument.
///
/// Finally, an index is joined to the Dictionary to make it possible,
/// given a key to identify which block contains this key.
///
/// The codec was designed in such a way that the sstable
/// reader is not aware of block, and yet can read any sequence of blocks,
/// as long as the slice of bytes it is given starts and stops at
/// block boundary.
///
/// (See also README.md)
#[derive(Debug, Clone)]
pub struct Dictionary<TSSTable: SSTable = VoidSSTable> {
    pub sstable_slice: FileSlice,
    pub sstable_index: SSTableIndex,
    num_bytes: ByteCount,
    num_terms: u64,
    phantom_data: PhantomData<TSSTable>,
}

impl Dictionary<VoidSSTable> {
    pub fn build_for_tests(terms: &[&str]) -> Dictionary {
        let mut terms = terms.to_vec();
        terms.sort();
        let mut buffer = Vec::new();
        let mut dictionary_writer = Self::builder(&mut buffer).unwrap();
        for term in terms {
            dictionary_writer.insert(term, &()).unwrap();
        }
        dictionary_writer.finish().unwrap();
        Dictionary::from_bytes(OwnedBytes::new(buffer)).unwrap()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TermOrdHit {
    /// Exact term ord hit
    Exact(TermOrdinal),
    /// Next best term ordinal
    Next(TermOrdinal),
}

impl TermOrdHit {
    fn into_exact(self) -> Option<TermOrdinal> {
        match self {
            TermOrdHit::Exact(ord) => Some(ord),
            TermOrdHit::Next(_) => None,
        }
    }

    fn map<F: FnOnce(TermOrdinal) -> TermOrdinal>(self, f: F) -> Self {
        match self {
            TermOrdHit::Exact(ord) => TermOrdHit::Exact(f(ord)),
            TermOrdHit::Next(ord) => TermOrdHit::Next(f(ord)),
        }
    }
}

impl<TSSTable: SSTable> Dictionary<TSSTable> {
    pub fn builder<W: io::Write>(wrt: W) -> io::Result<crate::Writer<W, TSSTable::ValueWriter>> {
        Ok(TSSTable::writer(wrt))
    }

    pub(crate) fn sstable_reader_block(
        &self,
        block_addr: BlockAddr,
    ) -> io::Result<Reader<TSSTable::ValueReader>> {
        let data = self.sstable_slice.read_bytes_slice(block_addr.byte_range)?;
        Ok(TSSTable::reader(data))
    }

    pub(crate) async fn sstable_delta_reader_for_key_range_async(
        &self,
        key_range: impl RangeBounds<[u8]>,
        limit: Option<u64>,
        automaton: &impl Automaton,
        merge_holes_under_bytes: usize,
    ) -> io::Result<DeltaReader<TSSTable::ValueReader>> {
        let match_all = automaton.will_always_match(&automaton.start());
        if match_all {
            let slice = self.file_slice_for_range(key_range, limit);
            let data = slice.read_bytes_async().await?;
            Ok(TSSTable::delta_reader(data))
        } else {
            let blocks = stream::iter(self.get_block_iterator_for_range_and_automaton(
                key_range,
                automaton,
                merge_holes_under_bytes,
            ));
            let data = blocks
                .map(|block_addr| {
                    let first_ordinal = block_addr.first_ordinal;
                    async move {
                        let bytes = self
                            .sstable_slice
                            .read_bytes_slice_async(block_addr.byte_range)
                            .await?;
                        io::Result::Ok((bytes, first_ordinal))
                    }
                })
                .buffered(5)
                .try_collect::<Vec<_>>()
                .await?;
            Ok(DeltaReader::from_multiple_blocks(data))
        }
    }

    pub(crate) fn sstable_delta_reader_for_key_range(
        &self,
        key_range: impl RangeBounds<[u8]>,
        limit: Option<u64>,
        automaton: &impl Automaton,
    ) -> io::Result<DeltaReader<TSSTable::ValueReader>> {
        let match_all = automaton.will_always_match(&automaton.start());
        if match_all {
            let slice = self.file_slice_for_range(key_range, limit);
            let data = slice.read_bytes()?;
            Ok(TSSTable::delta_reader(data))
        } else {
            // if operations are sync, we assume latency is almost null, and there is no point in
            // merging across holes
            let blocks = self.get_block_iterator_for_range_and_automaton(key_range, automaton, 0);
            let data = blocks
                .map(|block_addr| {
                    let first_ordinal = block_addr.first_ordinal;
                    self.sstable_slice
                        .read_bytes_slice(block_addr.byte_range)
                        .map(|bytes| (bytes, first_ordinal))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(DeltaReader::from_multiple_blocks(data))
        }
    }

    /// Partitions the blocks matching `automaton` into up to `max_chunks` contiguous chunks when
    /// the total matched byte volume is at least `2 * min_bytes_per_chunk`, returning per-chunk
    /// `(OwnedBytes, first_ordinal)` slices ready for parallel scanning.
    ///
    /// Returns `Ok(None)` when the total volume is small enough to scan in a single pass.
    pub fn automaton_block_chunks(
        &self,
        automaton: &impl Automaton,
        max_chunks: usize,
        min_bytes_per_chunk: usize,
    ) -> io::Result<Option<Vec<Vec<(OwnedBytes, u64)>>>> {
        if max_chunks <= 1 || automaton.will_always_match(&automaton.start()) {
            return Ok(None);
        }
        let blocks: Vec<BlockAddr> = self
            .sstable_index
            .get_block_for_automaton(automaton)
            .map(|(_, block_addr)| block_addr)
            .collect();
        let total_bytes: usize = blocks.iter().map(|b| b.byte_range.len()).sum();
        if total_bytes < min_bytes_per_chunk.saturating_mul(2) || blocks.len() < 2 {
            return Ok(None);
        }
        let num_chunks = (total_bytes / min_bytes_per_chunk).clamp(2, max_chunks);
        let target_bytes = total_bytes.div_ceil(num_chunks);

        let mut chunks: Vec<Vec<(OwnedBytes, u64)>> = Vec::with_capacity(num_chunks);
        let mut cur_slices: Vec<(OwnedBytes, u64)> = Vec::new();
        let mut cur_bytes: usize = 0;
        let mut merged_range: Option<(std::ops::Range<usize>, u64)> = None;

        let flush_merged = |merged: &mut Option<(std::ops::Range<usize>, u64)>,
                            out: &mut Vec<(OwnedBytes, u64)>|
         -> io::Result<()> {
            if let Some((range, first_ord)) = merged.take() {
                let bytes = self.sstable_slice.read_bytes_slice(range)?;
                out.push((bytes, first_ord));
            }
            Ok(())
        };

        for block in blocks {
            let block_len = block.byte_range.len();
            if cur_bytes >= target_bytes && chunks.len() + 1 < num_chunks {
                flush_merged(&mut merged_range, &mut cur_slices)?;
                if !cur_slices.is_empty() {
                    chunks.push(std::mem::take(&mut cur_slices));
                }
                cur_bytes = 0;
            }
            match &mut merged_range {
                Some((range, _)) if range.end == block.byte_range.start => {
                    range.end = block.byte_range.end;
                }
                _ => {
                    flush_merged(&mut merged_range, &mut cur_slices)?;
                    merged_range = Some((block.byte_range, block.first_ordinal));
                }
            }
            cur_bytes += block_len;
        }
        flush_merged(&mut merged_range, &mut cur_slices)?;
        if !cur_slices.is_empty() {
            chunks.push(cur_slices);
        }
        if chunks.len() <= 1 {
            return Ok(None);
        }
        Ok(Some(chunks))
    }

    /// Streams all values matching `automaton` from pre-fetched SSTable block slices without
    /// reconstructing keys.
    pub fn stream_values_from_slices<A>(
        &self,
        slices: Vec<(OwnedBytes, u64)>,
        automaton: A,
    ) -> io::Result<Vec<TSSTable::Value>>
    where
        A: Automaton,
        A::State: Clone,
        TSSTable::Value: Clone,
    {
        let delta_reader = DeltaReader::<TSSTable::ValueReader>::from_multiple_blocks(slices);
        let mut streamer = self
            .search(automaton)
            .without_keys()
            .into_stream_given_delta_reader(delta_reader)?;
        let mut values = Vec::new();
        while let Some(value) = streamer.next_without_key() {
            values.push(value.clone());
        }
        Ok(values)
    }

    /// Partitions all blocks in the SSTable into up to `max_chunks` contiguous byte-balanced
    /// chunks without walking the FST index.
    ///
    /// Returns `Ok(None)` when `max_chunks <= 1` or the total SSTable byte volume is smaller than
    /// `2 * min_bytes_per_chunk`.
    pub fn all_block_chunks(
        &self,
        max_chunks: usize,
        min_bytes_per_chunk: usize,
    ) -> io::Result<Option<Vec<(OwnedBytes, u64)>>> {
        let total_bytes = self.sstable_slice.len();
        if max_chunks <= 1 || total_bytes < min_bytes_per_chunk.saturating_mul(2) {
            return Ok(None);
        }
        let num_chunks = (total_bytes / min_bytes_per_chunk).clamp(2, max_chunks);
        let target_bytes = total_bytes.div_ceil(num_chunks);

        let mut chunks: Vec<(OwnedBytes, u64)> = Vec::with_capacity(num_chunks);
        let mut cur_range: Option<(std::ops::Range<usize>, u64)> = None;
        let mut block_id = 0u64;
        while let Some(block) = self.sstable_index.get_block(block_id) {
            block_id += 1;
            match &mut cur_range {
                Some((range, first_ord))
                    if range.len() >= target_bytes && chunks.len() + 1 < num_chunks =>
                {
                    let bytes = self.sstable_slice.read_bytes_slice(range.clone())?;
                    chunks.push((bytes, *first_ord));
                    cur_range = Some((block.byte_range, block.first_ordinal));
                }
                Some((range, _)) if range.end == block.byte_range.start => {
                    range.end = block.byte_range.end;
                }
                Some((range, first_ord)) => {
                    let bytes = self.sstable_slice.read_bytes_slice(range.clone())?;
                    chunks.push((bytes, *first_ord));
                    cur_range = Some((block.byte_range, block.first_ordinal));
                }
                None => {
                    cur_range = Some((block.byte_range, block.first_ordinal));
                }
            }
        }
        if let Some((range, first_ord)) = cur_range {
            let bytes = self.sstable_slice.read_bytes_slice(range)?;
            chunks.push((bytes, first_ord));
        }
        if chunks.len() <= 1 {
            return Ok(None);
        }
        Ok(Some(chunks))
    }

    /// Streams all values whose keys contain `needle` from a pre-fetched contiguous SSTable byte
    /// slice using SIMD `memchr::memmem`, block-level rarest-byte prefiltering, and delta-prefix
    /// match inheritance.
    ///
    /// When `disallow_newline` is `true`, keys containing `b'\n'` are excluded so results match
    /// standard regex `.*<needle>.*` semantics (where `.` does not match `\n`) on valid UTF-8
    /// keys.
    pub fn stream_values_containing_bytes(
        &self,
        slice: OwnedBytes,
        needle: &[u8],
        disallow_newline: bool,
    ) -> io::Result<Vec<TSSTable::Value>>
    where
        TSSTable::Value: Clone,
    {
        let mut delta_reader = DeltaReader::<TSSTable::ValueReader>::new(slice);
        let mut values = Vec::new();
        let mut key_buf: Vec<u8> = Vec::with_capacity(256);

        if needle.is_empty() {
            while let Some(_block_keys) = delta_reader.next_block_lazy()? {
                key_buf.clear();
                while delta_reader.advance_in_block() {
                    let common_prefix_len = delta_reader.common_prefix_len();
                    let suffix = delta_reader.suffix();
                    key_buf.truncate(common_prefix_len);
                    key_buf.extend_from_slice(suffix);
                    if !disallow_newline || memchr::memchr(b'\n', &key_buf).is_none() {
                        delta_reader.ensure_values_loaded()?;
                        values.push(delta_reader.value().clone());
                    }
                }
            }
            return Ok(values);
        }

        let finder = memchr::memmem::Finder::new(needle);
        let filter_byte = select_prefilter_byte(needle);
        let needle_len = needle.len();
        let overlap = needle_len - 1;

        while let Some(block_keys) = delta_reader.next_block_lazy()? {
            if memchr::memchr(filter_byte, block_keys).is_none() {
                continue;
            }
            key_buf.clear();
            // Invariant: if `matched_end == Some(end)`, then `key_buf[..end]` contains `needle`
            // and (when `disallow_newline`) `key_buf` contains no `b'\n'`.
            let mut matched_end: Option<usize> = None;
            // True only when the previous key contained `needle` but was rejected because it also
            // contained `b'\n'`. In that rare case, the next key may inherit `needle` in its
            // prefix without inheriting the `\n`, so we search from index 0.
            let mut prev_rejected_by_newline = false;

            while delta_reader.advance_in_block() {
                let common_prefix_len = delta_reader.common_prefix_len();
                let suffix = delta_reader.suffix();
                key_buf.truncate(common_prefix_len);
                key_buf.extend_from_slice(suffix);

                if let Some(end) = matched_end {
                    if end <= common_prefix_len {
                        if !disallow_newline || memchr::memchr(b'\n', suffix).is_none() {
                            delta_reader.ensure_values_loaded()?;
                            values.push(delta_reader.value().clone());
                        } else {
                            matched_end = None;
                            prev_rejected_by_newline = true;
                        }
                        continue;
                    }
                    matched_end = None;
                }

                let search_start = if prev_rejected_by_newline {
                    0
                } else {
                    common_prefix_len.saturating_sub(overlap)
                };
                if let Some(rel_pos) = finder.find(&key_buf[search_start..]) {
                    if !disallow_newline || memchr::memchr(b'\n', &key_buf).is_none() {
                        matched_end = Some(search_start + rel_pos + needle_len);
                        prev_rejected_by_newline = false;
                        delta_reader.ensure_values_loaded()?;
                        values.push(delta_reader.value().clone());
                    } else {
                        prev_rejected_by_newline = true;
                    }
                } else {
                    prev_rejected_by_newline = false;
                }
            }
        }
        Ok(values)
    }

    pub(crate) fn sstable_delta_reader_block(
        &self,
        block_addr: BlockAddr,
    ) -> io::Result<DeltaReader<TSSTable::ValueReader>> {
        let data = self.sstable_slice.read_bytes_slice(block_addr.byte_range)?;
        Ok(TSSTable::delta_reader(data))
    }

    pub(crate) async fn sstable_delta_reader_block_async(
        &self,
        block_addr: BlockAddr,
    ) -> io::Result<DeltaReader<TSSTable::ValueReader>> {
        let data = self
            .sstable_slice
            .read_bytes_slice_async(block_addr.byte_range)
            .await?;
        Ok(TSSTable::delta_reader(data))
    }

    /// This function returns a file slice covering a set of sstable blocks
    /// that include the key range passed in arguments. Optionally returns
    /// only block for up to `limit` matching terms.
    ///
    /// It works by identifying
    /// - `first_block`: the block containing the start boundary key
    /// - `last_block`: the block containing the end boundary key.
    ///
    /// And then returning the range that spans over all blocks between.
    /// and including first_block and last_block, aka:
    /// `[first_block.start_offset .. last_block.end_offset)`
    ///
    /// Technically this function does not provide the tightest fit, as
    /// for simplification, it treats the start bound of the `key_range`
    /// as if it was inclusive, even if it is exclusive.
    /// On the rare edge case where a user asks for `(start_key, end_key]`
    /// and `start_key` happens to be the last key of a block, we return a
    /// slice that is the first block was not necessary.
    pub fn file_slice_for_range(
        &self,
        key_range: impl RangeBounds<[u8]>,
        limit: Option<u64>,
    ) -> FileSlice {
        let first_block_id = match key_range.start_bound() {
            Bound::Included(key) | Bound::Excluded(key) => {
                let Some(first_block_id) = self.sstable_index.locate_with_key(key) else {
                    return FileSlice::empty();
                };
                Some(first_block_id)
            }
            Bound::Unbounded => None,
        };

        let last_block_id = match key_range.end_bound() {
            Bound::Included(key) | Bound::Excluded(key) => self.sstable_index.locate_with_key(key),
            Bound::Unbounded => None,
        };

        let start_bound = if let Some(first_block_id) = first_block_id {
            let Some(block_addr) = self.sstable_index.get_block(first_block_id) else {
                return FileSlice::empty();
            };
            Bound::Included(block_addr.byte_range.start)
        } else {
            Bound::Unbounded
        };

        let last_block_id = if let Some(limit) = limit {
            let second_block_id = first_block_id.map(|id| id + 1).unwrap_or(0);
            if let Some(block_addr) = self.sstable_index.get_block(second_block_id) {
                let ordinal_limit = block_addr.first_ordinal + limit;
                let last_block_limit = self.sstable_index.locate_with_ord(ordinal_limit);
                if let Some(last_block_id) = last_block_id {
                    Some(last_block_id.min(last_block_limit))
                } else {
                    Some(last_block_limit)
                }
            } else {
                last_block_id
            }
        } else {
            last_block_id
        };
        let end_bound = last_block_id
            .and_then(|block_id| self.sstable_index.get_block(block_id))
            .map(|block_addr| Bound::Excluded(block_addr.byte_range.end))
            .unwrap_or(Bound::Unbounded);

        self.sstable_slice.slice((start_bound, end_bound))
    }

    fn get_block_iterator_for_range_and_automaton<'a>(
        &'a self,
        key_range: impl RangeBounds<[u8]>,
        automaton: &'a impl Automaton,
        merge_holes_under_bytes: usize,
    ) -> impl Iterator<Item = BlockAddr> + 'a {
        let lower_bound = match key_range.start_bound() {
            Bound::Included(key) | Bound::Excluded(key) => {
                self.sstable_index.locate_with_key(key).unwrap_or(u64::MAX)
            }
            Bound::Unbounded => 0,
        };

        let upper_bound = match key_range.end_bound() {
            Bound::Included(key) | Bound::Excluded(key) => {
                self.sstable_index.locate_with_key(key).unwrap_or(u64::MAX)
            }
            Bound::Unbounded => u64::MAX,
        };
        let block_range = lower_bound..=upper_bound;
        self.sstable_index
            .get_block_for_automaton(automaton)
            .filter(move |(block_id, _)| block_range.contains(block_id))
            .map(|(_, block_addr)| block_addr)
            .coalesce(move |first, second| {
                if first.byte_range.end + merge_holes_under_bytes >= second.byte_range.start {
                    Ok(BlockAddr {
                        first_ordinal: first.first_ordinal,
                        byte_range: first.byte_range.start..second.byte_range.end,
                    })
                } else {
                    Err((first, second))
                }
            })
    }

    /// Opens a `TermDictionary`.
    pub fn open(term_dictionary_file: FileSlice) -> io::Result<Self> {
        let num_bytes = term_dictionary_file.num_bytes();
        let (main_slice, footer_len_slice) = term_dictionary_file.split_from_end(20);
        let mut footer_len_bytes: OwnedBytes = footer_len_slice.read_bytes()?;
        let index_offset = u64::deserialize(&mut footer_len_bytes)?;
        let num_terms = u64::deserialize(&mut footer_len_bytes)?;
        let version = u32::deserialize(&mut footer_len_bytes)?;
        let (sstable_slice, index_slice) = main_slice.split(index_offset as usize);
        let sstable_index_bytes = index_slice.read_bytes()?;

        let sstable_index = SSTableIndex::open(version, index_offset, sstable_index_bytes)?;

        Ok(Dictionary {
            sstable_slice,
            sstable_index,
            num_bytes,
            num_terms,
            phantom_data: PhantomData,
        })
    }

    /// Creates a term dictionary from the supplied bytes.
    pub fn from_bytes(owned_bytes: OwnedBytes) -> io::Result<Self> {
        Dictionary::open(FileSlice::new(Arc::new(owned_bytes)))
    }

    /// Creates an empty term dictionary which contains no terms.
    pub fn empty() -> Self {
        let term_dictionary_data: Vec<u8> = Self::builder(Vec::<u8>::new())
            .expect("Creating a TermDictionaryBuilder in a Vec<u8> should never fail")
            .finish()
            .expect("Writing in a Vec<u8> should never fail");
        let empty_dict_file = FileSlice::from(term_dictionary_data);
        Dictionary::open(empty_dict_file).unwrap()
    }

    /// Returns the number of terms in the dictionary.
    /// Term ordinals range from 0 to `num_terms() - 1`.
    pub fn num_terms(&self) -> usize {
        self.num_terms as usize
    }

    /// Returns the total number of bytes used by the dictionary on disk.
    pub fn num_bytes(&self) -> ByteCount {
        self.num_bytes
    }

    /// Decode a DeltaReader up to key, returning the number of terms traversed
    ///
    /// If the key was not found, returns Ok(None).
    /// After calling this function, it is possible to call `DeltaReader::value` to get the
    /// associated value.
    fn decode_up_to_key<K: AsRef<[u8]>>(
        &self,
        key: K,
        sstable_delta_reader: &mut DeltaReader<TSSTable::ValueReader>,
    ) -> io::Result<Option<TermOrdinal>> {
        self.decode_up_to_or_next(key, sstable_delta_reader)
            .map(|hit| hit.into_exact())
    }
    /// Decode a DeltaReader up to key, returning the number of terms traversed
    ///
    /// If the key was not found, it returns the next term id.
    fn decode_up_to_or_next<K: AsRef<[u8]>>(
        &self,
        key: K,
        sstable_delta_reader: &mut DeltaReader<TSSTable::ValueReader>,
    ) -> io::Result<TermOrdHit> {
        let mut term_ord = 0;
        let key_bytes = key.as_ref();
        let mut key_comparator = DeltaKeyComparator::new();
        while sstable_delta_reader.advance()? {
            match key_comparator.compare(
                key_bytes,
                sstable_delta_reader.common_prefix_len(),
                sstable_delta_reader.suffix(),
            ) {
                Ordering::Less => {
                    term_ord += 1;
                }
                Ordering::Equal => return Ok(TermOrdHit::Exact(term_ord)),
                Ordering::Greater => return Ok(TermOrdHit::Next(term_ord)),
            }
        }

        Ok(TermOrdHit::Next(term_ord))
    }

    /// Returns the ordinal associated with a given term.
    pub fn term_ord<K: AsRef<[u8]>>(&self, key: K) -> io::Result<Option<TermOrdinal>> {
        let key_bytes = key.as_ref();

        let Some(block_addr) = self.sstable_index.get_block_with_key(key_bytes) else {
            return Ok(None);
        };

        let first_ordinal = block_addr.first_ordinal;
        let mut sstable_delta_reader = self.sstable_delta_reader_block(block_addr)?;
        self.decode_up_to_key(key_bytes, &mut sstable_delta_reader)
            .map(|opt| opt.map(|ord| ord + first_ordinal))
    }

    /// Returns the ordinal associated with a given term or its closest next term_id
    /// The closest next term_id may not exist.
    pub fn term_ord_or_next<K: AsRef<[u8]>>(&self, key: K) -> io::Result<TermOrdHit> {
        let key_bytes = key.as_ref();

        let Some(block_addr) = self.sstable_index.get_block_with_key(key_bytes) else {
            // TODO: Would be more consistent to return last_term id + 1
            return Ok(TermOrdHit::Next(u64::MAX));
        };

        let first_ordinal = block_addr.first_ordinal;
        let mut sstable_delta_reader = self.sstable_delta_reader_block(block_addr)?;
        self.decode_up_to_or_next(key_bytes, &mut sstable_delta_reader)
            .map(|opt| opt.map(|ord| ord + first_ordinal))
    }

    /// Converts strings into a Bound range.
    /// This does handle several special cases if the term is not exactly in the dictionary.
    /// e.g. [bbb, ddd]
    /// lower_bound: Bound::Included(aaa) => Included(0) // "Next" term id
    /// lower_bound: Bound::Excluded(aaa) => Included(0) // "Next" term id + Change the Bounds
    /// lower_bound: Bound::Included(ccc) => Included(1) // "Next" term id
    /// lower_bound: Bound::Excluded(ccc) => Included(1) // "Next" term id + Change the Bounds
    /// lower_bound: Bound::Included(zzz) => Included(2) // "Next" term id
    /// lower_bound: Bound::Excluded(zzz) => Included(2) // "Next" term id + Change the Bounds
    /// For zzz we should have some post processing to return an empty query`
    ///
    /// upper_bound: Bound::Included(aaa) => Excluded(0) // "Next" term id + Change the bounds
    /// upper_bound: Bound::Excluded(aaa) => Excluded(0) // "Next" term id
    /// upper_bound: Bound::Included(ccc) => Excluded(1) // Next term id + Change the bounds
    /// upper_bound: Bound::Excluded(ccc) => Excluded(1) // Next term id
    /// upper_bound: Bound::Included(zzz) => Excluded(2) // Next term id + Change the bounds
    /// upper_bound: Bound::Excluded(zzz) => Excluded(2) // Next term id
    pub fn term_bounds_to_ord<K: AsRef<[u8]>>(
        &self,
        lower_bound: Bound<K>,
        upper_bound: Bound<K>,
    ) -> io::Result<(Bound<TermOrdinal>, Bound<TermOrdinal>)> {
        let lower_bound = transform_bound_inner_res(&lower_bound, |start_bound_bytes| {
            let ord = self.term_ord_or_next(start_bound_bytes)?;
            match ord {
                TermOrdHit::Exact(ord) => Ok(TransformBound::Existing(ord)),
                TermOrdHit::Next(ord) => Ok(TransformBound::NewBound(Bound::Included(ord))), /* Change bounds to included */
            }
        })?;
        let upper_bound = transform_bound_inner_res(&upper_bound, |end_bound_bytes| {
            let ord = self.term_ord_or_next(end_bound_bytes)?;
            match ord {
                TermOrdHit::Exact(ord) => Ok(TransformBound::Existing(ord)),
                TermOrdHit::Next(ord) => Ok(TransformBound::NewBound(Bound::Excluded(ord))), /* Change bounds to excluded */
            }
        })?;
        Ok((lower_bound, upper_bound))
    }

    /// Returns the term associated with a given term ordinal.
    ///
    /// Term ordinals are defined as the position of the term in
    /// the sorted list of terms.
    ///
    /// Returns true if and only if the term has been found.
    ///
    /// Regardless of whether the term is found or not,
    /// the buffer may be modified.
    pub fn ord_to_term(&self, ord: TermOrdinal, bytes: &mut Vec<u8>) -> io::Result<bool> {
        // find block in which the term would be
        let block_addr = self.sstable_index.get_block_with_ord(ord);
        let first_ordinal = block_addr.first_ordinal;

        // then search inside that block only
        let mut sstable_delta_reader = self.sstable_delta_reader_block(block_addr)?;
        for _ in first_ordinal..=ord {
            if !sstable_delta_reader.advance()? {
                return Ok(false);
            }
            bytes.truncate(sstable_delta_reader.common_prefix_len());
            bytes.extend_from_slice(sstable_delta_reader.suffix());
        }
        Ok(true)
    }

    /// Returns the terms for a _sorted_ list of term ordinals.
    ///
    /// Returns true if and only if all terms have been found.
    pub fn sorted_ords_to_term_cb(
        &self,
        ords: &[TermOrdinal],
        mut cb: impl FnMut(&[u8]),
    ) -> io::Result<bool> {
        assert!(ords.is_sorted());
        if ords.is_empty() {
            return Ok(true);
        }

        // Dense sequential fast path: when the requested ordinals cover a large fraction of the
        // dictionary (e.g. top-level string cardinality over all docs), stream the contiguous
        // SSTable once instead of binary-searching and re-slicing every block boundary.
        if self.num_terms > 0 && ords.len().saturating_mul(4) >= self.num_terms as usize {
            let mut delta_reader =
                DeltaReader::<TSSTable::ValueReader>::new(self.sstable_slice.read_bytes()?);
            let mut bytes = Vec::with_capacity(256);
            let mut current_ord: u64 = 0;
            for &target_ord in ords {
                if target_ord >= self.num_terms {
                    return Ok(false);
                }
                if current_ord > 0 && target_ord == current_ord - 1 {
                    cb(&bytes);
                    continue;
                }
                while current_ord <= target_ord {
                    if !delta_reader.advance()? {
                        return Ok(false);
                    }
                    bytes.truncate(delta_reader.common_prefix_len());
                    bytes.extend_from_slice(delta_reader.suffix());
                    current_ord += 1;
                }
                cb(&bytes);
            }
            return Ok(true);
        }

        let mut ords = ords.iter().copied();
        let Some(mut ord) = ords.next() else {
            return Ok(true);
        };

        // Open the block for the first ordinal.
        let mut bytes = Vec::new();
        let (mut current_block_addr, block_id) = self.sstable_index.get_and_locate_with_ord(ord);
        let mut current_sstable_delta_reader =
            self.sstable_delta_reader_block(current_block_addr.clone())?;
        let mut current_block_ordinal = current_block_addr.first_ordinal;
        let mut current_block_end_bound = self
            .sstable_index
            .get_block(block_id + 1)
            .map(|block_addr| block_addr.first_ordinal)
            .unwrap_or(u64::MAX);

        loop {
            // move to the ord inside the current block
            while current_block_ordinal <= ord {
                if !current_sstable_delta_reader.advance()? {
                    return Ok(false);
                }
                bytes.truncate(current_sstable_delta_reader.common_prefix_len());
                bytes.extend_from_slice(current_sstable_delta_reader.suffix());
                current_block_ordinal += 1;
            }
            cb(&bytes);

            // fetch the next ordinal
            let next_ord = loop {
                let Some(next_ord) = ords.next() else {
                    return Ok(true);
                };
                if next_ord == ord {
                    // This is the same ordinal, let's just call the callback directly.
                    cb(&bytes);
                } else {
                    // we checked it was sorted beforehands
                    debug_assert!(next_ord > ord);
                    break next_ord;
                }
            };

            if next_ord >= current_block_end_bound {
                let (new_block_addr, block_id) =
                    self.sstable_index.get_and_locate_with_ord(next_ord);
                current_block_addr = new_block_addr;
                current_block_ordinal = current_block_addr.first_ordinal;
                let data = self
                    .sstable_slice
                    .read_bytes_slice(current_block_addr.byte_range.clone())?;
                current_sstable_delta_reader.reset_block(data);
                bytes.clear();
                current_block_end_bound = self
                    .sstable_index
                    .get_block(block_id + 1)
                    .map(|block_addr| block_addr.first_ordinal)
                    .unwrap_or(u64::MAX)
            }
            ord = next_ord;
        }
    }

    /// Returns the number of terms in the dictionary.
    pub fn term_info_from_ord(&self, term_ord: TermOrdinal) -> io::Result<Option<TSSTable::Value>> {
        // find block in which the term would be
        let block_addr = self.sstable_index.get_block_with_ord(term_ord);
        let first_ordinal = block_addr.first_ordinal;

        // then search inside that block only
        let mut sstable_reader = self.sstable_reader_block(block_addr)?;
        for _ in first_ordinal..=term_ord {
            if !sstable_reader.advance()? {
                return Ok(None);
            }
        }
        Ok(Some(sstable_reader.value().clone()))
    }

    /// Lookups the value corresponding to the key.
    pub fn get<K: AsRef<[u8]>>(&self, key: K) -> io::Result<Option<TSSTable::Value>> {
        if let Some(block_addr) = self.sstable_index.get_block_with_key(key.as_ref()) {
            let sstable_reader = self.sstable_delta_reader_block(block_addr)?;
            return self.do_get(key, sstable_reader);
        }
        Ok(None)
    }

    /// Lookups the value corresponding to the key.
    pub async fn get_async<K: AsRef<[u8]>>(&self, key: K) -> io::Result<Option<TSSTable::Value>> {
        if let Some(block_addr) = self.sstable_index.get_block_with_key(key.as_ref()) {
            let sstable_reader = self.sstable_delta_reader_block_async(block_addr).await?;
            return self.do_get(key, sstable_reader);
        }
        Ok(None)
    }

    fn do_get<K: AsRef<[u8]>>(
        &self,
        key: K,
        mut reader: DeltaReader<TSSTable::ValueReader>,
    ) -> io::Result<Option<TSSTable::Value>> {
        if let Some(_ord) = self.decode_up_to_key(key, &mut reader)? {
            Ok(Some(reader.value().clone()))
        } else {
            Ok(None)
        }
    }

    /// Returns a range builder, to stream all of the terms
    /// within an interval.
    pub fn range(&self) -> StreamerBuilder<'_, TSSTable> {
        StreamerBuilder::new(self, AlwaysMatch)
    }

    /// Returns a range builder filtered with a prefix.
    pub fn prefix_range<K: AsRef<[u8]>>(&self, prefix: K) -> StreamerBuilder<'_, TSSTable> {
        let lower_bound = prefix.as_ref();
        let mut upper_bound = lower_bound.to_vec();
        for idx in (0..upper_bound.len()).rev() {
            if upper_bound[idx] == 255 {
                upper_bound.pop();
            } else {
                upper_bound[idx] += 1;
                break;
            }
        }
        let mut builder = self.range().ge(lower_bound);
        if !upper_bound.is_empty() {
            builder = builder.lt(upper_bound);
        }
        builder
    }

    /// A stream of all the sorted terms.
    pub fn stream(&self) -> io::Result<Streamer<'_, TSSTable>> {
        self.range().into_stream()
    }

    /// Returns a search builder, to stream all of the terms
    /// within the Automaton
    pub fn search<'a, A: Automaton + 'a>(
        &'a self,
        automaton: A,
    ) -> StreamerBuilder<'a, TSSTable, A>
    where
        A::State: Clone,
    {
        StreamerBuilder::<TSSTable, A>::new(self, automaton)
    }

    #[doc(hidden)]
    pub async fn warm_up_dictionary(&self) -> io::Result<()> {
        self.sstable_slice.read_bytes_async().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ops::{Bound, Range};
    use std::sync::{Arc, Mutex};

    use common::OwnedBytes;

    use super::Dictionary;
    use crate::dictionary::TermOrdHit;
    use crate::{MonotonicU64SSTable, TermOrdinal};

    #[derive(Debug)]
    struct PermissionedHandle {
        bytes: OwnedBytes,
        allowed_range: Mutex<Range<usize>>,
    }

    impl PermissionedHandle {
        fn new(bytes: Vec<u8>) -> Self {
            let bytes = OwnedBytes::new(bytes);
            PermissionedHandle {
                allowed_range: Mutex::new(0..bytes.len()),
                bytes,
            }
        }

        fn restrict(&self, range: Range<usize>) {
            *self.allowed_range.lock().unwrap() = range;
        }
    }

    impl common::HasLen for PermissionedHandle {
        fn len(&self) -> usize {
            self.bytes.len()
        }
    }

    impl common::file_slice::FileHandle for PermissionedHandle {
        fn read_bytes(&self, range: Range<usize>) -> std::io::Result<OwnedBytes> {
            let allowed_range = self.allowed_range.lock().unwrap();
            if !allowed_range.contains(&range.start) || !allowed_range.contains(&(range.end - 1)) {
                return Err(std::io::Error::other(format!(
                    "invalid range, allowed {allowed_range:?}, requested {range:?}"
                )));
            }

            Ok(self.bytes.slice(range))
        }
    }

    fn make_test_sstable() -> (Dictionary<MonotonicU64SSTable>, Arc<PermissionedHandle>) {
        let mut builder = Dictionary::<MonotonicU64SSTable>::builder(Vec::new()).unwrap();

        // this makes 256k keys, enough to fill multiple blocks.
        for elem in 0..0x3ffff {
            let key = format!("{elem:05X}").into_bytes();
            builder.insert(&key, &elem).unwrap();
        }

        let table = builder.finish().unwrap();
        let table = Arc::new(PermissionedHandle::new(table));
        let slice = common::file_slice::FileSlice::new(table.clone());

        let dictionary = Dictionary::<MonotonicU64SSTable>::open(slice).unwrap();

        // if the last block is id 0, tests are meaningless
        assert_ne!(dictionary.sstable_index.locate_with_ord(u64::MAX), 0);
        assert_eq!(dictionary.num_terms(), 0x3ffff);
        (dictionary, table)
    }

    #[test]
    fn test_term_to_ord_or_next() {
        let dict = {
            let mut builder = Dictionary::<MonotonicU64SSTable>::builder(Vec::new()).unwrap();

            builder.insert(b"bbb", &1).unwrap();
            builder.insert(b"ddd", &2).unwrap();

            let table = builder.finish().unwrap();
            let table = Arc::new(PermissionedHandle::new(table));
            let slice = common::file_slice::FileSlice::new(table.clone());

            Dictionary::<MonotonicU64SSTable>::open(slice).unwrap()
        };

        assert_eq!(dict.term_ord_or_next(b"aaa").unwrap(), TermOrdHit::Next(0));
        assert_eq!(dict.term_ord_or_next(b"bbb").unwrap(), TermOrdHit::Exact(0));
        assert_eq!(dict.term_ord_or_next(b"bb").unwrap(), TermOrdHit::Next(0));
        assert_eq!(dict.term_ord_or_next(b"bbbb").unwrap(), TermOrdHit::Next(1));
        assert_eq!(dict.term_ord_or_next(b"dd").unwrap(), TermOrdHit::Next(1));
        assert_eq!(dict.term_ord_or_next(b"ddd").unwrap(), TermOrdHit::Exact(1));
        assert_eq!(dict.term_ord_or_next(b"dddd").unwrap(), TermOrdHit::Next(2));

        // This is not u64::MAX because for very small sstables (only one block),
        // we don't store an index, and the pseudo-index always reply that the
        // answer lies in block number 0
        assert_eq!(
            dict.term_ord_or_next(b"zzzzzzz").unwrap(),
            TermOrdHit::Next(2)
        );
    }
    #[test]
    fn test_term_to_ord_or_next_2() {
        let dict = {
            let mut builder = Dictionary::<MonotonicU64SSTable>::builder(Vec::new()).unwrap();

            let mut term_ord = 0;
            builder.insert(b"bbb", &term_ord).unwrap();

            // Fill blocks in between
            for elem in 0..50_000 {
                term_ord += 1;
                let key = format!("ccccc{elem:05X}").into_bytes();
                builder.insert(&key, &term_ord).unwrap();
            }

            term_ord += 1;
            builder.insert(b"eee", &term_ord).unwrap();

            let table = builder.finish().unwrap();
            let table = Arc::new(PermissionedHandle::new(table));
            let slice = common::file_slice::FileSlice::new(table.clone());

            Dictionary::<MonotonicU64SSTable>::open(slice).unwrap()
        };

        assert_eq!(dict.term_ord(b"bbb").unwrap(), Some(0));
        assert_eq!(dict.term_ord_or_next(b"bbb").unwrap(), TermOrdHit::Exact(0));
        assert_eq!(dict.term_ord_or_next(b"aaa").unwrap(), TermOrdHit::Next(0));
        assert_eq!(dict.term_ord_or_next(b"bb").unwrap(), TermOrdHit::Next(0));
        assert_eq!(dict.term_ord_or_next(b"bbbb").unwrap(), TermOrdHit::Next(1));
        assert_eq!(
            dict.term_ord_or_next(b"ee").unwrap(),
            TermOrdHit::Next(50001)
        );
        assert_eq!(
            dict.term_ord_or_next(b"eee").unwrap(),
            TermOrdHit::Exact(50001)
        );
        assert_eq!(
            dict.term_ord_or_next(b"eeee").unwrap(),
            TermOrdHit::Next(u64::MAX)
        );

        assert_eq!(
            dict.term_ord_or_next(b"zzzzzzz").unwrap(),
            TermOrdHit::Next(u64::MAX)
        );
    }

    #[test]
    fn test_term_bounds_to_ord() {
        let dict = {
            let mut builder = Dictionary::<MonotonicU64SSTable>::builder(Vec::new()).unwrap();

            builder.insert(b"bbb", &1).unwrap();
            builder.insert(b"ddd", &2).unwrap();

            let table = builder.finish().unwrap();
            let table = Arc::new(PermissionedHandle::new(table));
            let slice = common::file_slice::FileSlice::new(table.clone());

            Dictionary::<MonotonicU64SSTable>::open(slice).unwrap()
        };

        // Test cases for lower_bound
        let test_lower_bound = |bound, expected| {
            assert_eq!(
                dict.term_bounds_to_ord::<&[u8]>(bound, Bound::Included(b"ignored"))
                    .unwrap()
                    .0,
                expected
            );
        };

        test_lower_bound(Bound::Included(b"aaa".as_slice()), Bound::Included(0));
        test_lower_bound(Bound::Excluded(b"aaa".as_slice()), Bound::Included(0));

        test_lower_bound(Bound::Included(b"bbb".as_slice()), Bound::Included(0));
        test_lower_bound(Bound::Excluded(b"bbb".as_slice()), Bound::Excluded(0));

        test_lower_bound(Bound::Included(b"ccc".as_slice()), Bound::Included(1));
        test_lower_bound(Bound::Excluded(b"ccc".as_slice()), Bound::Included(1));

        test_lower_bound(Bound::Included(b"zzz".as_slice()), Bound::Included(2));
        test_lower_bound(Bound::Excluded(b"zzz".as_slice()), Bound::Included(2));

        // Test cases for upper_bound
        let test_upper_bound = |bound, expected| {
            assert_eq!(
                dict.term_bounds_to_ord::<&[u8]>(Bound::Included(b"ignored"), bound,)
                    .unwrap()
                    .1,
                expected
            );
        };
        test_upper_bound(Bound::Included(b"ccc".as_slice()), Bound::Excluded(1));
        test_upper_bound(Bound::Excluded(b"ccc".as_slice()), Bound::Excluded(1));
        test_upper_bound(Bound::Included(b"zzz".as_slice()), Bound::Excluded(2));
        test_upper_bound(Bound::Excluded(b"zzz".as_slice()), Bound::Excluded(2));
        test_upper_bound(Bound::Included(b"ddd".as_slice()), Bound::Included(1));
        test_upper_bound(Bound::Excluded(b"ddd".as_slice()), Bound::Excluded(1));
    }

    #[test]
    fn test_ord_term_conversion() {
        let (dic, slice) = make_test_sstable();

        let block = dic.sstable_index.get_block_with_ord(100_000);
        slice.restrict(block.byte_range);

        let mut res = Vec::new();

        // middle of a block
        assert!(dic.ord_to_term(100_000, &mut res).unwrap());
        assert_eq!(res, format!("{:05X}", 100_000).into_bytes());
        assert_eq!(dic.term_info_from_ord(100_000).unwrap().unwrap(), 100_000);
        assert_eq!(dic.get(&res).unwrap().unwrap(), 100_000);
        assert_eq!(dic.term_ord(&res).unwrap().unwrap(), 100_000);

        // start of a block
        assert!(dic.ord_to_term(block.first_ordinal, &mut res).unwrap());
        assert_eq!(res, format!("{:05X}", block.first_ordinal).into_bytes());
        assert_eq!(
            dic.term_info_from_ord(block.first_ordinal)
                .unwrap()
                .unwrap(),
            block.first_ordinal
        );
        assert_eq!(dic.get(&res).unwrap().unwrap(), block.first_ordinal);
        assert_eq!(dic.term_ord(&res).unwrap().unwrap(), block.first_ordinal);

        // end of a block
        let ordinal = block.first_ordinal - 1;
        let new_range = dic.sstable_index.get_block_with_ord(ordinal).byte_range;
        slice.restrict(new_range);
        assert!(dic.ord_to_term(ordinal, &mut res).unwrap());
        assert_eq!(res, format!("{ordinal:05X}").into_bytes());
        assert_eq!(dic.term_info_from_ord(ordinal).unwrap().unwrap(), ordinal);
        assert_eq!(dic.get(&res).unwrap().unwrap(), ordinal);
        assert_eq!(dic.term_ord(&res).unwrap().unwrap(), ordinal);

        // before first block
        // 1st block must be loaded for key-related operations
        let block = dic.sstable_index.get_block_with_ord(0);
        slice.restrict(block.byte_range);

        assert!(dic.get(b"$$$").unwrap().is_none());
        assert!(dic.term_ord(b"$$$").unwrap().is_none());

        // after last block
        // last block must be loaded for ord related operations
        let ordinal = 0x40000 + 10;
        let new_range = dic.sstable_index.get_block_with_ord(ordinal).byte_range;
        slice.restrict(new_range);
        assert!(!dic.ord_to_term(ordinal, &mut res).unwrap());
        assert!(dic.term_info_from_ord(ordinal).unwrap().is_none());

        // last block isn't required to be loaded for key related operations
        slice.restrict(0..0);
        assert!(dic.get(b"~~~").unwrap().is_none());
        assert!(dic.term_ord(b"~~~").unwrap().is_none());

        slice.restrict(0..slice.bytes.len());
        // between 1000F and 10010, test case where matched prefix > prefix kept
        assert!(dic.term_ord(b"1000G").unwrap().is_none());
        // shorter than 10000, tests prefix case
        assert!(dic.term_ord(b"1000").unwrap().is_none());
    }

    #[test]
    fn test_sorted_ords_to_term() {
        let (dic, _slice) = make_test_sstable();

        // Single term
        let mut terms = Vec::new();
        assert!(
            dic.sorted_ords_to_term_cb(&[100_000], |term| {
                terms.push(term.to_vec());
            })
            .unwrap()
        );
        assert_eq!(terms, vec![format!("{:05X}", 100_000).into_bytes(),]);
        // Single term
        let mut terms = Vec::new();
        let ords: Vec<TermOrdinal> = (100_001..100_002).collect();
        assert!(
            dic.sorted_ords_to_term_cb(&ords, |term| {
                terms.push(term.to_vec());
            })
            .unwrap()
        );
        assert_eq!(terms, vec![format!("{:05X}", 100_001).into_bytes(),]);
        // both terms
        let mut terms = Vec::new();
        assert!(
            dic.sorted_ords_to_term_cb(&[100_000, 100_001], |term| {
                terms.push(term.to_vec());
            })
            .unwrap()
        );
        assert_eq!(
            terms,
            vec![
                format!("{:05X}", 100_000).into_bytes(),
                format!("{:05X}", 100_001).into_bytes(),
            ]
        );
        // Test cross block
        let mut terms = Vec::new();
        let ords: Vec<TermOrdinal> = (98653..=98655).collect();
        assert!(
            dic.sorted_ords_to_term_cb(&ords, |term| {
                terms.push(term.to_vec());
            })
            .unwrap()
        );
        assert_eq!(
            terms,
            vec![
                format!("{:05X}", 98653).into_bytes(),
                format!("{:05X}", 98654).into_bytes(),
                format!("{:05X}", 98655).into_bytes(),
            ]
        );
        // redundant
        let mut terms = Vec::new();
        let ords: Vec<TermOrdinal> = vec![1, 1, 2];
        assert!(
            dic.sorted_ords_to_term_cb(&ords, |term| {
                terms.push(term.to_vec());
            })
            .unwrap()
        );
        assert_eq!(
            terms,
            vec![
                format!("{:05X}", 1).into_bytes(),
                format!("{:05X}", 1).into_bytes(),
                format!("{:05X}", 2).into_bytes(),
            ]
        );
        // redundant cross block
        let mut terms = Vec::new();
        let ords: Vec<TermOrdinal> = vec![98653, 98653, 98654, 98654, 98655, 98655];
        assert!(
            dic.sorted_ords_to_term_cb(&ords, |term| {
                terms.push(term.to_vec());
            })
            .unwrap()
        );
        assert_eq!(
            terms,
            vec![
                format!("{:05X}", 98_653).into_bytes(),
                format!("{:05X}", 98_653).into_bytes(),
                format!("{:05X}", 98_654).into_bytes(),
                format!("{:05X}", 98_654).into_bytes(),
                format!("{:05X}", 98_655).into_bytes(),
                format!("{:05X}", 98_655).into_bytes(),
            ]
        );
    }

    #[test]
    fn test_range() {
        let (dic, slice) = make_test_sstable();

        let start = dic
            .sstable_index
            .get_block_with_key(b"10000")
            .unwrap()
            .byte_range;
        let end = dic
            .sstable_index
            .get_block_with_key(b"18000")
            .unwrap()
            .byte_range;
        slice.restrict(start.start..end.end);

        let mut stream = dic.range().ge(b"10000").lt(b"18000").into_stream().unwrap();

        for i in 0x10000..0x18000 {
            assert!(stream.advance());
            assert_eq!(stream.term_ord(), i);
            assert_eq!(stream.value(), &i);
            assert_eq!(stream.key(), format!("{i:05X}").into_bytes());
        }
        assert!(!stream.advance());

        // verify limiting the number of results reduce the size read
        slice.restrict(start.start..(end.end - 1));

        let mut stream = dic
            .range()
            .ge(b"10000")
            .lt(b"18000")
            .limit(0xfff)
            .into_stream()
            .unwrap();

        for i in 0x10000..0x10fff {
            assert!(stream.advance());
            assert_eq!(stream.term_ord(), i);
            assert_eq!(stream.value(), &i);
            assert_eq!(stream.key(), format!("{i:05X}").into_bytes());
        }
        // there might be more successful elements after, though how many is undefined

        slice.restrict(0..slice.bytes.len());

        let mut stream = dic.stream().unwrap();
        for i in 0..0x3ffff {
            assert!(stream.advance());
            assert_eq!(stream.term_ord(), i);
            assert_eq!(stream.value(), &i);
            assert_eq!(stream.key(), format!("{i:05X}").into_bytes());
        }
        assert!(!stream.advance());
    }

    #[test]
    fn test_prefix() {
        let (dic, _slice) = make_test_sstable();
        {
            let mut stream = dic.prefix_range("1").into_stream().unwrap();
            for i in 0x10000..0x20000 {
                assert!(stream.advance());
                assert_eq!(stream.term_ord(), i);
                assert_eq!(stream.value(), &i);
                assert_eq!(stream.key(), format!("{i:05X}").into_bytes());
            }
            assert!(!stream.advance());
        }
        {
            let mut stream = dic.prefix_range("").into_stream().unwrap();
            for i in 0..0x3ffff {
                assert!(stream.advance(), "failed at {i:05X}");
                assert_eq!(stream.term_ord(), i);
                assert_eq!(stream.value(), &i);
                assert_eq!(stream.key(), format!("{i:05X}").into_bytes());
            }
            assert!(!stream.advance());
        }
        {
            let mut stream = dic.prefix_range("0FF").into_stream().unwrap();
            for i in 0x0ff00..=0x0ffff {
                assert!(stream.advance(), "failed at {i:05X}");
                assert_eq!(stream.term_ord(), i);
                assert_eq!(stream.value(), &i);
                assert_eq!(stream.key(), format!("{i:05X}").into_bytes());
            }
            assert!(!stream.advance());
        }
    }

    #[test]
    fn test_prefix_edge() {
        let dict = {
            let mut builder = Dictionary::<MonotonicU64SSTable>::builder(Vec::new()).unwrap();
            builder.insert(&[0, 254], &0).unwrap();
            builder.insert(&[0, 255], &1).unwrap();
            builder.insert(&[0, 255, 12], &2).unwrap();
            builder.insert(&[1], &2).unwrap();
            builder.insert(&[1, 0], &2).unwrap();
            let table = builder.finish().unwrap();
            let table = Arc::new(PermissionedHandle::new(table));
            let slice = common::file_slice::FileSlice::new(table.clone());
            Dictionary::<MonotonicU64SSTable>::open(slice).unwrap()
        };

        let mut stream = dict.prefix_range(&[0, 255]).into_stream().unwrap();
        assert!(stream.advance());
        assert_eq!(stream.key(), &[0, 255]);
        assert!(stream.advance());
        assert_eq!(stream.key(), &[0, 255, 12]);
        assert!(!stream.advance());
    }

    #[test]
    fn test_search_term_ord_is_the_dictionary_ordinal_not_a_scan_counter() {
        // An automaton prunes the blocks it cannot match, so the stream never reads the terms
        // in them. `term_ord()` must still report the term's ordinal in the whole dictionary;
        // counting advances alone reports its position among the blocks actually scanned, which
        // silently mislabels terms for any caller that resolves ordinals back to terms.
        let (dict, _) = make_test_sstable();

        // Matches exactly one key, far enough in that everything before it is pruned.
        let late_key = b"3FFFE";
        let expected_ord: TermOrdinal = 0x3FFFE;
        assert_eq!(dict.term_ord(late_key).unwrap(), Some(expected_ord));

        let pattern = tantivy_fst::Regex::new("3FFFE").unwrap();
        let mut stream = dict.search(pattern).into_stream().unwrap();
        assert!(stream.advance());
        assert_eq!(stream.key(), late_key);
        assert_eq!(stream.term_ord(), expected_ord);
        assert!(!stream.advance());

        // The ordinal a match reports must round-trip back to that same match, for every match
        // of a pattern whose hits are spread across several pruned-apart blocks.
        let pattern = tantivy_fst::Regex::new("[0-3]FFFF").unwrap();
        let mut stream = dict.search(pattern).into_stream().unwrap();
        let mut seen = 0;
        while stream.advance() {
            let key = stream.key().to_vec();
            let ord = stream.term_ord();
            assert_eq!(dict.term_ord(&key).unwrap(), Some(ord), "key {key:?}");
            seen += 1;
        }
        assert_eq!(seen, 3); // 0FFFF, 1FFFF, 2FFFF — the dictionary stops below 3FFFF
    }

    #[test]
    fn test_stream_values_containing_bytes_matches_regex() {
        let mut keys: Vec<Vec<u8>> = vec![
            b"".to_vec(),
            b"a".to_vec(),
            b"abcgoogle".to_vec(),
            b"abcgoogle\ndef".to_vec(),
            b"abcgoogle_after".to_vec(),
            b"abcgooglex".to_vec(),
            b"goo".to_vec(),
            b"goog".to_vec(),
            b"google".to_vec(),
            b"google\n".to_vec(),
            b"googlea".to_vec(),
            b"http://a.google.com/1".to_vec(),
            b"http://a.google.com/2\n3".to_vec(),
            b"http://a.google.com/4".to_vec(),
            b"http://example.com/no".to_vec(),
            b"http://google.com/page".to_vec(),
            b"http://gooogle.com/".to_vec(),
            b"z\ngoogle".to_vec(),
            b"z_google".to_vec(),
        ];
        // Add enough keys to span multiple blocks.
        for i in 0..4000u32 {
            keys.push(format!("http://site-{i:04}.example.org/path/{i}").into_bytes());
            if i % 97 == 0 {
                keys.push(format!("http://site-{i:04}.google.org/path/{i}").into_bytes());
            }
            if i % 193 == 0 {
                keys.push(format!("http://site-{i:04}.google.org/line\n{i}").into_bytes());
            }
        }
        keys.sort();
        keys.dedup();

        let mut buffer = Vec::new();
        let mut writer = Dictionary::<crate::MonotonicU64SSTable>::builder(&mut buffer).unwrap();
        for (ord, key) in keys.iter().enumerate() {
            writer.insert(key, &(ord as u64)).unwrap();
        }
        writer.finish().unwrap();
        let dict =
            Dictionary::<crate::MonotonicU64SSTable>::from_bytes(OwnedBytes::new(buffer)).unwrap();

        for (pattern, needle) in [
            (".*google.*", b"google".as_slice()),
            (".*\\.google\\..*", b".google.".as_slice()),
            (".*site-0097.*", b"site-0097".as_slice()),
            (".*NO_SUCH_NEEDLE.*", b"NO_SUCH_NEEDLE".as_slice()),
        ] {
            let regex = tantivy_fst::Regex::new(pattern).unwrap();
            let mut stream = dict.search(regex).into_stream().unwrap();
            let mut expected = Vec::new();
            while stream.advance() {
                expected.push(*stream.value());
            }

            let full = dict.sstable_slice.read_bytes().unwrap();
            let actual = dict
                .stream_values_containing_bytes(full, needle, true)
                .unwrap();
            assert_eq!(actual, expected, "mismatch for pattern {pattern}");

            // Also verify chunked scanning produces identical results.
            if let Some(chunks) = dict.all_block_chunks(4, 1024).unwrap() {
                let mut chunked = Vec::new();
                for (slice, _) in chunks {
                    chunked.extend(
                        dict.stream_values_containing_bytes(slice, needle, true)
                            .unwrap(),
                    );
                }
                assert_eq!(chunked, expected, "chunked mismatch for pattern {pattern}");
            }
        }
    }
}

/// Picks a byte from `needle` that is relatively rare in URLs and English text so a block-level
/// `memchr` scan can skip non-matching SSTable blocks before decoding delta keys.
#[inline]
fn select_prefilter_byte(needle: &[u8]) -> u8 {
    let mut best_byte = needle[0];
    let mut best_score = byte_rarity_score(best_byte);
    for &byte in &needle[1..] {
        let score = byte_rarity_score(byte);
        if score > best_score {
            best_score = score;
            best_byte = byte;
        }
    }
    best_byte
}

#[inline]
fn byte_rarity_score(byte: u8) -> u8 {
    match byte {
        b'e'
        | b't'
        | b'a'
        | b'o'
        | b'i'
        | b'n'
        | b's'
        | b'r'
        | b'h'
        | b'/'
        | b'.'
        | b'-'
        | b'_'
        | b':'
        | b' '
        | b'0'..=b'9' => 1,
        b'l' | b'd' | b'c' | b'u' | b'm' | b'p' | b'w' => 2,
        b'a'..=b'z' => 3,
        b'A'..=b'Z' => 4,
        _ => 5,
    }
}
