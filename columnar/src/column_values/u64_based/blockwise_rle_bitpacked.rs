use std::io::{self, Write};
use std::num::NonZeroU64;
use std::ops::{Range, RangeInclusive};
use std::sync::Arc;

use common::{BinarySerializable, CountingWriter, DeserializeFrom, OwnedBytes, VInt};
use fastdivide::DividerU64;
use tantivy_bitpacker::{compute_num_bits, BitPacker, BitUnpacker};

use crate::column_values::u64_based::{ColumnCodec, ColumnCodecEstimator, ColumnStats};
use crate::{ColumnValues, RowId};

const BLOCK_SIZE: u32 = 256u32;
const BITMAP_WORDS: usize = (BLOCK_SIZE as usize) / 64;
const BITMAP_BYTES: usize = BITMAP_WORDS * 8;

#[derive(Debug, Default)]
struct BlockMeta {
    /// Block minimum value in normalized `(val - stats.min_value) / stats.gcd` space.
    min_val: u64,
    bit_unpacker: BitUnpacker,
    /// Number of distinct runs in this block (`1..=256`).
    num_runs: u16,
    data_start_offset: usize,
}

impl BlockMeta {
    #[inline]
    fn payload_num_bytes(&self) -> usize {
        let bit_width = self.bit_unpacker.bit_width() as usize;
        if bit_width == 0 {
            0
        } else if self.num_runs == BLOCK_SIZE as u16 {
            (BLOCK_SIZE as usize * bit_width) / 8
        } else {
            BITMAP_BYTES + (self.num_runs as usize * bit_width).div_ceil(8)
        }
    }

    /// Upper bound on normalized values in this block derived from `min_val` and `bit_width`.
    #[inline]
    fn max_val_upper_bound(&self) -> u64 {
        let bit_width = self.bit_unpacker.bit_width();
        let max_offset = if bit_width == 0 {
            0
        } else if bit_width == 64 {
            u64::MAX
        } else {
            (1u64 << bit_width) - 1
        };
        self.min_val.saturating_add(max_offset)
    }
}

impl BinarySerializable for BlockMeta {
    fn serialize<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        VInt(self.min_val).serialize(writer)?;
        self.bit_unpacker.bit_width().serialize(writer)?;
        ((self.num_runs - 1) as u8).serialize(writer)?;
        Ok(())
    }

    fn deserialize<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let min_val = VInt::deserialize(reader)?.0;
        let bit_width = u8::deserialize(reader)?;
        let num_runs_minus_1 = u8::deserialize(reader)?;
        Ok(BlockMeta {
            min_val,
            bit_unpacker: BitUnpacker::new(bit_width),
            num_runs: u16::from(num_runs_minus_1) + 1,
            data_start_offset: 0,
        })
    }
}

#[inline]
fn vint_encoded_len(mut val: u64) -> u64 {
    let mut len = 1u64;
    while val >= 0x80 {
        val >>= 7;
        len += 1;
    }
    len
}

#[inline]
fn compute_num_blocks(num_vals: u32) -> u32 {
    num_vals.div_ceil(BLOCK_SIZE)
}

#[inline(always)]
const fn div_ceil(numerator: u64, divisor: NonZeroU64) -> u64 {
    let quotient = numerator / divisor.get();
    let remainder = numerator % divisor.get();
    if remainder > 0 {
        quotient + 1
    } else {
        quotient
    }
}

fn transform_range_to_normalized(
    stats: &ColumnStats,
    range: RangeInclusive<u64>,
) -> Option<RangeInclusive<u64>> {
    if range.is_empty() || *range.end() < stats.min_value || *range.start() > stats.max_value {
        return None;
    }
    let shifted_start = range.start().saturating_sub(stats.min_value);
    let shifted_end = range.end().saturating_sub(stats.min_value);
    let norm_start = div_ceil(shifted_start, stats.gcd);
    let norm_end = shifted_end / stats.gcd;
    if norm_start > norm_end {
        None
    } else {
        Some(norm_start..=norm_end)
    }
}

/// Counts set bits at positions `0..=in_block` in the 256-bit `change_bitmap`.
/// Because bit 0 is always 1 for non-empty blocks, the result is always `>= 1`.
#[inline(always)]
fn rank_ones_inclusive(bitmap_bytes: &[u8], in_block: usize) -> u32 {
    debug_assert!(bitmap_bytes.len() >= BITMAP_BYTES);
    debug_assert!(in_block < BLOCK_SIZE as usize);
    let word_idx = in_block >> 6;
    let bit_idx = in_block & 63;
    let mut ones = 0u32;
    for w in 0..word_idx {
        let word = u64::from_le_bytes(bitmap_bytes[w * 8..w * 8 + 8].try_into().unwrap());
        ones += word.count_ones();
    }
    let cur_word = u64::from_le_bytes(
        bitmap_bytes[word_idx * 8..word_idx * 8 + 8]
            .try_into()
            .unwrap(),
    );
    let mask = u64::MAX >> (63 - bit_idx);
    ones + (cur_word & mask).count_ones()
}

struct BlockSummary {
    min_raw: u64,
    range_raw: u64,
    num_runs: u16,
}

#[derive(Default)]
pub struct BlockwiseRleBitpackedEstimator {
    summaries: Vec<BlockSummary>,
    cur_len: u16,
    cur_prev: u64,
    cur_min: u64,
    cur_max: u64,
    cur_runs: u16,
}

impl BlockwiseRleBitpackedEstimator {
    #[inline]
    fn flush_current_block(&mut self) {
        if self.cur_len == 0 {
            return;
        }
        self.summaries.push(BlockSummary {
            min_raw: self.cur_min,
            range_raw: self.cur_max - self.cur_min,
            num_runs: self.cur_runs,
        });
        self.cur_len = 0;
        self.cur_runs = 0;
    }
}

impl ColumnCodecEstimator for BlockwiseRleBitpackedEstimator {
    #[inline]
    fn collect(&mut self, value: u64) {
        if self.cur_len == 0 {
            self.cur_prev = value;
            self.cur_min = value;
            self.cur_max = value;
            self.cur_runs = 1;
            self.cur_len = 1;
        } else {
            if value != self.cur_prev {
                self.cur_prev = value;
                self.cur_runs += 1;
                if value < self.cur_min {
                    self.cur_min = value;
                }
                if value > self.cur_max {
                    self.cur_max = value;
                }
            }
            self.cur_len += 1;
            if self.cur_len == BLOCK_SIZE as u16 {
                self.flush_current_block();
            }
        }
    }

    fn finalize(&mut self) {
        self.flush_current_block();
    }

    fn estimate(&self, stats: &ColumnStats) -> Option<u64> {
        let gcd = stats.gcd.get();
        let divider = DividerU64::divide_by(gcd);
        let mut data_bytes = 0u64;
        let mut footer_bytes = 0u64;
        for summary in &self.summaries {
            let norm_min = divider.divide(summary.min_raw - stats.min_value);
            let norm_range = divider.divide(summary.range_raw);
            let bit_width = compute_num_bits(norm_range) as u64;
            footer_bytes += vint_encoded_len(norm_min) + 2;
            if bit_width > 0 {
                if summary.num_runs == BLOCK_SIZE as u16 {
                    data_bytes += (BLOCK_SIZE as u64 * bit_width) / 8;
                } else {
                    data_bytes +=
                        BITMAP_BYTES as u64 + (summary.num_runs as u64 * bit_width).div_ceil(8);
                }
            }
        }
        Some(stats.num_bytes() + data_bytes + footer_bytes + 4)
    }

    fn serialize(
        &self,
        stats: &ColumnStats,
        vals: &mut dyn Iterator<Item = u64>,
        wrt: &mut dyn Write,
    ) -> io::Result<()> {
        stats.serialize(wrt)?;
        let num_blocks = compute_num_blocks(stats.num_rows) as usize;
        let mut blocks = Vec::with_capacity(num_blocks);
        let gcd_divider = DividerU64::divide_by(stats.gcd.get());

        let mut run_vals: Vec<u64> = Vec::with_capacity(BLOCK_SIZE as usize);
        let mut bit_packer = BitPacker::new();

        for _ in 0..num_blocks {
            run_vals.clear();
            let mut bitmap = [0u64; BITMAP_WORDS];
            let mut block_len = 0usize;
            let mut prev_val = 0u64;
            let mut block_min = u64::MAX;
            let mut block_max = 0u64;

            for raw_val in vals.take(BLOCK_SIZE as usize) {
                let norm_val = gcd_divider.divide(raw_val - stats.min_value);
                if block_len == 0 || norm_val != prev_val {
                    bitmap[block_len >> 6] |= 1u64 << (block_len & 63);
                    run_vals.push(norm_val);
                    prev_val = norm_val;
                    if norm_val < block_min {
                        block_min = norm_val;
                    }
                    if norm_val > block_max {
                        block_max = norm_val;
                    }
                }
                block_len += 1;
            }
            assert!(block_len > 0);

            let num_runs = run_vals.len() as u16;
            let bit_width = compute_num_bits(block_max - block_min);

            if bit_width > 0 {
                if num_runs < BLOCK_SIZE as u16 {
                    for word in bitmap {
                        wrt.write_all(&word.to_le_bytes())?;
                    }
                }
                for &run_val in &run_vals {
                    bit_packer.write(run_val - block_min, bit_width, wrt)?;
                }
                bit_packer.flush(wrt)?;
            }

            blocks.push(BlockMeta {
                min_val: block_min,
                bit_unpacker: BitUnpacker::new(bit_width),
                num_runs,
                data_start_offset: 0,
            });
        }

        assert_eq!(blocks.len(), num_blocks);

        let mut counting_wrt = CountingWriter::wrap(wrt);
        for block in &blocks {
            block.serialize(&mut counting_wrt)?;
        }
        let footer_len = counting_wrt.written_bytes() as u32;
        footer_len.serialize(&mut counting_wrt)?;
        Ok(())
    }
}

/// Blockwise run-length encoded + bitpacked u64 codec (`BLOCK_SIZE = 256`).
///
/// Each 256-row block stores a 256-bit change bitmap (omitted for constant or duplicate-free
/// blocks) followed by bitpacked run values relative to the block minimum. Provides O(1) random
/// access via `popcnt` rank over at most 4 `u64` words with zero decompression overhead.
pub struct BlockwiseRleBitpackedCodec;

impl ColumnCodec<u64> for BlockwiseRleBitpackedCodec {
    type ColumnValues = BlockwiseRleBitpackedReader;
    type Estimator = BlockwiseRleBitpackedEstimator;

    fn load(mut bytes: OwnedBytes) -> io::Result<Self::ColumnValues> {
        let stats = ColumnStats::deserialize(&mut bytes)?;
        let footer_len: u32 = (&bytes[bytes.len() - 4..]).deserialize()?;
        let footer_offset = bytes.len() - 4 - footer_len as usize;
        let (data, mut footer) = bytes.split(footer_offset);
        let num_blocks = compute_num_blocks(stats.num_rows) as usize;
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut offset = 0usize;
        for _ in 0..num_blocks {
            let mut block = BlockMeta::deserialize(&mut footer)?;
            block.data_start_offset = offset;
            offset += block.payload_num_bytes();
            blocks.push(block);
        }
        Ok(BlockwiseRleBitpackedReader {
            blocks: blocks.into_boxed_slice().into(),
            data,
            stats,
        })
    }
}

#[derive(Clone)]
pub struct BlockwiseRleBitpackedReader {
    blocks: Arc<[BlockMeta]>,
    data: OwnedBytes,
    stats: ColumnStats,
}

impl BlockwiseRleBitpackedReader {
    #[inline(always)]
    fn decode_norm(&self, norm_val: u64) -> u64 {
        self.stats
            .min_value
            .wrapping_add(self.stats.gcd.get().wrapping_mul(norm_val))
    }

    #[inline(always)]
    fn block_len(&self, block_id: usize) -> usize {
        let block_start = (block_id as u32) * BLOCK_SIZE;
        (self.stats.num_rows - block_start).min(BLOCK_SIZE) as usize
    }

    /// Iterates over the `(run_idx, start_in_block, end_in_block)` spans of a bitmap-encoded block.
    #[inline]
    fn for_each_run_span(
        bitmap_bytes: &[u8],
        block_len: usize,
        mut callback: impl FnMut(u32, usize, usize),
    ) {
        let mut run_idx = 0u32;
        let mut run_start = 0usize;
        let mut have_prev = false;
        for word_idx in 0..BITMAP_WORDS {
            let base_bit = word_idx * 64;
            if base_bit >= block_len {
                break;
            }
            let mut word = u64::from_le_bytes(
                bitmap_bytes[word_idx * 8..word_idx * 8 + 8]
                    .try_into()
                    .unwrap(),
            );
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let pos = base_bit + bit;
                if pos >= block_len {
                    break;
                }
                if have_prev {
                    callback(run_idx, run_start, pos);
                    run_idx += 1;
                }
                run_start = pos;
                have_prev = true;
                word &= word - 1;
            }
        }
        if have_prev {
            callback(run_idx, run_start, block_len);
        }
    }
}

impl ColumnValues for BlockwiseRleBitpackedReader {
    #[inline(always)]
    fn get_val(&self, idx: u32) -> u64 {
        let block_id = (idx / BLOCK_SIZE) as usize;
        let in_block = (idx % BLOCK_SIZE) as usize;
        let block = &self.blocks[block_id];
        if block.bit_unpacker.bit_width() == 0 {
            return self.decode_norm(block.min_val);
        }
        let (run_idx, run_data_offset) = if block.num_runs == BLOCK_SIZE as u16 {
            (in_block as u32, block.data_start_offset)
        } else {
            let bitmap =
                &self.data[block.data_start_offset..block.data_start_offset + BITMAP_BYTES];
            let ones = rank_ones_inclusive(bitmap, in_block);
            (ones - 1, block.data_start_offset + BITMAP_BYTES)
        };
        let offset_val = block
            .bit_unpacker
            .get(run_idx, &self.data[run_data_offset..]);
        self.decode_norm(block.min_val + offset_val)
    }

    fn get_vals(&self, docs: &[RowId], output: &mut [u64]) {
        debug_assert_eq!(docs.len(), output.len());
        let len = docs.len();
        if len >= 4 {
            let first = docs[0];
            if docs[len - 1] == first.wrapping_add((len - 1) as u32)
                && docs
                    .iter()
                    .enumerate()
                    .all(|(idx, &doc)| doc == first.wrapping_add(idx as u32))
            {
                self.get_range(first as u64, output);
                return;
            }
        }
        let min_value = self.stats.min_value;
        let gcd = self.stats.gcd.get();
        let data = self.data.as_slice();
        let mut i = 0usize;

        while i < len {
            let block_id = (docs[i] >> 8) as usize;
            let block = &self.blocks[block_id];
            let block_base = min_value.wrapping_add(gcd.wrapping_mul(block.min_val));
            let bit_unpacker = block.bit_unpacker;

            let mut block_end = i + 1;
            while block_end < len && (docs[block_end] >> 8) as usize == block_id {
                block_end += 1;
            }
            let block_docs = &docs[i..block_end];
            let block_out = &mut output[i..block_end];
            i = block_end;

            if bit_unpacker.bit_width() == 0 {
                block_out.fill(block_base);
                continue;
            }

            if block.num_runs == BLOCK_SIZE as u16 {
                let run_data = &data[block.data_start_offset..];
                if gcd == 1 {
                    for (&doc, out_slot) in block_docs.iter().zip(block_out.iter_mut()) {
                        let offset_val = bit_unpacker.get(doc & 255, run_data);
                        *out_slot = block_base.wrapping_add(offset_val);
                    }
                } else {
                    for (&doc, out_slot) in block_docs.iter().zip(block_out.iter_mut()) {
                        let offset_val = bit_unpacker.get(doc & 255, run_data);
                        *out_slot = block_base.wrapping_add(gcd.wrapping_mul(offset_val));
                    }
                }
                continue;
            }

            let bitmap = &data[block.data_start_offset..block.data_start_offset + BITMAP_BYTES];
            let words = [
                u64::from_le_bytes(bitmap[0..8].try_into().unwrap()),
                u64::from_le_bytes(bitmap[8..16].try_into().unwrap()),
                u64::from_le_bytes(bitmap[16..24].try_into().unwrap()),
                u64::from_le_bytes(bitmap[24..32].try_into().unwrap()),
            ];
            let p1 = words[0].count_ones();
            let p2 = p1 + words[1].count_ones();
            let p3 = p2 + words[2].count_ones();
            let word_prefix = [0u32, p1, p2, p3];
            let run_data = &data[block.data_start_offset + BITMAP_BYTES..];

            let mut prev_run_idx = u32::MAX;
            let mut prev_val = 0u64;
            if gcd == 1 {
                for (&doc, out_slot) in block_docs.iter().zip(block_out.iter_mut()) {
                    let in_block = (doc & 255) as usize;
                    let word_idx = in_block >> 6;
                    let bit_idx = in_block & 63;
                    let mask = u64::MAX >> (63 - bit_idx);
                    let run_idx = word_prefix[word_idx] + (words[word_idx] & mask).count_ones() - 1;
                    if run_idx != prev_run_idx {
                        prev_run_idx = run_idx;
                        prev_val = block_base.wrapping_add(bit_unpacker.get(run_idx, run_data));
                    }
                    *out_slot = prev_val;
                }
            } else {
                for (&doc, out_slot) in block_docs.iter().zip(block_out.iter_mut()) {
                    let in_block = (doc & 255) as usize;
                    let word_idx = in_block >> 6;
                    let bit_idx = in_block & 63;
                    let mask = u64::MAX >> (63 - bit_idx);
                    let run_idx = word_prefix[word_idx] + (words[word_idx] & mask).count_ones() - 1;
                    if run_idx != prev_run_idx {
                        prev_run_idx = run_idx;
                        let offset_val = bit_unpacker.get(run_idx, run_data);
                        prev_val = block_base.wrapping_add(gcd.wrapping_mul(offset_val));
                    }
                    *out_slot = prev_val;
                }
            }
        }
    }

    fn get_range(&self, start: u64, output: &mut [u64]) {
        if output.is_empty() {
            return;
        }
        let min_value = self.stats.min_value;
        let gcd = self.stats.gcd.get();
        let data = self.data.as_slice();
        let mut pos = start as u32;
        let mut written = 0usize;
        let total = output.len();

        while written < total {
            let block_id = (pos / BLOCK_SIZE) as usize;
            let in_block_start = (pos % BLOCK_SIZE) as usize;
            let block_len = self.block_len(block_id);
            let take = (block_len - in_block_start).min(total - written);
            let in_block_end = in_block_start + take;
            let block = &self.blocks[block_id];
            let block_base = min_value.wrapping_add(gcd.wrapping_mul(block.min_val));
            let out_slice = &mut output[written..written + take];

            if block.bit_unpacker.bit_width() == 0 {
                out_slice.fill(block_base);
            } else if block.num_runs == BLOCK_SIZE as u16 {
                block.bit_unpacker.get_range(
                    in_block_start as u32,
                    &data[block.data_start_offset..],
                    out_slice,
                );
                if gcd == 1 {
                    for val in out_slice.iter_mut() {
                        *val = block_base.wrapping_add(*val);
                    }
                } else {
                    for val in out_slice.iter_mut() {
                        *val = block_base.wrapping_add(gcd.wrapping_mul(*val));
                    }
                }
            } else {
                let bitmap = &data[block.data_start_offset..block.data_start_offset + BITMAP_BYTES];
                let run_data = &data[block.data_start_offset + BITMAP_BYTES..];
                let bit_unpacker = block.bit_unpacker;
                let mut run_idx = rank_ones_inclusive(bitmap, in_block_start) - 1;
                let mut cur_start = in_block_start;
                let mut cur_val = if gcd == 1 {
                    block_base.wrapping_add(bit_unpacker.get(run_idx, run_data))
                } else {
                    block_base.wrapping_add(gcd.wrapping_mul(bit_unpacker.get(run_idx, run_data)))
                };

                // Scan only bitmap bits strictly inside `(in_block_start..in_block_end)`.
                let first_scan_pos = in_block_start + 1;
                if first_scan_pos < in_block_end {
                    let start_word = first_scan_pos >> 6;
                    let end_word = (in_block_end - 1) >> 6;
                    for word_idx in start_word..=end_word {
                        let base_bit = word_idx * 64;
                        let mut word = u64::from_le_bytes(
                            bitmap[word_idx * 8..word_idx * 8 + 8].try_into().unwrap(),
                        );
                        if word_idx == start_word {
                            let shift = first_scan_pos & 63;
                            word &= u64::MAX << shift;
                        }
                        if word_idx == end_word {
                            let end_in_word = in_block_end - base_bit;
                            if end_in_word < 64 {
                                word &= (1u64 << end_in_word) - 1;
                            }
                        }
                        while word != 0 {
                            let bit = word.trailing_zeros() as usize;
                            let next_run_pos = base_bit + bit;
                            let run_start = cur_start - in_block_start;
                            let run_end = next_run_pos - in_block_start;
                            if run_end == run_start + 1 {
                                out_slice[run_start] = cur_val;
                            } else {
                                out_slice[run_start..run_end].fill(cur_val);
                            }
                            cur_start = next_run_pos;
                            run_idx += 1;
                            let offset_val = bit_unpacker.get(run_idx, run_data);
                            cur_val = if gcd == 1 {
                                block_base.wrapping_add(offset_val)
                            } else {
                                block_base.wrapping_add(gcd.wrapping_mul(offset_val))
                            };
                            word &= word - 1;
                        }
                    }
                }
                let tail_start = cur_start - in_block_start;
                let tail_end = in_block_end - in_block_start;
                if tail_end == tail_start + 1 {
                    out_slice[tail_start] = cur_val;
                } else {
                    out_slice[tail_start..tail_end].fill(cur_val);
                }
            }

            pos += take as u32;
            written += take;
        }
    }

    fn get_vals_opt(&self, indexes: &[u32], output: &mut [Option<u64>]) {
        assert_eq!(indexes.len(), output.len());
        let mut buf = [0u64; 64];
        for (idx_chunk, out_chunk) in indexes.chunks(64).zip(output.chunks_mut(64)) {
            let chunk_len = idx_chunk.len();
            self.get_vals(idx_chunk, &mut buf[..chunk_len]);
            for (out_slot, &val) in out_chunk.iter_mut().zip(&buf[..chunk_len]) {
                *out_slot = Some(val);
            }
        }
    }

    fn get_row_ids_for_value_range(
        &self,
        range: RangeInclusive<u64>,
        doc_id_range: Range<u32>,
        positions: &mut Vec<u32>,
    ) {
        positions.clear();
        let clamped_end = doc_id_range.end.min(self.stats.num_rows);
        if doc_id_range.start >= clamped_end {
            return;
        }
        let Some(norm_range) = transform_range_to_normalized(&self.stats, range) else {
            return;
        };
        let norm_start = *norm_range.start();
        let norm_end = *norm_range.end();

        let first_block = (doc_id_range.start / BLOCK_SIZE) as usize;
        let last_block = ((clamped_end - 1) / BLOCK_SIZE) as usize;
        let mut block_positions: Vec<u32> = Vec::new();

        for block_id in first_block..=last_block {
            let block = &self.blocks[block_id];
            let block_max_ub = block.max_val_upper_bound();
            if block_max_ub < norm_start || block.min_val > norm_end {
                continue;
            }
            let block_doc_start = (block_id as u32) * BLOCK_SIZE;
            let block_len = self.block_len(block_id);
            let req_start = doc_id_range.start.max(block_doc_start);
            let req_end = clamped_end.min(block_doc_start + block_len as u32);
            if block.min_val >= norm_start && block_max_ub <= norm_end {
                positions.extend(req_start..req_end);
                continue;
            }
            let in_block_req_start = (req_start - block_doc_start) as usize;
            let in_block_req_end = (req_end - block_doc_start) as usize;

            if block.num_runs == BLOCK_SIZE as u16 {
                let run_data = &self.data[block.data_start_offset..];
                let block_norm_start = norm_start.saturating_sub(block.min_val);
                let block_norm_end = norm_end - block.min_val;
                block.bit_unpacker.get_ids_for_value_range(
                    block_norm_start..=block_norm_end,
                    in_block_req_start as u32..in_block_req_end as u32,
                    run_data,
                    &mut block_positions,
                );
                positions.extend(
                    block_positions
                        .iter()
                        .map(|&in_block| block_doc_start + in_block),
                );
            } else {
                let bitmap =
                    &self.data[block.data_start_offset..block.data_start_offset + BITMAP_BYTES];
                let run_data = &self.data[block.data_start_offset + BITMAP_BYTES..];
                Self::for_each_run_span(bitmap, block_len, |run_idx, run_start, run_end| {
                    if run_end <= in_block_req_start || run_start >= in_block_req_end {
                        return;
                    }
                    let norm_val = block.min_val + block.bit_unpacker.get(run_idx, run_data);
                    if norm_val >= norm_start && norm_val <= norm_end {
                        let clipped_start = run_start.max(in_block_req_start) as u32;
                        let clipped_end = run_end.min(in_block_req_end) as u32;
                        positions.extend(
                            (block_doc_start + clipped_start)..(block_doc_start + clipped_end),
                        );
                    }
                });
            }
        }
    }

    #[inline(always)]
    fn min_value(&self) -> u64 {
        self.stats.min_value
    }

    #[inline(always)]
    fn max_value(&self) -> u64 {
        self.stats.max_value
    }

    #[inline(always)]
    fn num_vals(&self) -> u32 {
        self.stats.num_rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column_values::u64_based::tests::create_and_validate;

    #[test]
    fn test_rle_bitpacked_constant_and_runs() {
        let mut vals = Vec::new();
        // Constant first block (256 identical values)
        vals.extend(std::iter::repeat_n(1_758_585_600_000_000_000u64, 256));
        // Clustered millisecond timestamps with long runs
        for t in 0u64..100 {
            let ts = 1_758_585_600_000_000_000u64 + (t % 30) * 1_000_000;
            vals.extend(std::iter::repeat_n(ts, 17));
        }
        // Duplicate-free full block (256 distinct values)
        for i in 0u64..256 {
            vals.push(1_758_585_600_000_000_000u64 + i * 1_000_000);
        }
        // Partial tail block
        vals.extend([1_758_585_600_000_000_000u64; 5]);
        vals.push(1_758_585_700_000_000_000u64);

        let (est_rate, actual_rate) =
            create_and_validate::<BlockwiseRleBitpackedCodec>(&vals, "rle_runs").unwrap();
        assert!((est_rate - actual_rate).abs() < 1e-6);
        assert!(
            actual_rate < 0.10,
            "expected strong compression, got {actual_rate}"
        );
    }

    #[test]
    fn test_rle_bitpacked_datasets() {
        let data_sets = crate::column_values::u64_based::tests::get_codec_test_datasets();
        for (mut data, name) in data_sets {
            create_and_validate::<BlockwiseRleBitpackedCodec>(&data, name);
            data.reverse();
            create_and_validate::<BlockwiseRleBitpackedCodec>(&data, name);
        }
    }

    #[test]
    fn test_rle_bitpacked_64bit_span() {
        let vals = vec![0u64, u64::MAX, u64::MAX, 0u64, 1u64 << 60, 1u64 << 60];
        create_and_validate::<BlockwiseRleBitpackedCodec>(&vals, "full_64bit").unwrap();
    }
}
