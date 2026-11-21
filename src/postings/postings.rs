use crate::docset::DocSet;

/// Postings (also called inverted list)
///
/// For a given term, it is the list of doc ids of the doc
/// containing the term. Optionally, for each document,
/// it may also give access to the term frequency
/// as well as the list of term positions.
///
/// Its main implementation is `SegmentPostings`,
/// but other implementations mocking `SegmentPostings` exist,
/// for merging segments or for testing.
pub trait Postings: DocSet + 'static {
    /// The number of times the term appears in the document.
    fn term_freq(&self) -> u32;

    /// Returns the positions offsetted with a given value.
    /// It is not necessary to clear the `output` before calling this method.
    /// The output vector will be resized to the `term_freq`.
    fn positions_with_offset(&mut self, offset: u32, output: &mut Vec<u32>) {
        output.clear();
        self.append_positions_with_offset(offset, output);
    }

    /// Returns the positions offsetted with a given value.
    /// Data will be appended to the output.
    fn append_positions_with_offset(&mut self, offset: u32, output: &mut Vec<u32>);

    /// Returns the positions of the term in the given document.
    /// The output vector will be resized to the `term_freq`.
    fn positions(&mut self, output: &mut Vec<u32>) {
        self.positions_with_offset(0u32, output);
    }

    /// Fast path when `self.term_freq() == 1` that returns the single offsetted position
    /// without touching a `Vec<u32>`. Returns `None` if `term_freq() != 1` or positions are
    /// unavailable.
    #[inline]
    fn single_position_with_offset(&mut self, offset: u32) -> Option<u32> {
        if self.term_freq() != 1 {
            return None;
        }
        let mut tmp = Vec::with_capacity(1);
        self.append_positions_with_offset(offset, &mut tmp);
        if tmp.len() == 1 {
            Some(tmp[0])
        } else {
            None
        }
    }
}

impl Postings for Box<dyn Postings> {
    fn term_freq(&self) -> u32 {
        (**self).term_freq()
    }

    fn append_positions_with_offset(&mut self, offset: u32, output: &mut Vec<u32>) {
        (**self).append_positions_with_offset(offset, output);
    }

    #[inline]
    fn single_position_with_offset(&mut self, offset: u32) -> Option<u32> {
        (**self).single_position_with_offset(offset)
    }
}
