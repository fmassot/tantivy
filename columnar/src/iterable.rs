use std::ops::Range;
use std::sync::Arc;

use crate::{ColumnValues, RowId};

pub trait Iterable<T = u64> {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = T> + '_>;

    /// Returns the underlying contiguous slice when available, enabling zero-virtual-dispatch
    /// passes during columnar serialization.
    #[inline]
    fn as_slice(&self) -> Option<&[T]> {
        None
    }
}

impl<T: Copy> Iterable<T> for &[T] {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = T> + '_> {
        Box::new(self.iter().copied())
    }

    #[inline]
    fn as_slice(&self) -> Option<&[T]> {
        Some(self)
    }
}

impl<T: Copy> Iterable<T> for Range<T>
where Range<T>: Iterator<Item = T>
{
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = T> + '_> {
        Box::new(self.clone())
    }
}

impl Iterable for Arc<dyn crate::ColumnValues<RowId>> {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        Box::new(self.iter().map(|row_id| row_id as u64))
    }
}
