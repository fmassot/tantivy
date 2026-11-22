use std::io;

use sstable::SSTable;
use stacker::{Addr, MemoryArena, SharedArenaHashMap};

pub(crate) struct TermIdMapping {
    unordered_to_ord: Vec<OrderedId>,
}

impl TermIdMapping {
    pub fn to_ord(&self, unordered: UnorderedId) -> OrderedId {
        self.unordered_to_ord[unordered.0 as usize]
    }
}

/// When we add values, we cannot know their ordered id yet.
/// For this reason, we temporarily assign them a `UnorderedId`
/// that will be mapped to an `OrderedId` upon serialization.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct UnorderedId(pub u32);

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct OrderedId(pub u32);

/// `DictionaryBuilder` for dictionary encoding.
///
/// It stores the different terms encounterred and assigns them a temporary value
/// we call unordered id.
///
/// Upon serialization, we will sort the ids and hence build a `UnorderedId -> Term ordinal`
/// mapping.
pub(crate) struct DictionaryBuilder {
    dict: SharedArenaHashMap,
    /// Cached ID for the empty term `b""` (extremely common in sparse/wide string columns).
    empty_term_id: Option<UnorderedId>,
    /// 1-entry cache for consecutive identical non-empty terms: stores the arena address of the
    /// key bytes, its byte length, and its assigned `UnorderedId`.
    last_term_key_addr: Addr,
    last_term_len: u32,
    last_term_id: UnorderedId,
}

impl Default for DictionaryBuilder {
    fn default() -> Self {
        Self {
            dict: SharedArenaHashMap::default(),
            empty_term_id: None,
            last_term_key_addr: Addr::null_pointer(),
            last_term_len: 0,
            last_term_id: UnorderedId(0),
        }
    }
}

impl DictionaryBuilder {
    /// Get or allocate an unordered id.
    /// (This ID is simply an auto-incremented id.)
    #[inline]
    pub fn get_or_allocate_id(&mut self, term: &[u8], arena: &mut MemoryArena) -> UnorderedId {
        if term.is_empty() {
            if let Some(id) = self.empty_term_id {
                return id;
            }
            let next_id = self.dict.len() as u32;
            let unordered_id = self
                .dict
                .mutate_or_create(term, arena, |id_opt: Option<u32>| id_opt.unwrap_or(next_id));
            let id = UnorderedId(unordered_id);
            self.empty_term_id = Some(id);
            return id;
        }
        if !self.last_term_key_addr.is_null()
            && self.last_term_len as usize == term.len()
            && arena.slice(self.last_term_key_addr, term.len()) == term
        {
            return self.last_term_id;
        }
        let next_id = self.dict.len() as u32;
        let (unordered_id, val_addr) =
            self.dict
                .mutate_or_create_with_addr(term, arena, |id_opt: Option<u32>| {
                    id_opt.unwrap_or(next_id)
                });
        let id = UnorderedId(unordered_id);
        // `SharedArenaHashMap` truncates keys > `u16::MAX` and stores `[u16 len][key bytes][val]`
        // contiguously inside one arena page, so `val_addr - key_len` points to the stored key.
        if term.len() <= u16::MAX as usize {
            self.last_term_key_addr = val_addr.offset((term.len() as u32).wrapping_neg());
            self.last_term_len = term.len() as u32;
            self.last_term_id = id;
        }
        id
    }

    fn build_sorted_terms<'a>(&'a self, arena: &'a MemoryArena) -> Vec<(&'a [u8], UnorderedId)> {
        let mut terms: Vec<(&[u8], UnorderedId)> = self
            .dict
            .iter(arena)
            .map(|(k, v)| (k, arena.read(v)))
            .collect();
        terms.sort_unstable_by_key(|(key, _)| *key);
        terms
    }

    /// Serialize the dictionary into an fst, and returns the
    /// `UnorderedId -> TermOrdinal` map.
    pub fn serialize<'a, W: io::Write + 'a>(
        &self,
        arena: &MemoryArena,
        wrt: &mut W,
    ) -> io::Result<TermIdMapping> {
        let terms = self.build_sorted_terms(arena);
        // TODO Remove the allocation.
        let mut unordered_to_ord: Vec<OrderedId> = vec![OrderedId(0u32); terms.len()];
        let mut sstable_builder = sstable::VoidSSTable::writer(wrt);
        for (ord, (key, unordered_id)) in terms.into_iter().enumerate() {
            let ordered_id = OrderedId(ord as u32);
            sstable_builder.insert(key, &())?;
            unordered_to_ord[unordered_id.0 as usize] = ordered_id;
        }
        sstable_builder.finish()?;
        Ok(TermIdMapping { unordered_to_ord })
    }

    /// Build the `UnorderedId -> OrderedId` mapping in memory without serializing.
    pub fn build_term_id_mapping(&self, arena: &MemoryArena) -> TermIdMapping {
        let terms = self.build_sorted_terms(arena);
        let mut unordered_to_ord: Vec<OrderedId> = vec![OrderedId(0u32); terms.len()];
        for (ord, (_key, unordered_id)) in terms.into_iter().enumerate() {
            unordered_to_ord[unordered_id.0 as usize] = OrderedId(ord as u32);
        }
        TermIdMapping { unordered_to_ord }
    }

    pub(crate) fn mem_usage(&self) -> usize {
        self.dict.mem_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_builder() {
        let mut arena = MemoryArena::default();
        let mut dictionary_builder = DictionaryBuilder::default();
        let hello_uid = dictionary_builder.get_or_allocate_id(b"hello", &mut arena);
        let happy_uid = dictionary_builder.get_or_allocate_id(b"happy", &mut arena);
        let tax_uid = dictionary_builder.get_or_allocate_id(b"tax", &mut arena);
        let mut buffer = Vec::new();
        let id_mapping = dictionary_builder.serialize(&arena, &mut buffer).unwrap();
        assert_eq!(id_mapping.to_ord(hello_uid), OrderedId(1));
        assert_eq!(id_mapping.to_ord(happy_uid), OrderedId(0));
        assert_eq!(id_mapping.to_ord(tax_uid), OrderedId(2));
    }
}
