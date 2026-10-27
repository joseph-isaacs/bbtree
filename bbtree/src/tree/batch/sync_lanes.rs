//! Batched lookups without futures: each lane is an explicit state machine,
//! run on the async executor's schedule. Each round advances every live lane
//! to its next fetch, which it prefetches, so the others run while its line
//! arrives.
//!
//! The same descent, stages and prefetches as [`Lane::find`], for an A/B
//! against the async lanes (doc 12).

use core::borrow::Borrow;

use super::{BatchSearch, Lane, Prefetcher};
use crate::inline_key::InlineKeyStrategy;
use crate::tree::node::Node;
use crate::tree::node::internal::Internal;
use crate::tree::node::leaf::{CandidateLine, Leaf};
use crate::tree::root::Tree;
use crate::tree::search::Found;

/// What a lane does when it next runs: the line it prefetched last
/// has arrived, or is arriving.
#[derive(Clone, Copy)]
enum Stage {
    /// Search the internal node, whose search lines are in hand.
    Internal,
    /// Read the chosen edge's child.
    Edge,
    /// Search the leaf, whose header line is in hand.
    LeafHeader,
    /// Compare the key with the leaf's only candidate entry.
    CandidateEntry,
    /// Compare, with the candidate's first heap bytes in hand.
    CandidateBytes,
}

/// After a lane runs: it prefetched its next line, or wrote its result.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Progress {
    Fetching,
    Finished,
}

/// [`Lane`] and where its descent stands. `index` is the chosen edge, then
/// the leaf candidate, as in [`Lane::find`].
struct SyncLane<'q, K, V, S, Q: ?Sized> {
    lane: Lane<'q, K, V, S, Q, ()>,
    index: u16,
    stage: Stage,
}

impl<'q, K, V, S, Q> SyncLane<'q, K, V, S, Q>
where
    K: Borrow<Q>,
    S: InlineKeyStrategy<K>,
    Q: Ord + ?Sized,
{
    #[inline(always)]
    fn new(lane: Lane<'q, K, V, S, Q, ()>) -> Self {
        Self {
            lane,
            index: 0,
            stage: Stage::Internal,
        }
    }

    /// Prefetches the query's window, if the root's search reads past what
    /// the query has in hand. False for a lane with nothing to search.
    /// Leaves the root's search to [`Self::advance`]: a second call site of
    /// it outlines the node searches.
    #[inline(always)]
    fn start(&self) -> bool {
        if !self.lane.active {
            return false;
        }
        if let Some(line) = self.lane.key.window_line(usize::from(self.lane.prefix_len)) {
            Prefetcher::hint(line);
        }
        true
    }

    /// Runs the lane's stage, up to its next fetch or its result.
    #[inline(always)]
    fn advance<'a>(&mut self, output: &mut Option<&'a V>) -> Progress
    where
        K: 'a,
        V: 'a,
    {
        let lane = &mut self.lane;
        match self.stage {
            Stage::Internal => match Internal::at(lane.node).search(&lane.key) {
                Ok(index) => {
                    let separator = Found::Separator(Internal::at(lane.node), index);
                    *output = Some(separator.key_value().1);
                    Progress::Finished
                }
                Err(edge) => {
                    self.index = edge as u16;
                    self.stage = Stage::Edge;
                    Self::fetch(Internal::at(lane.node).edge_line(edge))
                }
            },
            Stage::Edge => {
                let index = usize::from(self.index);
                // SAFETY: the search returned an initialized edge, and the
                // whole batch borrows the tree: no node or index changes.
                match unsafe { Internal::at(lane.node).child_unchecked(index) } {
                    Node::Leaf(leaf) => {
                        lane.node = leaf.node_ptr();
                        self.stage = Stage::LeafHeader;
                        Self::fetch(leaf.search_line())
                    }
                    Node::Internal(child) => {
                        lane.node = child.node_ptr();
                        self.stage = Stage::Internal;
                        Self::fetch(child.search_line())
                    }
                }
            }
            Stage::LeafHeader => {
                let leaf = Leaf::<K, V, S>::at(lane.node);
                if Leaf::<K, V, S>::plan().inline_width().is_some() {
                    match leaf.candidate_line(&lane.key) {
                        CandidateLine::Absent => return Progress::Finished,
                        CandidateLine::One(candidate, line) => {
                            self.index = candidate as u16;
                            lane.prefix_len = leaf.prefix_len() as u16;
                            self.stage = Stage::CandidateEntry;
                            return Self::fetch(line);
                        }
                        CandidateLine::Unknown => {}
                    }
                }
                *output = leaf
                    .find(&lane.key, BatchSearch)
                    .map(|index| Found::InLeaf(leaf, index).key_value().1);
                Progress::Finished
            }
            Stage::CandidateEntry => {
                if Leaf::<K, V, S>::plan().is_for_heap_byte_strings() {
                    // SAFETY: the candidate is an initialized key of this
                    // leaf, and the prefix the leaf's; neither changes while
                    // the tree is borrowed.
                    let line = unsafe {
                        Leaf::at(lane.node).candidate_bytes_line(
                            &lane.key,
                            usize::from(self.index),
                            usize::from(lane.prefix_len),
                        )
                    };
                    if let Some(line) = line {
                        self.stage = Stage::CandidateBytes;
                        return Self::fetch(line);
                    }
                }
                self.confirm_candidate(output)
            }
            Stage::CandidateBytes => self.confirm_candidate(output),
        }
    }

    /// Compares the key with the leaf's only candidate, in full.
    #[inline(always)]
    fn confirm_candidate<'a>(&self, output: &mut Option<&'a V>) -> Progress
    where
        K: 'a,
        V: 'a,
    {
        let leaf = Leaf::at(self.lane.node);
        // SAFETY: as in `advance`.
        let found = unsafe {
            leaf.find_candidate(
                &self.lane.key,
                usize::from(self.index),
                usize::from(self.lane.prefix_len),
            )
        };
        *output = found.map(|index| Found::InLeaf(leaf, index).key_value().1);
        Progress::Finished
    }

    #[inline(always)]
    fn fetch(line: *const u8) -> Progress {
        Prefetcher::hint(line);
        Progress::Fetching
    }
}

impl<'a, K: 'a, V: 'a, S, Q> SyncLane<'_, K, V, S, Q>
where
    K: Borrow<Q>,
    S: InlineKeyStrategy<K>,
    Q: Ord + ?Sized,
{
    /// The async executor's schedule: a bit per live lane, up to 64, each
    /// round advancing every live lane once. The next lane's index comes from
    /// the mask in a register, not from memory, so the CPU can start its step
    /// before the current one ends.
    #[inline(always)]
    fn run_rounds(lanes: &mut [Self], outputs: &mut [Option<&'a V>]) {
        debug_assert!(lanes.len() <= u64::BITS as usize);
        let mut live = 0_u64;
        for (index, lane) in lanes.iter().enumerate() {
            live |= u64::from(lane.start()) << index;
        }
        while live != 0 {
            let mut round = live;
            while round != 0 {
                let index = round.trailing_zeros() as usize;
                round &= round - 1;
                if lanes[index].advance(&mut outputs[index]) == Progress::Finished {
                    live &= !(1 << index);
                }
            }
        }
    }
}

impl<K, V, S: InlineKeyStrategy<K>> Tree<K, V, S> {
    /// [`Self::find_many`], its lanes explicit state machines instead of
    /// futures, on the async executor's schedule; `length` is the tree's.
    pub(crate) fn find_many_sync<Q, const N: usize>(
        &self,
        keys: [&Q; N],
        length: usize,
    ) -> [Option<&V>; N]
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if N < 2 || !Self::batches(length) {
            return keys.map(|key| self.find_in_batch(key));
        }
        let Some(root_page) = self.root_page() else {
            return [None; N];
        };
        let Node::Internal(root) = root_page.node() else {
            return keys.map(|key| self.find_in_batch(key));
        };
        let shared = root_page.shared_prefix();
        let mut lanes = keys.map(|key| SyncLane::new(Lane::new(key, root, shared, ())));
        let mut values = [None; N];
        const MASK_BITS: usize = u64::BITS as usize;
        if N <= MASK_BITS {
            SyncLane::run_rounds(&mut lanes, &mut values);
        } else {
            // More lanes than mask bits: 64 at a time.
            for (lanes, values) in lanes
                .chunks_mut(MASK_BITS)
                .zip(values.chunks_mut(MASK_BITS))
            {
                SyncLane::run_rounds(lanes, values);
            }
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use std::format;
    use std::string::String;
    use std::vec::Vec;

    use crate::inline_key::Auto;
    use crate::tree::node::page::TestPages;
    use crate::tree::root::Tree;
    use crate::tree::test_support::TestRng;

    #[test]
    fn sync_lookups_match_async_lanes() {
        let pages = TestPages::<u64, u64>::new();
        let mut tree = Tree::<_, _, Auto>::EMPTY;
        let mut length = 0;
        let mut keys: Vec<u64> = (0..10_000).map(|key| key * 2).collect();
        TestRng::new(55).shuffle(&mut keys);
        for &key in &keys {
            tree.insert(key, key, &pages, &mut length);
        }
        // 65 runs the lanes 64 at a time.
        for chunk in keys.chunks(65) {
            let queries: [u64; 65] =
                core::array::from_fn(|i| chunk[i % chunk.len()] + (i as u64 % 2));
            assert_eq!(
                tree.find_many_sync(queries.each_ref(), usize::MAX),
                tree.find_many(queries.each_ref(), usize::MAX)
            );
        }
        for chunk in keys.chunks(64) {
            let queries: [u64; 64] =
                core::array::from_fn(|i| chunk[i % chunk.len()] + (i as u64 % 2));
            assert_eq!(
                tree.find_many_sync(queries.each_ref(), usize::MAX),
                tree.find_many(queries.each_ref(), usize::MAX)
            );
        }
        tree.destroy(&pages);
        pages.assert_all_released();
    }

    #[test]
    fn sync_string_lookups_match_async_lanes() {
        let pages = TestPages::<String, usize>::new();
        let mut tree = Tree::<_, _, Auto>::EMPTY;
        let mut length = 0;
        let tail = "abc0123456789".repeat(4);
        let keys: Vec<String> = (0..5_000_u64)
            .map(|index| {
                format!(
                    "shared/{:016x}/{tail}",
                    index.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                )
            })
            .collect();
        for (index, key) in keys.iter().enumerate() {
            tree.insert(key.clone(), index, &pages, &mut length);
        }
        for chunk in keys.as_chunks::<16>().0 {
            let queries: [String; 16] = core::array::from_fn(|lane| {
                let mut key = chunk[lane].clone();
                if lane % 3 == 1 {
                    key.push('z');
                } else if lane % 3 == 2 {
                    key.replace_range(..1, "a");
                }
                key
            });
            let queries = queries.each_ref().map(String::as_str);
            assert_eq!(
                tree.find_many_sync(queries, usize::MAX),
                tree.find_many(queries, usize::MAX)
            );
        }
        tree.destroy(&pages);
        pages.assert_all_released();
    }
}
