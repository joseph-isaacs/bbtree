//! Batched lookups without futures: each lane is an explicit state machine,
//! and a FIFO work queue of lane indices runs them. Popping a lane advances
//! it to its next fetch, which it prefetches; a lane still descending goes
//! back to the queue's tail, so the others run while its line arrives.
//!
//! The same descent, stages and prefetches as [`Lane::find`], for an A/B
//! against the async executor (doc 12).

use core::borrow::Borrow;

use super::{BatchSearch, Lane, Prefetcher};
use crate::inline_key::InlineKeyStrategy;
use crate::tree::node::Node;
use crate::tree::node::internal::Internal;
use crate::tree::node::leaf::{CandidateLine, Leaf};
use crate::tree::root::Tree;
use crate::tree::search::Found;

/// What a queued lane does when it next runs: the line it prefetched last
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
struct QueuedLane<'q, K, V, S, Q: ?Sized> {
    lane: Lane<'q, K, V, S, Q, ()>,
    index: u16,
    stage: Stage,
}

impl<'q, K, V, S, Q> QueuedLane<'q, K, V, S, Q>
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

/// The indices of a batch's descending lanes, oldest fetch first, in a
/// ring of `N` slots, since each lane is in it at most once. Borrows the
/// ring rather than holding it: indexing an array in the same value as the
/// cursors keeps LLVM from holding the cursors in registers.
struct WorkQueue<'r, const N: usize> {
    ring: &'r mut [usize; N],
    head: usize,
    len: usize,
}

impl<'r, const N: usize> WorkQueue<'r, N> {
    fn new(ring: &'r mut [usize; N]) -> Self {
        Self {
            ring,
            head: 0,
            len: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, lane: usize) {
        let tail = self.head + self.len;
        self.ring[if tail >= N { tail - N } else { tail }] = lane;
        self.len += 1;
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<usize> {
        if self.len == 0 {
            return None;
        }
        let lane = self.ring[self.head];
        self.head = if self.head + 1 == N { 0 } else { self.head + 1 };
        self.len -= 1;
        Some(lane)
    }
}

impl<K, V, S: InlineKeyStrategy<K>> Tree<K, V, S> {
    /// [`Self::find_many`], its lanes run by a work queue instead of the
    /// async executor; `length` is the tree's.
    pub(crate) fn find_many_queued<Q, const N: usize>(
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
        let mut lanes = keys.map(|key| QueuedLane::new(Lane::new(key, root, shared, ())));
        let mut values = [None; N];
        let mut ring = [0; N];
        let mut queue = WorkQueue::new(&mut ring);
        for (index, lane) in lanes.iter().enumerate() {
            if lane.start() {
                queue.push(index);
            }
        }
        while let Some(index) = queue.pop() {
            if lanes[index].advance(&mut values[index]) == Progress::Fetching {
                queue.push(index);
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
    fn queued_lookups_match_async_lanes() {
        let pages = TestPages::<u64, u64>::new();
        let mut tree = Tree::<_, _, Auto>::EMPTY;
        let mut length = 0;
        let mut keys: Vec<u64> = (0..10_000).map(|key| key * 2).collect();
        TestRng::new(55).shuffle(&mut keys);
        for &key in &keys {
            tree.insert(key, key, &pages, &mut length);
        }
        for chunk in keys.chunks(64) {
            let queries: [u64; 64] =
                core::array::from_fn(|i| chunk[i % chunk.len()] + (i as u64 % 2));
            assert_eq!(
                tree.find_many_queued(queries.each_ref(), usize::MAX),
                tree.find_many(queries.each_ref(), usize::MAX)
            );
        }
        tree.destroy(&pages);
        pages.assert_all_released();
    }

    #[test]
    fn queued_string_lookups_match_async_lanes() {
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
                tree.find_many_queued(queries, usize::MAX),
                tree.find_many(queries, usize::MAX)
            );
        }
        tree.destroy(&pages);
        pages.assert_all_released();
    }
}
