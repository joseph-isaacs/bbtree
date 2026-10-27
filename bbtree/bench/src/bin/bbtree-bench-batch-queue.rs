//! A/B of batched lookups (doc 12) on 10M-entry maps: one key a call
//! (`get`), lanes polled by the async executor (`get_many`), and lanes run
//! by a sync work queue (`get_many_queued`). One pass per cell, ns per key.

use std::hint::black_box;
use std::time::Instant;

use bbtree::unsync::BBTreeMap;
use bbtree_bench::BaseAllocator;

#[global_allocator]
static ALLOCATOR: BaseAllocator = BaseAllocator;

/// SplitMix64's finalizer: distinct inputs give distinct, scattered keys.
struct KeyScrambler;

impl KeyScrambler {
    fn scramble(index: u64) -> u64 {
        let mut z = index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// How a cell looks its probes up.
#[derive(Clone, Copy)]
enum Lookup {
    OneKeyACall,
    AsyncLanes,
    WorkQueue,
}

impl Lookup {
    const ALL: [Self; 3] = [Self::OneKeyACall, Self::AsyncLanes, Self::WorkQueue];

    fn name(self) -> &'static str {
        match self {
            Self::OneKeyACall => "get",
            Self::AsyncLanes => "get_many async",
            Self::WorkQueue => "get_many queue",
        }
    }
}

/// One map and its probes: keys it holds, and keys between them it lacks.
struct BatchCells<K> {
    map: BBTreeMap<K, u64>,
    hits: Vec<K>,
    misses: Vec<K>,
}

impl<K: Ord + Clone> BatchCells<K> {
    const ENTRIES: u64 = 10_000_000;
    const PROBES: usize = 3_000_000;

    fn new(key: impl Fn(u64) -> K) -> Self {
        // Even keys, in random insertion order, as a map filled over time
        // would be: Fisher-Yates.
        let mut order: Vec<u64> = (0..Self::ENTRIES).collect();
        for last in (1..order.len()).rev() {
            let pick = KeyScrambler::scramble(last as u64) % (last as u64 + 1);
            order.swap(last, pick as usize);
        }
        let mut map = BBTreeMap::new();
        for index in order {
            map.insert(key(index * 2), index);
        }
        assert_eq!(map.len() as u64, Self::ENTRIES);
        let probe = |salt: u64, odd: u64| -> Vec<K> {
            (0..Self::PROBES as u64)
                .map(|i| key(KeyScrambler::scramble(i ^ salt) % Self::ENTRIES * 2 + odd))
                .collect()
        };
        Self {
            map,
            hits: probe(0x1234_5678, 0),
            misses: probe(0x8765_4321, 1),
        }
    }

    /// Probes a block of this many keys at a time, the blocks dealt to
    /// the lookups in turn: each lookup times its own probes, and a noisy
    /// neighbour slows all three alike.
    const BLOCK: usize = 4096;

    /// Each lookup's ns per key, after checking how many probes hit.
    fn time<const N: usize>(&self, probes: &[K], expected_hits: usize) -> [f64; 3] {
        let mut nanos = [0_u128; 3];
        let mut probed = [0_usize; 3];
        for (block, keys) in probes.chunks(Self::BLOCK).enumerate() {
            let turn = block % Lookup::ALL.len();
            let chunks = keys.as_chunks::<N>().0;
            let start = Instant::now();
            let hits: usize = chunks
                .iter()
                .map(|chunk| self.lookup(Lookup::ALL[turn], chunk.each_ref()))
                .sum();
            nanos[turn] += start.elapsed().as_nanos();
            probed[turn] += chunks.len() * N;
            assert_eq!(hits, expected_hits.min(chunks.len() * N));
        }
        core::array::from_fn(|turn| nanos[turn] as f64 / probed[turn] as f64)
    }

    /// How many of `keys` the map holds.
    #[inline(always)]
    fn lookup<const N: usize>(&self, lookup: Lookup, keys: [&K; N]) -> usize {
        match lookup {
            Lookup::OneKeyACall => keys.iter().filter(|k| self.map.get(*k).is_some()).count(),
            Lookup::AsyncLanes => black_box(self.map.get_many(keys)).iter().flatten().count(),
            Lookup::WorkQueue => black_box(self.map.get_many_queued(keys))
                .iter()
                .flatten()
                .count(),
        }
    }

    fn report(&self, keys: &str) {
        for (outcome, probes, expected) in
            [("hit", &self.hits, usize::MAX), ("miss", &self.misses, 0)]
        {
            for batch in [32, 64] {
                let cells = match batch {
                    32 => self.time::<32>(probes, expected),
                    _ => self.time::<64>(probes, expected),
                };
                let row: Vec<String> = Lookup::ALL
                    .iter()
                    .zip(cells)
                    .map(|(lookup, ns)| format!("{} {ns:6.1}", lookup.name()))
                    .collect();
                println!("{keys:6} {outcome:4} x{batch}: {}", row.join(" | "));
            }
        }
    }
}

fn main() {
    let only = std::env::args().nth(1);
    if only.as_deref().is_none_or(|keys| keys == "u64") {
        BatchCells::new(|n| n).report("u64");
    }
    if only.as_deref().is_none_or(|keys| keys == "string") {
        BatchCells::new(|n| format!("user/{:016x}/profile", KeyScrambler::scramble(n)))
            .report("string");
    }
}
