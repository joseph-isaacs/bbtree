//! A/B of batched lookups (doc 12): one key a call (`get`), lanes as
//! futures polled by the async executor (`get_many`), and lanes as state
//! machines on the same schedule (`get_many_sync`). One pass per cell, ns
//! per key. 10M-entry maps wait on memory; 64K-entry maps, just past the
//! batch cutoff, stay in cache, so their time is the lanes' own work.

use std::env;

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
    SyncLanes,
}

impl Lookup {
    const ALL: [Self; 3] = [Self::OneKeyACall, Self::AsyncLanes, Self::SyncLanes];

    fn name(self) -> &'static str {
        match self {
            Self::OneKeyACall => "get",
            Self::AsyncLanes => "get_many async",
            Self::SyncLanes => "get_many sync",
        }
    }
}

/// One map and its probes: keys it holds, and keys between them it lacks.
struct BatchCells<K> {
    map: BBTreeMap<K, u64>,
    entries: u64,
    hits: Vec<K>,
    misses: Vec<K>,
}

impl<K: Ord + Clone> BatchCells<K> {
    const PROBES: usize = 3_000_000;

    fn new(entries: u64, key: impl Fn(u64) -> K) -> Self {
        // Even keys, in random insertion order, as a map filled over time
        // would be: Fisher-Yates.
        let mut order: Vec<u64> = (0..entries).collect();
        for last in (1..order.len()).rev() {
            let pick = KeyScrambler::scramble(last as u64) % (last as u64 + 1);
            order.swap(last, pick as usize);
        }
        let mut map = BBTreeMap::new();
        for index in order {
            map.insert(key(index * 2), index);
        }
        assert_eq!(map.len() as u64, entries);
        let probe = |salt: u64, odd: u64| -> Vec<K> {
            (0..Self::PROBES as u64)
                .map(|i| key(KeyScrambler::scramble(i ^ salt) % entries * 2 + odd))
                .collect()
        };
        Self {
            map,
            entries,
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
            Lookup::OneKeyACall => self.one_key_a_call(keys),
            Lookup::AsyncLanes => self.async_lanes(keys),
            Lookup::SyncLanes => self.sync_lanes(keys),
        }
    }

    // One function per lookup, so each one's code has its own symbol to
    // read: one call per batch.

    #[inline(never)]
    fn one_key_a_call<const N: usize>(&self, keys: [&K; N]) -> usize {
        keys.iter().filter(|k| self.map.get(*k).is_some()).count()
    }

    #[inline(never)]
    fn async_lanes<const N: usize>(&self, keys: [&K; N]) -> usize {
        black_box(self.map.get_many(keys)).iter().flatten().count()
    }

    #[inline(never)]
    fn sync_lanes<const N: usize>(&self, keys: [&K; N]) -> usize {
        black_box(self.map.get_many_sync(keys))
            .iter()
            .flatten()
            .count()
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
                let entries = self.entries;
                println!(
                    "{keys:6} {entries:>8} {outcome:4} x{batch}: {}",
                    row.join(" | ")
                );
            }
        }
    }
}

/// `bbtree-bench-batch-sync [u64|string] [entries]`: both key kinds at 10M
/// and 64K entries unless named.
fn main() {
    let keys = env::args().nth(1);
    let entries: Vec<u64> = match env::args().nth(2) {
        Some(entries) => vec![entries.parse().expect("entries is a number")],
        None => vec![10_000_000, 65_536],
    };
    for &entries in &entries {
        if keys.as_deref().is_none_or(|keys| keys == "u64") {
            BatchCells::new(entries, |n| n).report("u64");
        }
        if keys.as_deref().is_none_or(|keys| keys == "string") {
            BatchCells::new(entries, |n| {
                format!("user/{:016x}/profile", KeyScrambler::scramble(n))
            })
            .report("string");
        }
    }
}
