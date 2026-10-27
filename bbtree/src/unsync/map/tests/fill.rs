//! How full random inserts leave the leaves: the measure that the fuller
//! leaves experiment (idea 5) moves, printed with and without it.

use std::vec::Vec;

use crate::tree::node::leaf::Leaf;
use crate::unsync::map::BBTreeMap;

/// `count` evenly spread `u64`s, from a fixed-seed xorshift.
fn spread_keys(count: usize) -> Vec<u64> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        })
        .collect()
}

#[test]
fn random_inserts_fill_leaves() {
    let keys = spread_keys(200_000);
    let mut map = BBTreeMap::new();
    for &key in &keys {
        map.insert(key, !key);
    }
    map.check_invariants();
    for key in &keys {
        assert_eq!(map.get(key), Some(&!key));
    }
    let leaf_lens = map.tree.leaf_lens();
    let capacity = Leaf::<u64, u64, crate::Auto>::capacity();
    let fill = keys.len() as f64 / (leaf_lens.len() * capacity) as f64;
    std::println!(
        "{} leaves of capacity {capacity}, {:.1}% full",
        leaf_lens.len(),
        fill * 100.0
    );
    if cfg!(feature = "fuller-leaves") {
        assert!(fill > 0.75, "leaves only {:.1}% full", fill * 100.0);
    }
}
