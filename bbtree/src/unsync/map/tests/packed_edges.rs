//! Packed edges (experiments): maps stay right against std's through
//! random inserts and removals, in the map's page arena where the edge
//! format needs one.

use std::collections::BTreeMap;
use std::vec::Vec;

use crate::Auto;
use crate::unsync::map::BBTreeMap;

/// `count` evenly spread `u64`s, from a fixed-seed xorshift.
fn spread_keys(count: usize) -> Vec<u64> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        })
        .collect()
}

#[cfg(any(feature = "arena-edges-32", feature = "arena-edges-24"))]
fn new_map() -> BBTreeMap<u64, u64, Auto, crate::PageArena> {
    BBTreeMap::new_in(crate::PageArena::new())
}

#[cfg(not(any(feature = "arena-edges-32", feature = "arena-edges-24")))]
fn new_map() -> BBTreeMap<u64, u64> {
    BBTreeMap::new()
}

#[test]
fn maps_match_std_through_inserts_and_removals() {
    let keys = spread_keys(200_000);
    let mut map = new_map();
    let mut model = BTreeMap::new();
    for &key in &keys {
        assert_eq!(map.insert(key, !key), model.insert(key, !key));
    }
    map.check_invariants();
    let copy = map.clone();
    for key in keys.iter().step_by(3) {
        assert_eq!(map.remove(key), model.remove(key));
    }
    map.check_invariants();
    for &key in keys.iter().step_by(7) {
        assert_eq!(map.insert(key, key), model.insert(key, key));
    }
    map.check_invariants();
    assert!(map.iter().eq(model.iter()));
    assert_eq!(copy.len(), keys.len());
    for key in &keys {
        assert_eq!(map.get(key), model.get(key));
        assert_eq!(copy.get(key), Some(&!key));
    }
}
