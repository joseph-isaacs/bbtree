//! One-byte leaf keys (experiment, idea 4): maps stay right through
//! splits, steals and merges, whose separators carry 2-byte windows up from 1-byte leaves.

use std::vec::Vec;

use crate::unsync::map::BBTreeMap;

#[test]
fn maps_stay_right_through_inserts_and_removals() {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let keys: Vec<u64> = (0..100_000)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        })
        .collect();
    let mut map = BBTreeMap::new();
    for &key in &keys {
        map.insert(key, !key);
    }
    map.check_invariants();
    for key in &keys {
        assert_eq!(map.get(key), Some(&!key));
    }
    for key in &keys[..70_000] {
        assert_eq!(map.remove(key), Some(!key));
    }
    map.check_invariants();
    for key in &keys[..70_000] {
        assert_eq!(map.get(key), None);
    }
    for key in &keys[70_000..] {
        assert_eq!(map.get(key), Some(&!key));
    }
}
