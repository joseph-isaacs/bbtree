//! The entry probe (experiment, idea 2): lookups stay right with hints
//! fresh, stale after changes, or absent, and with fresh hints the probe
//! finds most keys of evenly spread `u64`s itself.

use std::vec::Vec;

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

fn assert_lookups(map: &BBTreeMap<u64, u64>, present: &[u64], absent: &[u64]) {
    for &key in present {
        assert_eq!(map.get(&key), Some(&!key), "key {key:#x}");
    }
    for key in absent {
        assert_eq!(map.get(key), None, "key {key:#x}");
    }
}

#[test]
fn lookups_stay_right_with_fresh_stale_and_no_hints() {
    let keys = spread_keys(60_000);
    let (present, absent) = keys.split_at(40_000);
    let mut map = BBTreeMap::new();
    for &key in present {
        map.insert(key, !key);
    }
    assert_lookups(&map, present, absent);

    map.refresh_len_hints();
    assert_lookups(&map, present, absent);

    // Removing half the keys and inserting others leaves the hints stale.
    let (removed, kept) = present.split_at(20_000);
    let (inserted, still_absent) = absent.split_at(10_000);
    for key in removed {
        assert_eq!(map.remove(key), Some(!key));
    }
    for &key in inserted {
        map.insert(key, !key);
    }
    let now_present: Vec<u64> = kept.iter().chain(inserted).copied().collect();
    let now_absent: Vec<u64> = removed.iter().chain(still_absent).copied().collect();
    assert_lookups(&map, &now_present, &now_absent);
    assert_eq!(map.len(), now_present.len());
}

#[test]
fn fresh_hints_find_most_spread_keys_at_the_probe() {
    let keys = spread_keys(200_000);
    let mut map = BBTreeMap::new();
    for &key in &keys {
        map.insert(key, !key);
    }
    map.refresh_len_hints();
    let errors: Vec<isize> = keys
        .iter()
        .filter_map(|key| map.tree.entry_probe_error(key))
        .collect();
    let mut histogram = [0usize; 9];
    for error in &errors {
        histogram[(error.clamp(&-4, &4) + 4) as usize] += 1;
    }
    std::println!(
        "{} of {} keys predicted; error -4..=4 (clamped): {histogram:?}",
        errors.len(),
        keys.len()
    );
    let found = keys
        .iter()
        .filter(|key| map.tree.entry_probe_finds(key))
        .count();
    let share = found as f64 / keys.len() as f64;
    std::println!("the entry probe finds {:.1}% of keys", share * 100.0);
    assert!(share > 0.5, "the probe finds only {:.1}%", share * 100.0);
}
