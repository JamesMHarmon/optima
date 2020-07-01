use std::sync::Mutex;
use owning_ref::MutexGuardRef;

const BYTES_PER_KB: usize = 1000;
const BYTES_PER_MB: usize = BYTES_PER_KB * 1000;

pub struct TranspositionTable<P> {
    table: Vec::<TranspositionRow<P>>,
    key_mask: u64
}

impl<P> TranspositionTable<P> {
    pub fn new(tt_cache_size: usize) -> TranspositionTable<P> {
        let power = calculate_tt_capacity_power::<TranspositionRow<P>>(tt_cache_size);
        let capacity = 2u128.pow(power as u32);
        let mut table = Vec::with_capacity(capacity as usize);
        
        for _ in 0..capacity {
            table.push(Mutex::new(None));
        }

        TranspositionTable {
            table,
            key_mask: get_key_mask(power)
        }
    }

    pub fn get<'ret, 'me:'ret>(&'me self, tranposition_key: u64) -> Option<MutexGuardRef<'ret, Option<TranspositionEntry<P>>, P>> {
        let idx = tranposition_key & self.key_mask;

        let entry_guard = self.table[idx as usize].lock().unwrap();

        if entry_guard.is_none() {
            return None;
        }

        if entry_guard.as_ref().unwrap().full_key == tranposition_key {
            let entry = MutexGuardRef::new(entry_guard);

            Some(entry.map(|e| &e.as_ref().unwrap().value))
        } else {
            None
        }
    }

    // pub fn set(&self, tranposition_key: u64) {
    //     let idx = tranposition_key & self.key_mask;

    //     let cluster = self.table[idx as usize].lock().unwrap();

    //     return None;
    // }
}

fn get_key_mask(power: usize) -> u64 {
    (1 << power) | ((1 << power) - 1)
}

type TranspositionRow<P> = Mutex<Option<TranspositionEntry<P>>>;

pub struct TranspositionEntry<P> {
    full_key: u64,
    value: P
}

fn calculate_tt_capacity_power<P>(tt_cache_size_mb: usize) -> usize {
    let bytes_for_entry = std::mem::size_of::<TranspositionRow<P>>() as u128;
    let max_num_entries = (tt_cache_size_mb as u128 * BYTES_PER_MB as u128) / bytes_for_entry;
    
    let mut capacity_power = max_num_entries.checked_next_power_of_two().expect("Failed to determine cache size");

    if !max_num_entries.is_power_of_two() {
        capacity_power = capacity_power - 1;
    }

    capacity_power as usize
}
