pub fn single_bit_index(bit: u128) -> usize {
    let mut bit = bit;
    let mut n = 0;

    if bit >> 64 != 0 {
        n += 64;
        bit >>= 64;
    }

    n += single_bit_index_u64(bit as u64);

    n
}

pub fn single_bit_index_u64(bit: u64) -> usize {
    let mut bit = bit;
    let mut n = 0;

    if bit >> 32 != 0 {
        n += 32;
        bit >>= 32;
    }
    if bit >> 16 != 0 {
        n += 16;
        bit >>= 16;
    }
    if bit >> 8 != 0 {
        n += 8;
        bit >>= 8;
    }
    if bit >> 4 != 0 {
        n += 4;
        bit >>= 4;
    }
    if bit >> 2 != 0 {
        n += 2;
        bit >>= 2;
    }
    if bit >> 1 != 0 {
        n += 1;
    }

    n
}

pub fn first_set_bit(bits: u64) -> u64 {
    let without_first_bit = bits & (bits - 1);
    let first_bit = bits ^ without_first_bit;
    first_bit
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_bit_index_no_bit() {
        let bit = 0b0;
        let expected_bit_index = 0;
        let actual_bit_index = single_bit_index(bit);
        assert_eq!(expected_bit_index, actual_bit_index);
    }

    #[test]
    fn test_single_bit_index_first_bit() {
        let bit = 0b1;
        let expected_bit_index = 0;
        let actual_bit_index = single_bit_index(bit);
        assert_eq!(expected_bit_index, actual_bit_index);
    }

    #[test]
    fn test_single_bit_index_second_bit() {
        let bit = 0b10;
        let expected_bit_index = 1;
        let actual_bit_index = single_bit_index(bit);
        assert_eq!(expected_bit_index, actual_bit_index);
    }

    #[test]
    fn test_single_bit_index_third_bit() {
        let bit = 0b100;
        let expected_bit_index = 2;
        let actual_bit_index = single_bit_index(bit);
        assert_eq!(expected_bit_index, actual_bit_index);
    }

    #[test]
    fn test_single_bit_index_last_bit() {
        let bit = 0b1 << 127;
        let expected_bit_index = 127;
        let actual_bit_index = single_bit_index(bit);
        assert_eq!(expected_bit_index, actual_bit_index);
    }

    #[test]
    fn test_single_bit_index_u64_last_bit() {
        let bit = 0b1 << 63;
        let expected_bit_index = 63;
        let actual_bit_index = single_bit_index_u64(bit);
        assert_eq!(expected_bit_index, actual_bit_index);
    }

    #[test]
    fn test_first_set_bit_1() {
        let bits = 0b1;
        let first_bit = 0b1;
        assert_eq!(first_set_bit(bits), first_bit);
    }

    #[test]
    fn test_first_set_bit_2() {
        let bits = 0b10;
        let first_bit = 0b10;
        assert_eq!(first_set_bit(bits), first_bit);
    }

    #[test]
    fn test_first_set_bit_multiple() {
        let bits = 0b1010;
        let first_bit = 0b0010;
        assert_eq!(first_set_bit(bits), first_bit);
    }

    #[test]
    fn test_first_set_bit_last() {
        let bits = 0b1 << 63;
        let first_bit = 0b1 << 63;
        assert_eq!(first_set_bit(bits), first_bit);
    }
}