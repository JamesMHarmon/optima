pub fn single_bit_index(mut bit: u128) -> usize {
    let mut n = 0;

    if bit >> 64 != 0 {
        n += 64;
        bit >>= 64;
    }
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


#[cfg(test)]
mod tests {
    use super::*;

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
}