pub fn single_bit_index(bit: u128) -> usize {
    bit.trailing_zeros() as usize
}

pub fn single_bit_index_u64(bit: u64) -> usize {
    bit.trailing_zeros() as usize
}

pub fn first_set_bit(bits: u64) -> u64 {
    1 << single_bit_index_u64(bits)
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