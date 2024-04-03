use half::f16;

pub struct ConvInputBuilder<'input> {
    channel_size: usize,
    num_channels: usize,
    input: &'input mut [f16],
}

impl<'input> ConvInputBuilder<'input> {
    pub fn new(channel_size: usize, input: &mut [f16]) -> ConvInputBuilder<'_> {
        assert_eq!(input.len() % channel_size, 0);

        let num_channels = input.len() / channel_size;

        ConvInputBuilder {
            channel_size,
            num_channels,
            input,
        }
    }

    pub fn channel<'b>(&'b mut self, channel: usize) -> ChannelBuilder<'b, 'input> {
        assert!(
            channel < self.num_channels,
            "Cannot write to the provided channel: {} as it is out of range of the number of channels: {}",
            channel,
            self.num_channels
        );

        ChannelBuilder {
            channel,
            channel_size: self.channel_size,
            input_builder: self,
        }
    }
}

pub struct ChannelBuilder<'builder, 'input> {
    channel: usize,
    channel_size: usize,
    input_builder: &'builder mut ConvInputBuilder<'input>,
}

impl ChannelBuilder<'_, '_> {
    pub fn fill(&mut self, value: f16) {
        for index in 0..self.input_builder.channel_size {
            self.write_at_idx(index, value)
        }
    }

    pub fn write_at_idx(&mut self, index: usize, value: f16) {
        assert!(
            index < self.channel_size,
            "Cannot write to the provided index: {} as it is out of range of the channel: {}",
            index,
            self.channel_size
        );

        let input_idx = self.channel + (self.input_builder.num_channels * index);
        self.input_builder.input[input_idx] = value;
    }

    pub fn set_bits_at_indexes<I>(&mut self, indexes: I)
    where
        I: Iterator<Item = usize>,
    {
        indexes.for_each(|index| self.write_at_idx(index, f16::ONE));
    }
}

#[cfg(test)]
mod test {

    use half::f16;

    use crate::ConvInputBuilder;

    #[test]
    fn test_fill() {
        const CHANNEL_SIZE: usize = 64;
        const NUM_CHANNELS: usize = 4;
        let mut input = [f16::ZERO; CHANNEL_SIZE * NUM_CHANNELS];
        let mut builder = ConvInputBuilder::new(CHANNEL_SIZE, &mut input);
        builder.channel(0).fill(f16::ONE);
        builder.channel(1).fill(f16::NEG_ONE);

        for (idx, val) in input.into_iter().enumerate() {
            if idx % NUM_CHANNELS == 0 {
                assert_eq!(val, f16::ONE);
            } else if idx % NUM_CHANNELS == 1 {
                assert_eq!(val, f16::NEG_ONE);
            } else {
                assert_eq!(val, f16::ZERO);
            }
        }
    }

    #[test]
    fn test_write_at_idx() {
        const CHANNEL_SIZE: usize = 64;
        const NUM_CHANNELS: usize = 4;
        let mut input = [f16::ZERO; CHANNEL_SIZE * NUM_CHANNELS];
        let mut builder = ConvInputBuilder::new(CHANNEL_SIZE, &mut input);
        builder.channel(0).write_at_idx(0, f16::from_f32(1.0));
        builder.channel(0).write_at_idx(1, f16::from_f32(2.0));
        builder.channel(0).write_at_idx(2, f16::from_f32(3.0));
        builder.channel(1).write_at_idx(0, f16::from_f32(4.0));
        builder.channel(1).write_at_idx(1, f16::from_f32(5.0));
        builder.channel(1).write_at_idx(2, f16::from_f32(6.0));

        assert_eq!(input[0], f16::from_f32(1.0));
        assert_eq!(input[NUM_CHANNELS], f16::from_f32(2.0));
        assert_eq!(input[NUM_CHANNELS * 2], f16::from_f32(3.0));
        assert_eq!(input[1], f16::from_f32(4.0));
        assert_eq!(input[NUM_CHANNELS + 1], f16::from_f32(5.0));
        assert_eq!(input[NUM_CHANNELS * 2 + 1], f16::from_f32(6.0));
    }

    #[test]
    fn set_bits_at_indexes() {
        const CHANNEL_SIZE: usize = 64;
        const NUM_CHANNELS: usize = 4;
        let mut input = [f16::ZERO; CHANNEL_SIZE * NUM_CHANNELS];
        let mut builder = ConvInputBuilder::new(CHANNEL_SIZE, &mut input);

        let indexes = [0, 5, 10];
        builder.channel(1).set_bits_at_indexes(indexes.into_iter());

        assert_eq!(input[1], f16::ONE);
        assert_eq!(input[NUM_CHANNELS * 5 + 1], f16::ONE);
        assert_eq!(input[NUM_CHANNELS * 10 + 1], f16::ONE);
        assert_eq!(input.into_iter().map(f16::to_f32).sum::<f32>(), 3.0);
    }

    #[test]
    #[should_panic(
        expected = "Cannot write to the provided index: 64 as it is out of range of the channel: 64"
    )]
    fn panics_when_writing_to_invalid_idx() {
        const CHANNEL_SIZE: usize = 64;
        const NUM_CHANNELS: usize = 4;
        let mut input = [f16::ZERO; CHANNEL_SIZE * NUM_CHANNELS];
        let mut builder = ConvInputBuilder::new(CHANNEL_SIZE, &mut input);

        builder.channel(1).write_at_idx(CHANNEL_SIZE, f16::ONE);
    }

    #[test]
    #[should_panic(
        expected = "Cannot write to the provided channel: 4 as it is out of range of the number of channels: 4"
    )]
    fn panics_when_writing_to_invalid_channel() {
        const CHANNEL_SIZE: usize = 64;
        const NUM_CHANNELS: usize = 4;
        let mut input = [f16::ZERO; CHANNEL_SIZE * NUM_CHANNELS];
        let mut builder = ConvInputBuilder::new(CHANNEL_SIZE, &mut input);

        builder.channel(NUM_CHANNELS).write_at_idx(0, f16::ONE);
    }
}
