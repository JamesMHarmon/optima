use tensorflow::DataType;
use tensorflow::TensorDataCRepr;
use tensorflow::TensorType;
use std::fmt::Formatter;
use std::fmt::Display;

/// Half provides a Rust type for Half.
#[derive(Debug,Clone,Copy,Default)]
pub struct Half(u16);

impl Display for Half {
    fn fmt(&self, f: &mut Formatter<'_>) -> ::std::fmt::Result {
        let val: f32 = (*self).into();
        Display::fmt(&val, f)
    }
}

impl Into<f32> for Half {
    fn into(self) -> f32 {
        unsafe {
            // Assumes that the architecture uses IEEE-754 natively for floats
            // and twos-complement for integers.
            std::mem::transmute::<u32, f32>((self.0 as u32) << 16)
        }
    }
}

impl From<f32> for Half {
    fn from(value: f32) -> Self {
        unsafe {
            // Assumes that the architecture uses IEEE-754 natively for floats
            // and twos-complement for integers.
            Half((std::mem::transmute::<f32, u32>(value) >> 16) as u16)
        }
    }
}

impl PartialEq for Half {
    fn eq(&self, other: &Half) -> bool {
        let x: f32 = (*self).into();
        let y: f32 = (*other).into();
        x.eq(&y)
    }
}

impl PartialOrd for Half {
    fn partial_cmp(&self, other: &Half) -> Option<std::cmp::Ordering> {
        let x: f32 = (*self).into();
        let y: f32 = (*other).into();
        x.partial_cmp(&y)
    }
}

impl TensorType for Half {
    type InnerType = TensorDataCRepr<Half>;
        
    fn data_type() -> DataType {
        DataType::Half
    }

    fn zero() -> Self {
        Half::from(0.0f32)
    }

    fn one() -> Self {
        Half::from(1.0f32)
    }

    fn is_repr_c() -> bool {
        true
    }

    fn unpack(_data: &[u8], _count: usize) -> tensorflow::Result<Vec<Self>> {
        Err(tensorflow::Status::new_set(
            tensorflow::Code::Unimplemented,
            concat!("Unpacking is not necessary for ", stringify!(Half))).unwrap())
    }

    fn packed_size(_data: &[Self]) -> usize {
        0
    }

    fn pack(_data: &[Self], _buffer: &mut [u8]) -> tensorflow::Result<()> {
        Err(tensorflow::Status::new_set(
            tensorflow::Code::Unimplemented,
            concat!("Packing is not necessary for ", stringify!(Half))).unwrap())
    }
}
