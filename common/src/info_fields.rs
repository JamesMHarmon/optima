/// Allows a snapshot type to expose additional named fields for UGI info output
/// lines. Each field is a static key string paired with an `f32` value.
///
/// The associated type `Fields` must be a fixed-size array (or any type that
/// dereferences to a slice) so that implementations carry no heap allocation.
pub trait InfoFields {
    type Fields: AsRef<[(&'static str, f32)]>;
    fn info_fields(&self) -> Self::Fields;
}
