/// A CoW like enum that can either borrow or own the inner value.
/// Doesn't require T: Clone unless converting to owned.
pub enum BorrowedOrOwned<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}

impl<T> std::ops::Deref for BorrowedOrOwned<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        match self {
            BorrowedOrOwned::Borrowed(r) => r,
            BorrowedOrOwned::Owned(v) => v,
        }
    }
}

impl<'a, T> std::convert::AsRef<T> for BorrowedOrOwned<'a, T> {
    fn as_ref(&self) -> &T {
        self
    }
}
