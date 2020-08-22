use std::sync::Arc;

#[derive(Debug)]
pub struct List<T> {
    head: Link<T>,
}

type Link<T> = Option<Arc<Node<T>>>;

#[derive(Debug)]
struct Node<T> {
    elem: T,
    next: Link<T>,
    len: usize
}

impl<T> List<T> {
    pub fn new() -> Self {
        Self { head: None }
    }

    pub fn append(&self, elem: T) -> List<T> {
        let len = self.len() + 1;

        List {
            head: Some(Arc::new(Node {
                elem,
                next: self.head.clone(),
                len
            }))
        }
    }

    pub fn head(&self) -> Option<&T> {
        self.head.as_ref().map(|n| &n.elem)
    }

    pub fn tail(&self) -> List<T> {
        List {
            head: self.head.as_ref().and_then(|n| n.next.clone())
        }
    }

    pub fn len(&self) -> usize {
        self.head.as_ref().map_or(0, |n| n.len)
    }
}

impl<T> Clone for List<T> {
    fn clone(&self) -> Self {
        Self {
            head: self.head.as_ref().cloned()
        }
    }
}

pub struct Iter<'a, T> {
    next: Option<&'a Node<T>>,
}

impl<T> List<T> {
    pub fn iter<'a>(&'a self) -> Iter<'a, T> {
        Iter { next: self.head.as_deref() }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.next.map(|node| {
            self.next = node.next.as_deref();
            &node.elem
        })
    }
}

#[cfg(test)]
mod test {
    use super::List;

    #[test]
    fn basics() {
        let list = List::new();
        assert_eq!(list.head(), None);

        let list = list.append(1).append(2).append(3);
        assert_eq!(list.head(), Some(&3));

        let list = list.tail();
        assert_eq!(list.head(), Some(&2));

        let list = list.tail();
        assert_eq!(list.head(), Some(&1));

        let list = list.tail();
        assert_eq!(list.head(), None);

        // Make sure empty tail works
        let list = list.tail();
        assert_eq!(list.head(), None);
    }
}
