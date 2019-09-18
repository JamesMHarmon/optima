use serde::de::SeqAccess;
use std::fmt;
use std::marker::PhantomData;
use serde::ser::{Serialize, Serializer,SerializeTuple};
use serde::de::{Deserialize, Deserializer, Visitor};

#[allow(non_snake_case)]
#[derive(PartialEq, Debug)]
pub struct NodeMetrics<A> {
    pub visits: usize,
    pub W: f32,
    pub children_visits: Vec<(A, usize)>
}

impl<A> Serialize for NodeMetrics<A>
where
    A: Serialize
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tup = serializer.serialize_tuple(3)?;
        tup.serialize_element(&self.visits)?;
        tup.serialize_element(&self.W)?;
        tup.serialize_element(&self.children_visits)?;
        tup.end()
    }
}

struct NodeMetricsVisitor<A> {
    marker: PhantomData<A>
}

impl<A> NodeMetricsVisitor<A> {
    fn new() -> Self {
        NodeMetricsVisitor {
            marker: PhantomData
        }
    }
}

impl<'de, A> Visitor<'de> for NodeMetricsVisitor<A>
where
    A: Deserialize<'de>
{
    type Value = NodeMetrics<A>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("NodeMetrics")
    }

    fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        Ok(NodeMetrics {
            visits: seq.next_element()?.unwrap(),
            W: seq.next_element()?.unwrap(),
            children_visits: seq.next_element()?.unwrap()
        })
    }
}

impl<'de, A> Deserialize<'de> for NodeMetrics<A>
where
    A: Deserialize<'de>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_tuple(3, NodeMetricsVisitor::new())
    }
}
