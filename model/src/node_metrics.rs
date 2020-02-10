use serde::de::SeqAccess;
use std::fmt;
use std::marker::PhantomData;
use serde::ser::{Serialize, Serializer,SerializeTuple};
use serde::de::{Deserialize, Deserializer, Visitor};

#[allow(non_snake_case)]
#[derive(PartialEq, Debug)]
pub struct NodeMetrics<A> {
    pub visits: usize,
    pub children: Vec<(A, f32, usize)>
}

impl<A> Serialize for NodeMetrics<A>
where
    A: Serialize
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let metrics = &self.children.iter().map(|(a, _, v)| (a, v)).collect::<Vec<_>>();

        let mut tup = serializer.serialize_tuple(2)?;
        tup.serialize_element(&self.visits)?;
        tup.serialize_element(&metrics)?;

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
        let visits = seq.next_element()?.unwrap();
        let metrics: Vec<(A,usize)> = seq.next_element()?.unwrap();

        Ok(NodeMetrics {
            visits,
            children: metrics.into_iter().map(|(a, v)| (a, 0.0, v)).collect()
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
        deserializer.deserialize_tuple(2, NodeMetricsVisitor::new())
    }
}
