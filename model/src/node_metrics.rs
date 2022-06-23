use serde::de::SeqAccess;
use serde::de::{Deserialize, Deserializer, Visitor};
use serde::ser::{Serialize, SerializeTuple, Serializer};
use std::fmt;
use std::marker::PhantomData;

#[allow(non_snake_case)]
#[derive(PartialEq, Debug)]
pub struct NodeMetrics<A, V> {
    pub visits: usize,
    pub value: V,
    pub children: Vec<NodeChildMetrics<A>>,
}

#[allow(non_snake_case)]
#[derive(PartialEq, Debug)]
pub struct NodeChildMetrics<A> {
    action: A,
    Q: f32,
    M: f32,
    visits: usize,
}

#[allow(non_snake_case)]
impl<A> NodeChildMetrics<A> {
    pub fn new(action: A, Q: f32, M: f32, visits: usize) -> Self {
        Self {
            action,
            Q,
            M,
            visits,
        }
    }

    pub fn action(&self) -> &A {
        &self.action
    }

    pub fn Q(&self) -> f32 {
        self.Q
    }

    pub fn M(&self) -> f32 {
        self.M
    }

    pub fn visits(&self) -> usize {
        self.visits
    }
}

impl<A, V> Serialize for NodeMetrics<A, V>
where
    A: Serialize,
    V: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tup = serializer.serialize_tuple(3)?;
        tup.serialize_element(&self.visits)?;
        tup.serialize_element(&self.value)?;
        tup.serialize_element(&self.children)?;

        tup.end()
    }
}

struct NodeMetricsVisitor<A, V> {
    marker: PhantomData<A>,
    marker2: PhantomData<V>,
}

impl<A, V> NodeMetricsVisitor<A, V> {
    fn new() -> Self {
        NodeMetricsVisitor {
            marker: PhantomData,
            marker2: PhantomData,
        }
    }
}

impl<'de, A, V> Visitor<'de> for NodeMetricsVisitor<A, V>
where
    A: Deserialize<'de>,
    V: Deserialize<'de>,
{
    type Value = NodeMetrics<A, V>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("NodeMetrics")
    }

    fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        Ok(NodeMetrics {
            visits: seq.next_element()?.unwrap(),
            value: seq.next_element()?.unwrap(),
            children: seq.next_element()?.unwrap(),
        })
    }
}

impl<'de, A, V> Deserialize<'de> for NodeMetrics<A, V>
where
    A: Deserialize<'de>,
    V: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_tuple(3, NodeMetricsVisitor::new())
    }
}

impl<A> Serialize for NodeChildMetrics<A>
where
    A: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tup = serializer.serialize_tuple(4)?;

        tup.serialize_element(&self.action)?;
        tup.serialize_element(&self.Q)?;
        tup.serialize_element(&self.M)?;
        tup.serialize_element(&self.visits)?;

        tup.end()
    }
}

struct NodeChildMetricsVisitor<A> {
    marker: PhantomData<A>,
}

impl<A> NodeChildMetricsVisitor<A> {
    fn new() -> Self {
        NodeChildMetricsVisitor {
            marker: PhantomData,
        }
    }
}

impl<'de, A> Visitor<'de> for NodeChildMetricsVisitor<A>
where
    A: Deserialize<'de>,
{
    type Value = NodeChildMetrics<A>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("NodeChildMetrics")
    }

    fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        Ok(NodeChildMetrics {
            action: seq.next_element()?.unwrap(),
            Q: seq.next_element()?.unwrap(),
            M: seq.next_element()?.unwrap(),
            visits: seq.next_element()?.unwrap(),
        })
    }
}

impl<'de, A> Deserialize<'de> for NodeChildMetrics<A>
where
    A: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_tuple(4, NodeChildMetricsVisitor::new())
    }
}
