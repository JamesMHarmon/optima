use serde::de::SeqAccess;
use serde::de::{Deserialize, Deserializer, Visitor};
use serde::ser::{Serialize, SerializeTuple, Serializer};
use std::fmt;
use std::marker::PhantomData;

#[allow(non_snake_case)]
#[derive(PartialEq, Debug)]
pub struct NodeMetrics<A, P> {
    /// The total number of visits of the node. Should be children.visits.sum() + 1.
    pub visits: usize,
    /// Ancillery predictions by the neural network. Like score difference or moves left.
    pub predictions: P,
    /// The valid actions of the current game_state of the node.
    pub children: Vec<EdgeMetrics<A, P>>,
}

#[allow(non_snake_case)]
impl<A, P> NodeMetrics<A, P> {
    pub fn child_max_visits(&self) -> &EdgeMetrics<A, P> {
        self.children.iter().max_by_key(|c| c.visits).unwrap()
    }
}

#[allow(non_snake_case)]
#[derive(PartialEq, Debug)]
pub struct EdgeMetrics<A, PV> {
    /// The action that this edge represents.
    action: A,
    /// The number of visits for the child node of this specific edge.
    visits: usize,
    /// Predictions by the neural network that have been propagated from child to parent nodes. Like score difference or moves left.
    propagatedValues: PV,
}

#[allow(non_snake_case)]
impl<A, PV> EdgeMetrics<A, PV> {
    pub fn new(action: A, visits: usize, propagatedValues: PV) -> Self {
        Self {
            action,
            propagatedValues,
            visits,
        }
    }

    pub fn action(&self) -> &A {
        &self.action
    }

    pub fn visits(&self) -> usize {
        self.visits
    }

    pub fn propagatedValues(&self) -> &PV {
        &self.propagatedValues
    }
}

impl<A, P> Serialize for NodeMetrics<A, P>
where
    A: Serialize,
    P: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tup = serializer.serialize_tuple(3)?;
        tup.serialize_element(&self.visits)?;
        tup.serialize_element(&self.predictions)?;
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

impl<'de, A, P> Visitor<'de> for NodeMetricsVisitor<A, P>
where
    A: Deserialize<'de>,
    P: Deserialize<'de>,
{
    type Value = NodeMetrics<A, P>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("NodeMetrics")
    }

    fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        Ok(NodeMetrics {
            visits: seq.next_element()?.unwrap(),
            predictions: seq.next_element()?.unwrap(),
            children: seq.next_element()?.unwrap(),
        })
    }
}

impl<'de, A, P> Deserialize<'de> for NodeMetrics<A, P>
where
    A: Deserialize<'de>,
    P: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_tuple(3, NodeMetricsVisitor::new())
    }
}

impl<A, PV> Serialize for EdgeMetrics<A, PV>
where
    A: Serialize,
    PV: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tup = serializer.serialize_tuple(3)?;

        tup.serialize_element(&self.action)?;
        tup.serialize_element(&self.propagatedValues)?;
        tup.serialize_element(&self.visits)?;

        tup.end()
    }
}

struct NodeChildMetricsVisitor<A, PV> {
    marker: PhantomData<(A, PV)>,
}

impl<A, PV> NodeChildMetricsVisitor<A, PV> {
    fn new() -> Self {
        NodeChildMetricsVisitor {
            marker: PhantomData
        }
    }
}

impl<'de, A, PV> Visitor<'de> for NodeChildMetricsVisitor<A, PV>
where
    A: Deserialize<'de>,
    PV: Deserialize<'de>,
{
    type Value = EdgeMetrics<A, PV>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("NodeChildMetrics")
    }

    fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        Ok(EdgeMetrics {
            action: seq.next_element()?.unwrap(),
            propagatedValues: seq.next_element()?.unwrap(),
            visits: seq.next_element()?.unwrap(),
        })
    }
}

impl<'de, A, PV> Deserialize<'de> for EdgeMetrics<A, PV>
where
    A: Deserialize<'de>,
    PV: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_tuple(3, NodeChildMetricsVisitor::new())
    }
}
