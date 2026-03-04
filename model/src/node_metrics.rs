use common::{GameLength, PlayerValue};
use serde::de::SeqAccess;
use serde::de::{Deserialize, Deserializer, Visitor};
use serde::ser::{Serialize, SerializeTuple, Serializer};
use std::fmt;
use std::marker::PhantomData;

#[allow(non_snake_case)]
#[derive(PartialEq, Debug)]
pub struct NodeMetrics<A, P, SS> {
    /// The total number of visits of the node. Should be children.visits.sum() + 1.
    visits: usize,
    /// Ancillary predictions by the neural network. Like score difference or moves left.
    predictions: P,
    /// The valid actions of the current game_state of the node.
    children: Vec<EdgeMetrics<A, SS>>,
}

#[allow(non_snake_case)]
impl<A, P, SS> NodeMetrics<A, P, SS> {
    pub fn new(predictions: P, visits: usize, children: Vec<EdgeMetrics<A, SS>>) -> Self {
        Self {
            visits,
            predictions,
            children,
        }
    }

    pub fn child_max_visits(&self) -> &EdgeMetrics<A, SS> {
        self.children.iter().max_by_key(|c| c.visits).unwrap()
    }

    pub fn children(&self) -> &Vec<EdgeMetrics<A, SS>> {
        &self.children
    }

    pub fn visits(&self) -> usize {
        self.visits
    }

    pub fn predictions(&self) -> &P {
        &self.predictions
    }
}

#[allow(non_snake_case)]
#[derive(PartialEq, Debug)]
pub struct EdgeMetrics<A, SS> {
    /// The action that this edge represents.
    action: A,
    /// The number of visits for the child node of this specific edge.
    visits: usize,
    /// Predictions by the neural network that have been propagated from child to parent nodes. Like score difference or moves left.
    snapshot: SS,
}

#[allow(non_snake_case)]
impl<A, SS> EdgeMetrics<A, SS> {
    pub fn new(action: A, visits: usize, snapshot: SS) -> Self {
        Self {
            action,
            snapshot,
            visits,
        }
    }

    pub fn action(&self) -> &A {
        &self.action
    }

    pub fn visits(&self) -> usize {
        self.visits
    }

    pub fn snapshot(&self) -> &SS {
        &self.snapshot
    }
}

#[allow(non_snake_case)]
impl<A, SS> EdgeMetrics<A, SS>
where
    SS: PlayerValue,
{
    pub fn player_value(&self, player: usize) -> f32 {
        self.snapshot().player_value(player)
    }
}

#[allow(non_snake_case)]
impl<A, SS> EdgeMetrics<A, SS>
where
    SS: GameLength,
{
    pub fn game_length(&self) -> f32 {
        self.snapshot().game_length()
    }
}

impl<A, P, SS> Serialize for NodeMetrics<A, P, SS>
where
    A: Serialize,
    P: Serialize,
    SS: Serialize,
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

struct NodeMetricsVisitor<A, P, SS> {
    marker: PhantomData<(A, P, SS)>,
}

impl<A, P, SS> NodeMetricsVisitor<A, P, SS> {
    fn new() -> Self {
        NodeMetricsVisitor {
            marker: PhantomData,
        }
    }
}

impl<'de, A, P, SS> Visitor<'de> for NodeMetricsVisitor<A, P, SS>
where
    A: Deserialize<'de>,
    P: Deserialize<'de>,
    SS: Deserialize<'de>,
{
    type Value = NodeMetrics<A, P, SS>;

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

impl<'de, A, P, SS> Deserialize<'de> for NodeMetrics<A, P, SS>
where
    A: Deserialize<'de>,
    P: Deserialize<'de>,
    SS: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_tuple(3, NodeMetricsVisitor::new())
    }
}

impl<A, SS> Serialize for EdgeMetrics<A, SS>
where
    A: Serialize,
    SS: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tup = serializer.serialize_tuple(3)?;

        tup.serialize_element(&self.action)?;
        tup.serialize_element(&self.snapshot)?;
        tup.serialize_element(&self.visits)?;

        tup.end()
    }
}

struct NodeChildMetricsVisitor<A, SS> {
    marker: PhantomData<(A, SS)>,
}

impl<A, SS> NodeChildMetricsVisitor<A, SS> {
    fn new() -> Self {
        NodeChildMetricsVisitor {
            marker: PhantomData,
        }
    }
}

impl<'de, A, SS> Visitor<'de> for NodeChildMetricsVisitor<A, SS>
where
    A: Deserialize<'de>,
    SS: Deserialize<'de>,
{
    type Value = EdgeMetrics<A, SS>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("NodeChildMetrics")
    }

    fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        Ok(EdgeMetrics {
            action: seq.next_element()?.unwrap(),
            snapshot: seq.next_element()?.unwrap(),
            visits: seq.next_element()?.unwrap(),
        })
    }
}

impl<'de, A, SS> Deserialize<'de> for EdgeMetrics<A, SS>
where
    A: Deserialize<'de>,
    SS: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_tuple(3, NodeChildMetricsVisitor::new())
    }
}
