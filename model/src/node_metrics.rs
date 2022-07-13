use serde::de::SeqAccess;
use serde::de::{Deserialize, Deserializer, Visitor};
use serde::ser::{Serialize, SerializeTuple, Serializer};
use std::fmt;
use std::marker::PhantomData;

#[allow(non_snake_case)]
#[derive(PartialEq, Debug)]
pub struct NodeMetrics<A, V> {
    /// The total number of visits of the node. Should be children.visits.sum() + 1.
    pub visits: usize,
    /// The value of the position for the specific game_state of the node as predicted by the neural network
    pub value: V,
    /// The number of moves left for the specific game_state of the node as predicted by the neural network
    pub moves_left: f32,
    /// The valid actions of the current game_state of the node.
    pub children: Vec<NodeChildMetrics<A>>,
}

#[allow(non_snake_case)]
impl<A, V> NodeMetrics<A, V> {
    /// Difference between the Q of the specified action and the child with the highest Q.
    pub fn Q_diff(&self, action: &A) -> f32
    where
        A: PartialEq,
    {
        let max_Q = self.children.iter().map(|c| c.Q()).fold(f32::MIN, f32::max);
        let act_Q = self.children.iter().find(|c| c.action() == action);
        let act_Q = act_Q.expect("Specified action was not found").Q();
        max_Q - act_Q
    }

    pub fn child_max_visits(&self) -> &NodeChildMetrics<A> {
        self.children.iter().max_by_key(|c| c.visits).unwrap()
    }
}

#[allow(non_snake_case)]
#[derive(PartialEq, Debug)]
pub struct NodeChildMetrics<A> {
    /// The action that this edge represents.
    action: A,
    /// The Q score of the edge. This is an average of Q back propagated by the descendant nodes. Range is 0.0..=1.0. Q is from the perspective of the player to move of the parent node of this edge.
    Q: f32,
    /// The M score of the edge. This is an average of M back propagated by the descendant nodes.
    M: f32,
    /// The number of visits for the child node of this specific edge.
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

    pub fn M(&self) -> f32 {
        self.M
    }

    pub fn Q(&self) -> f32 {
        self.Q
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
        let mut tup = serializer.serialize_tuple(4)?;
        tup.serialize_element(&self.visits)?;
        tup.serialize_element(&self.value)?;
        tup.serialize_element(&self.moves_left)?;
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
            moves_left: seq.next_element()?.unwrap(),
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
        deserializer.deserialize_tuple(4, NodeMetricsVisitor::new())
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
