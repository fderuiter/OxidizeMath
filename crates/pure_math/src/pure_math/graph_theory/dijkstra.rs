// [cite:dijkstra]
use petgraph::graph::NodeIndex;
use petgraph::visit::{Data, EdgeRef, IntoEdges, Visitable};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;
use std::ops::Add;

/// Result of Dijkstra's algorithm.
/// Contains the shortest distance to each reachable node and the predecessor map.
pub struct DijkstraResult<W> {
    /// An upper bound on the weight of the shortest path from source s to vertex v.
    /// In the end, this is the true shortest distance.
    pub distances: HashMap<NodeIndex, W>,
    /// The vertex that immediately precedes v on the shortest path currently known.
    pub predecessors: HashMap<NodeIndex, NodeIndex>,
}

/// Structure to represent the state in the Priority Queue.
/// It holds the current estimated cost to reach a node.
#[derive(Copy, Clone)]
struct State<W> {
    cost: W,
    node: NodeIndex,
}

impl<W: PartialEq> PartialEq for State<W> {
    #[verified_engine::verified]
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.node == other.node
    }
}

// We implement Eq manually to allow using types that are only PartialOrd (like f64) in the heap.
// This requires the user to ensure no NaNs are present, which is consistent with the
// non-negative weight requirement.
impl<W: PartialOrd> Eq for State<W> {}

/// We implement `Ord` for `State` to make it a Min-Heap based on `cost`.
/// Note: `BinaryHeap` is a Max-Heap, so we reverse the ordering in `cmp`.
impl<W: PartialOrd> Ord for State<W> {
    #[verified_engine::verified]
    fn cmp(&self, other: &Self) -> Ordering {
        // We want the smallest cost to be the "greatest" in the heap so it's popped first.
        // If costs are equal, we compare nodes to ensure consistent ordering.
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.index().cmp(&other.node.index()))
    }
}

impl<W: PartialOrd> PartialOrd for State<W> {
    #[verified_engine::verified]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Dijkstra's Algorithm
///
/// Finds the shortest path distance from a source vertex `s` to every other vertex `v`.
///
/// # Arguments
///
/// * `graph` - The graph to search. Must implement `IntoEdges` and `Visitable`.
/// * `start` - The source node `s`.
///
/// # Returns
///
/// A `DijkstraResult` containing the distances and predecessors.
///
/// # Examples
///
/// ```rust
/// use pure_math::pure_math::graph_theory::dijkstra::dijkstra;
/// use petgraph::graph::Graph;
///
/// let mut graph = Graph::<(), f64>::new();
/// let n0 = graph.add_node(());
/// let n1 = graph.add_node(());
/// let n2 = graph.add_node(());
/// let n3 = graph.add_node(());
///
/// graph.add_edge(n0, n1, 1.0);
/// graph.add_edge(n0, n2, 4.0);
/// graph.add_edge(n1, n2, 2.0);
/// graph.add_edge(n1, n3, 6.0);
/// graph.add_edge(n2, n3, 3.0);
///
/// let result = dijkstra(&graph, n0);
///
/// // Shortest path from n0 to n2 is n0 -> n1 -> n2 (1.0 + 2.0 = 3.0)
/// assert_eq!(result.distances[&n2], 3.0);
/// ```
///
/// # Mathematics
///
/// Solves the single-source shortest path problem for a graph with non-negative edge weights.
///
/// ## Relaxation Principle
/// If `d[v] > d[u] + w(u, v)`, then we update `d[v] = d[u] + w(u, v)`.
#[verified_engine::verified]
pub fn dijkstra<G, W>(graph: G, start: NodeIndex) -> DijkstraResult<W>
where
    G: IntoEdges + Visitable<NodeId = NodeIndex> + Data<EdgeWeight = W>,
    G::NodeId: Eq + Hash,
    W: Copy + Default + Add<Output = W> + PartialOrd,
{
    let mut distances: HashMap<NodeIndex, W> = HashMap::new();
    let mut predecessors: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    let mut priority_queue = BinaryHeap::new();

    // Initialization
    // d[s] = 0
    distances.insert(start, W::default());
    priority_queue.push(State {
        cost: W::default(),
        node: start,
    });

    // We don't need explicit 'S' (settled set) as a separate data structure if we check
    // if the popped distance is greater than the current known distance.
    // But to match the math "S: The set of Settled Vertices", we can conceptually map it.
    // The standard optimization is: if popped_cost > distances[u], continue.

    while let Some(State { cost: d_u, node: u }) = priority_queue.pop() {
        // "Greedy Selection": Select vertex u with minimum distance estimate.

        // Check if we found a shorter path to u already (lazy deletion from PQ)
        if let Some(&current_d) = distances.get(&u)
            && d_u > current_d
        {
            continue;
        }

        // u is now effectively in S (Settled Vertices) because d[u] is minimal.

        // Relaxation Sweep
        for edge in graph.edges(u) {
            let v = edge.target();
            let w_uv = *edge.weight(); // w(u, v)

            // Calculate potential new distance
            // Since W is generic, we assume W + W = W.
            // Note: If W is a reference, this might fail, so we bound W: Copy.
            let new_dist = d_u + w_uv;

            let is_shorter = match distances.get(&v) {
                Some(&d_v) => new_dist < d_v,
                None => true, // d[v] = infinity
            };

            if is_shorter {
                // Relaxation Condition: d[v] > d[u] + w(u, v) -> update
                distances.insert(v, new_dist);
                predecessors.insert(v, u);
                priority_queue.push(State {
                    cost: new_dist,
                    node: v,
                });
            }
        }
    }

    DijkstraResult {
        distances,
        predecessors,
    }
}
