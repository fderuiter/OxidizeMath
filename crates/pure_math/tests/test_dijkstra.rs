//! Integration tests for test dijkstra.
#[cfg(test)]
mod tests {
    use petgraph::graph::Graph;
    use petgraph::{Directed, Undirected};
    use pure_math::pure_math::graph_theory::dijkstra::dijkstra;

    #[test]
    #[verified_engine::verified]
    fn test_dijkstra_directed_simple() {
        // Create a simple directed graph
        // (0) --10--> (1)
        // | \          ^
        // |  5         |
        // v   \        |
        // (2) --2--> (3) --1--> (1)
        //            |
        //            4
        //            v
        //            (4)

        let mut graph = Graph::<(), i32, Directed>::new();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        let n4 = graph.add_node(());

        graph.add_edge(n0, n1, 10);
        graph.add_edge(n0, n2, 5);
        graph.add_edge(n2, n3, 2);
        graph.add_edge(n3, n1, 1);
        graph.add_edge(n3, n4, 4);

        // Path 0->1: cost 10
        // Path 0->2->3->1: cost 5+2+1 = 8 (Shorter)
        // Path 0->2->3->4: cost 5+2+4 = 11

        let result = dijkstra(&graph, n0);

        assert_eq!(*result.distances.get(&n0).unwrap(), 0);
        assert_eq!(*result.distances.get(&n1).unwrap(), 8);
        assert_eq!(*result.distances.get(&n2).unwrap(), 5);
        assert_eq!(*result.distances.get(&n3).unwrap(), 7);
        assert_eq!(*result.distances.get(&n4).unwrap(), 11);

        // Check predecessor for n1, should be n3
        assert_eq!(*result.predecessors.get(&n1).unwrap(), n3);
        // Predecessor for n3 should be n2
        assert_eq!(*result.predecessors.get(&n3).unwrap(), n2);
        // Predecessor for n2 should be n0
        assert_eq!(*result.predecessors.get(&n2).unwrap(), n0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_dijkstra_undirected() {
        // Simple undirected graph
        // (0) --1-- (1) --2-- (2)
        //  |         |
        //  4         3
        //  |         |
        // (3) --1-- (4)

        let mut graph = Graph::<(), i32, Undirected>::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        let n4 = graph.add_node(());

        graph.add_edge(n0, n1, 1);
        graph.add_edge(n1, n2, 2);
        graph.add_edge(n0, n3, 4);
        graph.add_edge(n1, n4, 3);
        graph.add_edge(n3, n4, 1);

        // Shortest path to n4:
        // 0->1->4 = 1+3 = 4
        // 0->3->4 = 4+1 = 5
        // So dist(0, 4) = 4.

        // Shortest path to n3:
        // 0->3 = 4
        // 0->1->4->3 = 1+3+1 = 5
        // So dist(0, 3) = 4. Wait, 0->1->4 is 4, 4->3 is 1, so 5. 0->3 is 4. Correct.

        let result = dijkstra(&graph, n0);

        assert_eq!(*result.distances.get(&n4).unwrap(), 4);
        assert_eq!(*result.predecessors.get(&n4).unwrap(), n1);
    }

    #[test]
    #[verified_engine::verified]
    fn test_dijkstra_f64_weights() {
        // Graph with float weights
        // (0) -- 0.5 --> (1) -- 0.5 --> (2)
        // |
        // \-- 1.2 --> (2)

        let mut graph = Graph::<(), f64, Directed>::new();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, 0.5);
        graph.add_edge(n1, n2, 0.5);
        graph.add_edge(n0, n2, 1.2);

        // Path 0->1->2 is 1.0
        // Path 0->2 is 1.2

        let result = dijkstra(&graph, n0);

        assert!(
            (result.distances.get(&n2).unwrap() - 1.0).abs()
                < math_commons::registry::TOLERANCE_FAST
        );
        assert_eq!(*result.predecessors.get(&n2).unwrap(), n1);
    }
}
