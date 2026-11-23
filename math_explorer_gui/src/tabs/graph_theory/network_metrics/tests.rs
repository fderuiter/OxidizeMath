#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_network_metrics_recomputes_strictly_on_revision_advance() {
    let mut tool = NetworkMetricsTool::default();
    let initial_recomputes = tool.recompute_count();

    let mut ch = GraphTopologyChannel::default();
    // Path graph 0 - 1 - 2 - 3 has diameter 3 and clustering 0.0.
    ch.publish(
        &[[-1.0, 0.0], [-0.3, 0.0], [0.3, 0.0], [1.0, 0.0]],
        &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)],
    );
    tool.ingest_topology_channel(&ch);
    assert_eq!(tool.current_diameter(), Some(3));
    assert_eq!(tool.current_average_clustering(), Some(0.0));
    assert_eq!(tool.recompute_count(), initial_recomputes + 1);

    // Same revision: zero redundant recomputation.
    tool.ingest_topology_channel(&ch);
    assert_eq!(tool.recompute_count(), initial_recomputes + 1);

    // Add shortcut edge (0, 3): diameter drops from 3 to 2 on revision bump.
    ch.publish(
        &[[-1.0, 0.0], [-0.3, 0.0], [0.3, 0.0], [1.0, 0.0]],
        &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 3, 1.0)],
    );
    tool.ingest_topology_channel(&ch);
    assert_eq!(tool.current_diameter(), Some(2));
    assert_eq!(tool.recompute_count(), initial_recomputes + 2);
}
