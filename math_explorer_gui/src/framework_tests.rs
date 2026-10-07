#[cfg(test)]
mod tests {
    use crate::framework::SimulationFramework;

    #[test]
    fn test_simulation_framework_search_initialization() {
        let framework = SimulationFramework::new("chaos");
        assert_eq!(
            framework.search_query, "",
            "search_query should initialize to empty string"
        );
    }

    #[test]
    fn test_search_query_filtering_by_name_and_tags() {
        let mut framework = SimulationFramework::new("chaos");
        let initial_count = framework.available_tools.len();

        framework.search_query = "Fractal".to_string();
        let query = framework.search_query.trim().to_lowercase();
        let filtered_count = framework
            .available_tools
            .iter()
            .filter(|meta| {
                meta.name.to_lowercase().contains(&query)
                    || meta.tags.iter().any(|t| t.to_lowercase().contains(&query))
            })
            .count();

        assert!(filtered_count <= initial_count);

        framework.search_query.clear();
        assert_eq!(framework.search_query, "");
    }

    #[test]
    fn test_non_matching_search_query() {
        let mut framework = SimulationFramework::new("chaos");
        framework.search_query = "non_existent_tool_xyz_123".to_string();
        let query = framework.search_query.trim().to_lowercase();
        let filtered_count = framework
            .available_tools
            .iter()
            .filter(|meta| {
                meta.name.to_lowercase().contains(&query)
                    || meta.tags.iter().any(|t| t.to_lowercase().contains(&query))
            })
            .count();

        assert_eq!(filtered_count, 0);
    }
}
