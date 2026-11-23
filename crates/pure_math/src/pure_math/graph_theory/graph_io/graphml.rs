//! Deterministic Gephi/Cytoscape `.graphml` serializer and DTD-hardened pure-Rust parser.

use super::escape::{
    edge_style_to_str, ensure_node_index, escape_xml, format_compact_f64, shape_to_str,
    str_to_edge_style, str_to_shape, unescape_xml,
};
use super::{
    GraphTopologyDocument, MAX_TOPOLOGY_EDGES, TopologyCodecError, TopologyEdge, TopologyEdgeStyle,
    TopologyNode, TopologyNodeShape, check_import_byte_limit, strip_utf8_bom_str,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;

/// Serializes a [`GraphTopologyDocument`] deterministically into Gephi/Cytoscape `.graphml` XML.
#[must_use]
#[verified_engine::verified]
pub fn to_graphml_topology(doc: &GraphTopologyDocument) -> String {
    let mut out = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">\n",
    );
    write_builtin_keys(&mut out);
    let node_attr_keys = collect_custom_keys(doc.nodes.iter().map(|n| &n.attributes));
    let edge_attr_keys = collect_custom_keys(doc.edges.iter().map(|e| &e.attributes));
    let graph_attr_keys: Vec<&String> = doc.attributes.keys().collect();
    for (i, k) in graph_attr_keys.iter().enumerate() {
        write_key_decl(&mut out, &format!("gk{i}"), "graph", k);
    }
    for (i, k) in node_attr_keys.iter().enumerate() {
        write_key_decl(&mut out, &format!("nk{i}"), "node", k);
    }
    for (i, k) in edge_attr_keys.iter().enumerate() {
        write_key_decl(&mut out, &format!("ek{i}"), "edge", k);
    }
    let edgedef = if doc.directed {
        "directed"
    } else {
        "undirected"
    };
    let gid = if doc.name.trim().is_empty() {
        "G"
    } else {
        doc.name.trim()
    };
    let _ = writeln!(
        out,
        "  <graph id=\"{}\" edgedefault=\"{edgedef}\">",
        escape_xml(gid)
    );
    for (i, k) in graph_attr_keys.iter().enumerate() {
        if let Some(v) = doc.attributes.get(*k) {
            let _ = writeln!(out, "    <data key=\"gk{i}\">{}</data>", escape_xml(v));
        }
    }
    for node in &doc.nodes {
        write_graphml_node(&mut out, node, &node_attr_keys);
    }
    for (eidx, edge) in doc.edges.iter().enumerate() {
        write_graphml_edge(&mut out, eidx, edge, &edge_attr_keys);
    }
    out.push_str("  </graph>\n</graphml>\n");
    out
}

#[verified_engine::verified]
fn write_builtin_keys(out: &mut String) {
    out.push_str(
        "  <key id=\"d_label\" for=\"node\" attr.name=\"label\" attr.type=\"string\"/>\n\
         <key id=\"d_x\" for=\"node\" attr.name=\"x\" attr.type=\"double\"/>\n\
         <key id=\"d_y\" for=\"node\" attr.name=\"y\" attr.type=\"double\"/>\n\
         <key id=\"d_fill\" for=\"node\" attr.name=\"fill\" attr.type=\"string\"/>\n\
         <key id=\"d_stroke\" for=\"node\" attr.name=\"stroke\" attr.type=\"string\"/>\n\
         <key id=\"d_shape\" for=\"node\" attr.name=\"shape\" attr.type=\"string\"/>\n\
         <key id=\"e_weight\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>\n\
         <key id=\"e_label\" for=\"edge\" attr.name=\"label\" attr.type=\"string\"/>\n\
         <key id=\"e_style\" for=\"edge\" attr.name=\"style\" attr.type=\"string\"/>\n\
         <key id=\"e_color\" for=\"edge\" attr.name=\"color\" attr.type=\"string\"/>\n",
    );
}

#[verified_engine::verified]
fn collect_custom_keys<'a>(
    maps: impl Iterator<Item = &'a BTreeMap<String, String>>,
) -> Vec<String> {
    let mut set = BTreeSet::new();
    for map in maps {
        for k in map.keys() {
            set.insert(k.clone());
        }
    }
    set.into_iter().collect()
}

#[verified_engine::verified]
fn write_key_decl(out: &mut String, id: &str, domain: &str, name: &str) {
    let _ = writeln!(
        out,
        "  <key id=\"{id}\" for=\"{domain}\" attr.name=\"{}\" attr.type=\"string\"/>",
        escape_xml(name)
    );
}

#[verified_engine::verified]
fn write_data_tag(out: &mut String, key: &str, raw_val: &str) {
    let _ = writeln!(out, "      <data key=\"{key}\">{raw_val}</data>");
}

#[verified_engine::verified]
fn write_graphml_node(out: &mut String, node: &TopologyNode, custom_keys: &[String]) {
    let _ = writeln!(out, "    <node id=\"{}\">", escape_xml(&node.id));
    write_data_tag(out, "d_label", &escape_xml(&node.label));
    if let Some([x, y]) = node.position {
        write_data_tag(out, "d_x", &format_compact_f64(x));
        write_data_tag(out, "d_y", &format_compact_f64(y));
    }
    if let Some(fill) = &node.fill_hex {
        write_data_tag(out, "d_fill", &escape_xml(fill));
    }
    if let Some(stroke) = &node.stroke_hex {
        write_data_tag(out, "d_stroke", &escape_xml(stroke));
    }
    if node.shape != TopologyNodeShape::Circle {
        write_data_tag(out, "d_shape", shape_to_str(node.shape));
    }
    for (i, k) in custom_keys.iter().enumerate() {
        if let Some(val) = node.attributes.get(k) {
            write_data_tag(out, &format!("nk{i}"), &escape_xml(val));
        }
    }
    out.push_str("    </node>\n");
}

#[verified_engine::verified]
fn write_graphml_edge(out: &mut String, eidx: usize, edge: &TopologyEdge, custom_keys: &[String]) {
    let _ = writeln!(
        out,
        "    <edge id=\"e{eidx}\" source=\"{}\" target=\"{}\">",
        escape_xml(&edge.source),
        escape_xml(&edge.target)
    );
    write_data_tag(out, "e_weight", &format_compact_f64(edge.weight));
    if let Some(lbl) = &edge.label {
        write_data_tag(out, "e_label", &escape_xml(lbl));
    }
    if edge.style != TopologyEdgeStyle::Solid {
        write_data_tag(out, "e_style", edge_style_to_str(edge.style));
    }
    if let Some(color) = &edge.color_hex {
        write_data_tag(out, "e_color", &escape_xml(color));
    }
    for (i, k) in custom_keys.iter().enumerate() {
        if let Some(val) = edge.attributes.get(k) {
            write_data_tag(out, &format!("ek{i}"), &escape_xml(val));
        }
    }
    out.push_str("    </edge>\n");
}

#[derive(Default)]
struct GraphMlScanState {
    saw_graph_tag: bool,
    key_names: HashMap<String, String>,
    index_map: HashMap<String, usize>,
    active_node: Option<usize>,
    active_edge: Option<usize>,
    pending_node_xy: HashMap<usize, (Option<f64>, Option<f64>)>,
}

/// Parses a `.graphml` XML document while rejecting `<!DOCTYPE` and `<!ENTITY` declarations.
#[verified_engine::verified]
pub fn parse_graphml_topology(raw: &str) -> Result<GraphTopologyDocument, TopologyCodecError> {
    let cleaned = strip_utf8_bom_str(raw);
    check_import_byte_limit(cleaned)?;
    reject_forbidden_xml_declarations(cleaned)?;

    let mut doc = GraphTopologyDocument::new("G", false);
    let mut state = GraphMlScanState::default();

    let mut cursor = 0usize;
    while let Some(rel_lt) = cleaned[cursor..].find('<') {
        let lt = cursor + rel_lt;
        if cleaned[lt..].starts_with("<!--") {
            let end = cleaned[lt + 4..].find("-->").ok_or_else(|| {
                TopologyCodecError::Parse("Unterminated XML comment in GraphML".into())
            })?;
            cursor = lt + 4 + end + 3;
            continue;
        }
        if cleaned[lt..].starts_with("<?") {
            let end = cleaned[lt + 2..].find("?>").ok_or_else(|| {
                TopologyCodecError::Parse("Unterminated XML processing instruction".into())
            })?;
            cursor = lt + 2 + end + 2;
            continue;
        }
        let rel_gt = cleaned[lt + 1..]
            .find('>')
            .ok_or_else(|| TopologyCodecError::Parse("Unterminated XML tag in GraphML".into()))?;
        let tag_body = cleaned[lt + 1..lt + 1 + rel_gt].trim();
        cursor = lt + 1 + rel_gt + 1;
        process_graphml_tag(cleaned, &mut cursor, tag_body, &mut doc, &mut state)?;
    }

    for (idx, (mx, my)) in state.pending_node_xy {
        if let (Some(x), Some(y)) = (mx, my)
            && x.is_finite()
            && y.is_finite()
        {
            doc.nodes[idx].position = Some([x, y]);
        }
    }
    if !state.saw_graph_tag || doc.nodes.is_empty() {
        return Err(TopologyCodecError::EmptyTopology(
            "GraphML document contains no <graph> with valid <node> elements".into(),
        ));
    }
    doc.validate()?;
    Ok(doc)
}

#[verified_engine::verified]
fn reject_forbidden_xml_declarations(src: &str) -> Result<(), TopologyCodecError> {
    let bytes = src.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b'<' && bytes.get(i + 1) == Some(&b'!') {
            let tail = &src[i + 2..];
            let trimmed = tail.trim_start();
            let head: String = trimmed.chars().take(7).collect();
            let upper = head.to_ascii_uppercase();
            if upper.starts_with("DOCTYPE") || upper.starts_with("ENTITY") {
                return Err(TopologyCodecError::ForbiddenXmlDeclaration);
            }
        }
    }
    Ok(())
}

#[verified_engine::verified]
fn process_graphml_tag(
    src: &str,
    cursor: &mut usize,
    tag_body: &str,
    doc: &mut GraphTopologyDocument,
    st: &mut GraphMlScanState,
) -> Result<(), TopologyCodecError> {
    if let Some(closing) = tag_body.strip_prefix('/') {
        let cname = local_xml_name(closing.trim());
        if cname.eq_ignore_ascii_case("node") {
            st.active_node = None;
        } else if cname.eq_ignore_ascii_case("edge") {
            st.active_edge = None;
        }
        return Ok(());
    }
    let self_closing = tag_body.ends_with('/');
    let core = tag_body.trim_end_matches('/').trim();
    let (raw_name, attr_str) = split_tag_name_and_attrs(core);
    let name = local_xml_name(raw_name);
    let attrs = parse_xml_attributes(attr_str);

    if name.eq_ignore_ascii_case("key") {
        if let Some(id) = attrs.get("id") {
            let attr_name = attrs
                .get("attr.name")
                .cloned()
                .unwrap_or_else(|| id.clone());
            st.key_names.insert(id.clone(), attr_name);
        }
    } else if name.eq_ignore_ascii_case("graph") {
        st.saw_graph_tag = true;
        if let Some(gid) = attrs.get("id") {
            doc.name.clone_from(gid);
        }
        if let Some(ed) = attrs.get("edgedefault") {
            doc.directed = ed.eq_ignore_ascii_case("directed");
        }
    } else if name.eq_ignore_ascii_case("node") {
        let id = attrs.get("id").ok_or_else(|| {
            TopologyCodecError::Parse("GraphML <node> missing 'id' attribute".into())
        })?;
        let idx = ensure_node_index(&mut doc.nodes, &mut st.index_map, id)?;
        st.active_node = if self_closing { None } else { Some(idx) };
    } else if name.eq_ignore_ascii_case("edge") {
        let eidx = insert_graphml_edge(doc, &mut st.index_map, &attrs)?;
        st.active_edge = if self_closing { None } else { Some(eidx) };
    } else if name.eq_ignore_ascii_case("data") && !self_closing {
        let text = read_until_closing_data(src, cursor)?;
        if let Some(key_id) = attrs.get("key") {
            let resolved = st
                .key_names
                .get(key_id)
                .map_or(key_id.as_str(), String::as_str);
            apply_graphml_data(
                doc,
                st.active_node,
                st.active_edge,
                &mut st.pending_node_xy,
                resolved,
                &text,
            )?;
        }
    }
    Ok(())
}

#[verified_engine::verified]
fn insert_graphml_edge(
    doc: &mut GraphTopologyDocument,
    index_map: &mut HashMap<String, usize>,
    attrs: &HashMap<String, String>,
) -> Result<usize, TopologyCodecError> {
    if doc.edges.len() >= MAX_TOPOLOGY_EDGES {
        return Err(TopologyCodecError::TooManyEdges {
            actual: doc.edges.len() + 1,
            max: MAX_TOPOLOGY_EDGES,
        });
    }
    let source = attrs.get("source").ok_or_else(|| {
        TopologyCodecError::Parse("GraphML <edge> missing 'source' attribute".into())
    })?;
    let target = attrs.get("target").ok_or_else(|| {
        TopologyCodecError::Parse("GraphML <edge> missing 'target' attribute".into())
    })?;
    ensure_node_index(&mut doc.nodes, index_map, source)?;
    ensure_node_index(&mut doc.nodes, index_map, target)?;
    let eidx = doc.edges.len();
    doc.edges.push(TopologyEdge::new(source, target, 1.0));
    Ok(eidx)
}

#[verified_engine::verified]
fn apply_graphml_data(
    doc: &mut GraphTopologyDocument,
    active_node: Option<usize>,
    active_edge: Option<usize>,
    pending_node_xy: &mut HashMap<usize, (Option<f64>, Option<f64>)>,
    attr_name: &str,
    value: &str,
) -> Result<(), TopologyCodecError> {
    if let Some(nidx) = active_node {
        let node = &mut doc.nodes[nidx];
        match attr_name {
            "label" => node.label = value.to_string(),
            "x" => {
                if let Ok(v) = value.trim().parse::<f64>() {
                    pending_node_xy.entry(nidx).or_insert((None, None)).0 = Some(v);
                }
            }
            "y" => {
                if let Ok(v) = value.trim().parse::<f64>() {
                    pending_node_xy.entry(nidx).or_insert((None, None)).1 = Some(v);
                }
            }
            "fill" | "color" if node.fill_hex.is_none() => {
                node.fill_hex = Some(value.to_string());
            }
            "stroke" => node.stroke_hex = Some(value.to_string()),
            "shape" => node.shape = str_to_shape(value),
            other => {
                node.attributes.insert(other.to_string(), value.to_string());
            }
        }
    } else if let Some(eidx) = active_edge {
        let edge = &mut doc.edges[eidx];
        match attr_name {
            "weight" => {
                let w = value.trim().parse::<f64>().map_err(|_| {
                    TopologyCodecError::Parse(format!("Invalid GraphML edge weight '{value}'"))
                })?;
                if !w.is_finite() {
                    return Err(TopologyCodecError::Parse(
                        "Non-finite GraphML edge weight".into(),
                    ));
                }
                edge.weight = w;
            }
            "label" => edge.label = Some(value.to_string()),
            "style" => edge.style = str_to_edge_style(value),
            "color" => edge.color_hex = Some(value.to_string()),
            other => {
                edge.attributes.insert(other.to_string(), value.to_string());
            }
        }
    } else {
        doc.attributes
            .insert(attr_name.to_string(), value.to_string());
    }
    Ok(())
}

#[verified_engine::verified]
fn read_until_closing_data(src: &str, cursor: &mut usize) -> Result<String, TopologyCodecError> {
    let rest = &src[*cursor..];
    let mut search = 0usize;
    while let Some(rel) = rest[search..].find("</") {
        let idx = search + rel;
        let after = &rest[idx + 2..];
        if let Some(gt_rel) = after.find('>') {
            let cname = local_xml_name(after[..gt_rel].trim());
            if cname.eq_ignore_ascii_case("data") {
                let raw_text = &rest[..idx];
                *cursor += idx + 2 + gt_rel + 1;
                return Ok(unescape_xml(raw_text));
            }
            search = idx + 2 + gt_rel + 1;
        } else {
            break;
        }
    }
    Err(TopologyCodecError::Parse(
        "Unterminated <data> element in GraphML".into(),
    ))
}

#[verified_engine::verified]
fn local_xml_name(raw: &str) -> &str {
    raw.rsplit(':').next().unwrap_or(raw)
}

#[verified_engine::verified]
fn split_tag_name_and_attrs(core: &str) -> (&str, &str) {
    for (idx, ch) in core.char_indices() {
        if ch.is_whitespace() {
            return (&core[..idx], core[idx..].trim());
        }
    }
    (core, "")
}

#[verified_engine::verified]
fn parse_xml_attributes(raw: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let bytes = raw.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let key_start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'=' {
            i += 1;
        }
        if i == key_start {
            break;
        }
        let key = local_xml_name(&raw[key_start..i]).to_string();
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if bytes.get(i) != Some(&b'=') {
            continue;
        }
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if let Some(&quote) = bytes.get(i)
            && (quote == b'"' || quote == b'\'')
        {
            i += 1;
            let val_start = i;
            while i < bytes.len() && bytes[i] != quote {
                i += 1;
            }
            let val = unescape_xml(&raw[val_start..i]);
            map.insert(key, val);
            if i < bytes.len() {
                i += 1;
            }
        }
    }
    map
}

// [cite:graph_parameters_rust]
