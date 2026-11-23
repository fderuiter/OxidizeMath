#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Cross-platform `AccessKit` tree store and WAI-ARIA `DomNodeSpec` projection.

use eframe::egui::{
    self,
    accesskit::{Action, Live, Node, NodeId, Orientation, Role, Toggled, TreeUpdate},
};
use std::collections::{HashMap, HashSet};

/// Viewport-relative CSS bounding rectangle in CSS pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CssBounds {
    /// Left offset in CSS pixels.
    pub left: f32,
    /// Top offset in CSS pixels.
    pub top: f32,
    /// Width in CSS pixels.
    pub width: f32,
    /// Height in CSS pixels.
    pub height: f32,
}

/// Projected HTML tag and ARIA attribute specification for a single `AccessKit` node.
#[derive(Debug, Clone, PartialEq)]
pub struct DomNodeSpec {
    /// HTML element tag name (`"div"`, `"section"`, `"nav"`, `"button"`, `"input"`, etc.).
    pub tag: &'static str,
    /// Canonical lowercase `AccessKit` role slug (`"region"`, `"splitter"`, `"slider"`, etc.).
    pub data_ak_role: &'static str,
    /// Optional explicit WAI-ARIA `role` attribute.
    pub role: Option<&'static str>,
    /// Optional `<input type="...">` token.
    pub input_type: Option<&'static str>,
    /// Accessible label (`aria-label`).
    pub aria_label: Option<String>,
    /// Accessible description (`aria-description`).
    pub aria_description: Option<String>,
    /// Custom ARIA role description (`aria-roledescription`).
    pub aria_roledescription: Option<String>,
    /// Visible/accessible text content for leaf headings, buttons, and labels.
    pub text_content: Option<String>,
    /// String value for text inputs or `data-ak-value`.
    pub value: Option<String>,
    /// Placeholder string for text inputs.
    pub placeholder: Option<String>,
    /// Numeric value (`aria-valuenow`).
    pub numeric_now: Option<f64>,
    /// Minimum numeric value (`aria-valuemin` / `min`).
    pub numeric_min: Option<f64>,
    /// Maximum numeric value (`aria-valuemax` / `max`).
    pub numeric_max: Option<f64>,
    /// Numeric step (`step`).
    pub numeric_step: Option<f64>,
    /// Human-readable value text (`aria-valuetext`).
    pub aria_valuetext: Option<String>,
    /// Checked state (`"true"`, `"false"`, or `"mixed"`).
    pub aria_checked: Option<&'static str>,
    /// Pressed state (`"true"`, `"false"`, or `"mixed"`) for toggle buttons.
    pub aria_pressed: Option<&'static str>,
    /// Selected state (`"true"` or `"false"`).
    pub aria_selected: Option<&'static str>,
    /// Expanded state (`"true"` or `"false"`).
    pub aria_expanded: Option<&'static str>,
    /// Orientation (`"horizontal"` or `"vertical"`).
    pub aria_orientation: Option<&'static str>,
    /// Heading level (`aria-level`).
    pub aria_level: Option<usize>,
    /// Live politeness (`"polite"` or `"assertive"`).
    pub aria_live: Option<&'static str>,
    /// Modal dialog state (`aria-modal="true"`).
    pub aria_modal: bool,
    /// Disabled control state.
    pub disabled: bool,
    /// Keeps `aria-disabled="true"` without HTML `disabled` so keyboard users can focus disabled-reason tooltips.
    pub focusable_when_disabled: bool,
    /// Read-only control state.
    pub read_only: bool,
    /// Explicit `tabindex` attribute (`Some(0)` for focusable non-native elements).
    pub tab_index: Option<i32>,
    /// Ordered visible child `NodeId`s.
    pub children: Vec<NodeId>,
    /// Viewport CSS bounds over `#the_canvas_id`.
    pub bounds: Option<CssBounds>,
}

/// Incremental store retaining the latest merged `accesskit::TreeUpdate` nodes and root.
#[derive(Debug, Clone, Default)]
pub struct AccessKitTreeStore {
    nodes: HashMap<NodeId, Node>,
    root_id: Option<NodeId>,
    focus_id: Option<NodeId>,
}

impl AccessKitTreeStore {
    /// Creates an empty [`AccessKitTreeStore`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears all retained nodes, root, and focus state (invoked on `WebRunner::destroy()`).
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.root_id = None;
        self.focus_id = None;
    }

    /// Applies an `accesskit::TreeUpdate` and prunes unreachable orphan nodes.
    pub fn apply_update(&mut self, update: &TreeUpdate) {
        if let Some(tree) = &update.tree {
            self.root_id = Some(tree.root);
        }
        self.focus_id = Some(update.focus);
        for (id, node) in &update.nodes {
            self.nodes.insert(*id, node.clone());
        }
        self.prune_unreachable();
    }

    /// Returns the root `NodeId` if a tree has been initialized.
    #[must_use]
    pub fn root_id(&self) -> Option<NodeId> {
        self.root_id
    }

    /// Returns the currently focused `NodeId` from the latest `TreeUpdate`.
    #[must_use]
    pub fn focus_id(&self) -> Option<NodeId> {
        self.focus_id
    }

    /// Looks up a retained `accesskit::Node` by `NodeId`.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Iterates over all reachable `NodeId`s currently retained in the tree.
    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.keys().copied()
    }

    /// Returns the number of reachable nodes currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` when the store holds zero nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Projects a stored `NodeId` into its [`DomNodeSpec`] given `canvas_rect` and `zoom_factor`.
    #[must_use]
    pub fn build_dom_spec(
        &self,
        id: NodeId,
        canvas_rect: egui::Rect,
        zoom_factor: f32,
    ) -> Option<DomNodeSpec> {
        let node = self.nodes.get(&id)?;
        if node.is_hidden() {
            return None;
        }
        let children: Vec<NodeId> = node
            .children()
            .iter()
            .copied()
            .filter(|cid| self.nodes.get(cid).is_some_and(|c| !c.is_hidden()))
            .collect();
        Some(project_node_spec(node, children, canvas_rect, zoom_factor))
    }

    fn prune_unreachable(&mut self) {
        let Some(root) = self.root_id else {
            return;
        };
        let mut reachable = HashSet::with_capacity(self.nodes.len());
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if !reachable.insert(id) {
                continue;
            }
            if let Some(node) = self.nodes.get(&id) {
                stack.extend(node.children().iter().copied());
            }
        }
        self.nodes.retain(|id, _| reachable.contains(id));
    }
}

fn map_role_tokens(
    role: Role,
) -> (
    &'static str,
    &'static str,
    Option<&'static str>,
    Option<&'static str>,
) {
    match role {
        Role::Window => ("div", "window", None, None),
        Role::Region => ("section", "region", Some("region"), None),
        Role::Navigation => ("nav", "navigation", Some("navigation"), None),
        Role::Main => ("main", "main", Some("main"), None),
        Role::Toolbar => ("div", "toolbar", Some("toolbar"), None),
        Role::MenuBar => ("div", "menubar", Some("menubar"), None),
        Role::Menu => ("div", "menu", Some("menu"), None),
        Role::MenuItem => ("button", "menuitem", Some("menuitem"), None),
        Role::MenuItemCheckBox => ("button", "menuitemcheckbox", Some("menuitemcheckbox"), None),
        Role::MenuItemRadio => ("button", "menuitemradio", Some("menuitemradio"), None),
        Role::Heading => ("h2", "heading", Some("heading"), None),
        Role::Button | Role::DefaultButton => ("button", "button", None, None),
        Role::CheckBox => ("div", "checkbox", Some("checkbox"), None),
        Role::Switch => ("div", "switch", Some("switch"), None),
        Role::RadioButton => ("div", "radiobutton", Some("radio"), None),
        Role::Slider => ("input", "slider", Some("slider"), Some("range")),
        Role::SpinButton => ("input", "spinbutton", Some("spinbutton"), Some("number")),
        Role::Splitter => ("div", "splitter", Some("separator"), None),
        Role::TextInput => ("input", "textinput", None, Some("text")),
        Role::SearchInput => ("input", "searchinput", Some("searchbox"), Some("search")),
        Role::PasswordInput => ("input", "passwordinput", None, Some("password")),
        Role::MultilineTextInput => ("textarea", "multilinetextinput", None, None),
        Role::ComboBox => ("button", "combobox", Some("combobox"), None),
        Role::TabList => ("div", "tablist", Some("tablist"), None),
        Role::Tab => ("button", "tab", Some("tab"), None),
        Role::TabPanel => ("div", "tabpanel", Some("tabpanel"), None),
        Role::Dialog => ("div", "dialog", Some("dialog"), None),
        Role::AlertDialog => ("div", "alertdialog", Some("alertdialog"), None),
        _ => map_secondary_role_tokens(role),
    }
}

fn map_secondary_role_tokens(
    role: Role,
) -> (
    &'static str,
    &'static str,
    Option<&'static str>,
    Option<&'static str>,
) {
    match role {
        Role::Table => ("div", "table", Some("table"), None),
        Role::Grid => ("div", "grid", Some("grid"), None),
        Role::Row => ("div", "row", Some("row"), None),
        Role::ColumnHeader => ("div", "columnheader", Some("columnheader"), None),
        Role::RowHeader => ("div", "rowheader", Some("rowheader"), None),
        Role::Cell => ("div", "cell", Some("cell"), None),
        Role::List => ("div", "list", Some("list"), None),
        Role::ListItem => ("div", "listitem", Some("listitem"), None),
        Role::Meter => ("div", "meter", Some("meter"), None),
        Role::ProgressIndicator => ("div", "progressbar", Some("progressbar"), None),
        Role::GraphicsDocument => ("div", "graphics-document", Some("graphics-document"), None),
        Role::Image => ("div", "image", Some("img"), None),
        Role::Canvas => ("div", "canvas", Some("img"), None),
        Role::SvgRoot | Role::Figure => ("figure", "figure", Some("figure"), None),
        Role::Status => ("div", "status", Some("status"), None),
        Role::Alert => ("div", "alert", Some("alert"), None),
        Role::Link => ("a", "link", Some("link"), None),
        Role::Group => ("div", "group", Some("group"), None),
        Role::ScrollView => ("div", "scrollview", None, None),
        Role::Label | Role::Paragraph | Role::TextRun => ("span", "label", None, None),
        _ => ("div", "generic", None, None),
    }
}

fn compute_css_bounds(node: &Node, canvas_rect: egui::Rect, zoom_factor: f32) -> Option<CssBounds> {
    let b = node.bounds()?;
    let zoom = if zoom_factor.is_finite() && zoom_factor > 0.0 {
        zoom_factor
    } else {
        1.0
    };
    let left = canvas_rect.min.x + (b.x0 as f32) * zoom;
    let top = canvas_rect.min.y + (b.y0 as f32) * zoom;
    let width = ((b.x1 - b.x0) as f32 * zoom).max(1.0);
    let height = ((b.y1 - b.y0) as f32 * zoom).max(1.0);
    if left.is_finite() && top.is_finite() && width.is_finite() && height.is_finite() {
        Some(CssBounds {
            left,
            top,
            width,
            height,
        })
    } else {
        None
    }
}

fn toggled_str(toggled: Toggled) -> &'static str {
    match toggled {
        Toggled::False => "false",
        Toggled::True => "true",
        Toggled::Mixed => "mixed",
    }
}

fn resolve_checked_and_pressed(
    role: Role,
    toggled: Option<Toggled>,
) -> (Option<&'static str>, Option<&'static str>) {
    let Some(t) = toggled else {
        return (None, None);
    };
    let token = toggled_str(t);
    match role {
        Role::Button | Role::DefaultButton => (None, Some(token)),
        _ => (Some(token), None),
    }
}

fn resolve_labels_and_text(
    node: &Node,
    tag: &'static str,
    has_children: bool,
) -> (Option<String>, Option<String>, Option<String>) {
    let role = node.role();
    let raw_label = node.label().map(str::to_owned);
    let raw_val = node.value().map(str::to_owned);
    let aria_label = match role {
        Role::Label | Role::Paragraph | Role::TextRun => raw_label.clone(),
        _ => raw_label.clone().or_else(|| {
            matches!(role, Role::Heading)
                .then(|| raw_val.clone())
                .flatten()
        }),
    };
    let text_content = if has_children || matches!(tag, "input" | "textarea") {
        None
    } else {
        match role {
            Role::Label | Role::Paragraph | Role::TextRun => raw_val.clone().or(raw_label),
            Role::Heading | Role::Button | Role::DefaultButton | Role::Tab => {
                raw_label.or_else(|| raw_val.clone())
            }
            _ => None,
        }
    };
    let desc = match (node.description(), &raw_val, role) {
        (Some(d), _, _) => Some(d.to_owned()),
        (None, Some(v), Role::Image | Role::Canvas | Role::GraphicsDocument | Role::Figure) => {
            Some(v.clone())
        }
        _ => None,
    };
    (aria_label, text_content, desc)
}

fn resolve_tab_index(node: &Node, tag: &'static str, disabled: bool) -> Option<i32> {
    if disabled || matches!(tag, "button" | "input" | "textarea") {
        return None;
    }
    let role_focusable = matches!(
        node.role(),
        Role::Splitter
            | Role::CheckBox
            | Role::RadioButton
            | Role::Switch
            | Role::GraphicsDocument
            | Role::Link
    );
    (node.supports_action(Action::Focus) || role_focusable).then_some(0)
}

fn resolve_value_fields(
    node: &Node,
    tag: &'static str,
) -> (Option<f64>, Option<String>, Option<String>) {
    let role = node.role();
    let numeric_now = node.numeric_value();
    let is_range_like = matches!(
        role,
        Role::Slider | Role::SpinButton | Role::Splitter | Role::Meter | Role::ProgressIndicator
    );
    let aria_valuetext = node
        .value()
        .filter(|_| is_range_like || matches!(role, Role::ComboBox))
        .map(str::to_owned);
    let value = if tag == "input" && matches!(role, Role::Slider | Role::SpinButton) {
        numeric_now
            .map(|v| v.to_string())
            .or_else(|| node.value().map(str::to_owned))
    } else {
        node.value().map(str::to_owned)
    };
    (numeric_now, value, aria_valuetext)
}

fn resolve_live_and_orientation(node: &Node) -> (Option<&'static str>, Option<&'static str>) {
    let orientation = node.orientation().map(|o| match o {
        Orientation::Horizontal => "horizontal",
        Orientation::Vertical => "vertical",
    });
    let live = match node.live() {
        Some(Live::Polite) => Some("polite"),
        Some(Live::Assertive) => Some("assertive"),
        _ => None,
    };
    (orientation, live)
}

fn project_node_spec(
    node: &Node,
    children: Vec<NodeId>,
    canvas_rect: egui::Rect,
    zoom_factor: f32,
) -> DomNodeSpec {
    let role = node.role();
    let (tag, data_ak_role, aria_role, input_type) = map_role_tokens(role);
    let disabled = node.is_disabled();
    let focusable_when_disabled = disabled && node.supports_action(Action::Focus);
    let (aria_label, text_content, aria_description) =
        resolve_labels_and_text(node, tag, !children.is_empty());
    let (aria_checked, aria_pressed) = resolve_checked_and_pressed(role, node.toggled());
    let (numeric_now, value, aria_valuetext) = resolve_value_fields(node, tag);
    let (aria_orientation, aria_live) = resolve_live_and_orientation(node);

    DomNodeSpec {
        tag,
        data_ak_role,
        role: aria_role,
        input_type,
        aria_label,
        aria_description,
        aria_roledescription: node.role_description().map(str::to_owned).or_else(|| {
            matches!(role, Role::GraphicsDocument).then(|| "Interactive plot".to_owned())
        }),
        text_content,
        value,
        placeholder: node.placeholder().map(str::to_owned),
        numeric_now,
        numeric_min: node.min_numeric_value(),
        numeric_max: node.max_numeric_value(),
        numeric_step: node.numeric_value_step(),
        aria_valuetext,
        aria_checked,
        aria_pressed,
        aria_selected: node.is_selected().map(|b| if b { "true" } else { "false" }),
        aria_expanded: node.is_expanded().map(|b| if b { "true" } else { "false" }),
        aria_orientation,
        aria_level: node
            .level()
            .or_else(|| matches!(role, Role::Heading).then_some(2)),
        aria_live,
        aria_modal: node.is_modal() || matches!(role, Role::Dialog | Role::AlertDialog),
        disabled,
        focusable_when_disabled,
        read_only: node.is_read_only(),
        tab_index: resolve_tab_index(node, tag, disabled && !focusable_when_disabled),
        children,
        bounds: compute_css_bounds(node, canvas_rect, zoom_factor),
    }
}
