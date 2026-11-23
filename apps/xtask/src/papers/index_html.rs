#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IndexTarget {
    Archive,
    GitHubPages,
}

const INDEX_HTML_HEAD: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
  <title>OxidizeMath Academic Papers</title>
  <style>
    :root { --bg: #0d1117; --card-bg: #161b22; --card-border: #30363d; --text: #c9d1d9; --heading: #f0f6fc; --accent: #58a6ff; --accent-hover: #79c0ff; --meta: #8b949e; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; background-color: var(--bg); color: var(--text); line-height: 1.6; margin: 0; padding: 2rem 1rem; }
    .container { max-width: 1100px; margin: 0 auto; }
    header { margin-bottom: 2.5rem; border-bottom: 1px solid var(--card-border); padding-bottom: 1.5rem; }
    h1 { color: var(--heading); font-size: 2.25rem; margin: 0 0 0.5rem 0; }
    .subtitle { color: var(--meta); font-size: 1.1rem; margin: 0 0 1rem 0; }
    .actions-bar { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1rem; }
    .btn { display: inline-block; padding: 0.5rem 1rem; background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 6px; color: var(--accent); text-decoration: none; font-weight: 500; transition: all 0.2s ease; }
    .btn:hover { background-color: #21262d; border-color: var(--accent); color: var(--accent-hover); }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1.5rem; }
    .card { background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 1.25rem; display: flex; flex-direction: column; justify-content: space-between; }
    .card-title { color: var(--heading); font-size: 1.2rem; margin: 0 0 0.5rem 0; }
    .card-meta { font-size: 0.85rem; color: var(--meta); margin-bottom: 0.75rem; }
    .card-abstract { font-size: 0.9rem; color: var(--text); margin-bottom: 1.25rem; flex-grow: 1; }
    .card-links { display: flex; justify-content: space-between; align-items: center; margin-top: auto; }
    footer { margin-top: 4rem; text-align: center; color: var(--meta); font-size: 0.85rem; border-top: 1px solid var(--card-border); padding-top: 2rem; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>OxidizeMath Academic Papers</h1>
      <p class="subtitle">Permanent theoretical foundation and empirical verification publications (28 papers)</p>
      <div class="actions-bar">
"#;

pub(super) const INDEX_HTML_FOOTER: &str = r#"    </main>
    <footer>
      <p>Published by OxidizeMath Automated Continuous Verification &amp; Publishing System</p>
    </footer>
  </div>
</body>
</html>
"#;

pub(super) fn render_index_header(target: IndexTarget) -> String {
    let (bundle_href, bundle_label, checksums_href) = match target {
        IndexTarget::Archive => (
            "papers-bundle.tar.gz",
            "📦 Download All Papers (.tar.gz)",
            "SHA256SUMS",
        ),
        IndexTarget::GitHubPages => (
            "https://github.com/fderuiter/OxidizeMath/releases",
            "📦 GitHub Releases (.tar.gz)",
            "../SHA256SUMS",
        ),
    };
    format!(
        r#"{INDEX_HTML_HEAD}        <a class="btn" href="{bundle_href}">{bundle_label}</a>
        <a class="btn" href="{checksums_href}">🔒 Checksum Manifest (SHA256)</a>
        <a class="btn" href="https://github.com/fderuiter/OxidizeMath">💻 GitHub Repository</a>
      </div>
    </header>
    <main class="grid">
"#
    )
}

#[cfg(test)]
pub(super) fn render_paper_card(
    stem: &str,
    title: &str,
    author: &str,
    abstract_text: &str,
) -> String {
    render_paper_card_for_target(stem, title, author, abstract_text, IndexTarget::Archive)
}

pub(super) fn render_paper_card_for_target(
    stem: &str,
    title: &str,
    author: &str,
    abstract_text: &str,
    target: IndexTarget,
) -> String {
    let escaped_stem = escape_html(stem);
    let encoded_stem = escape_html(&encode_uri_component(stem));
    let escaped_title = escape_html(title);
    let escaped_author = escape_html(author);
    let escaped_abstract = escape_html(abstract_text);
    let pdf_prefix = match target {
        IndexTarget::Archive => "",
        IndexTarget::GitHubPages => "../",
    };

    format!(
        r#"      <article class="card">
        <div>
          <h2 class="card-title">{escaped_title}</h2>
          <div class="card-meta">By {escaped_author} &bull; <code>{escaped_stem}.pdf</code></div>
          <p class="card-abstract">{escaped_abstract}</p>
        </div>
        <div class="card-links">
          <a class="btn" href="{pdf_prefix}{encoded_stem}.pdf" target="_blank" rel="noopener">📄 Read PDF</a>
        </div>
      </article>
"#
    )
}

pub(super) fn extract_latex_command(content: &str, cmd: &str) -> Option<String> {
    let idx = content.find(cmd)?;
    let after_cmd = &content[idx + cmd.len()..];
    let open_brace = after_cmd.find('{')?;
    let start = &after_cmd[open_brace + 1..];

    let mut depth = 1;
    let mut end = 0;
    for (i, c) in start.char_indices() {
        if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                end = i;
                break;
            }
        }
    }

    if depth == 0 {
        Some(clean_latex(&start[..end]))
    } else {
        None
    }
}

pub(super) fn extract_latex_environment(content: &str, env: &str) -> Option<String> {
    let begin_tag = format!("\\begin{{{}}}", env);
    let end_tag = format!("\\end{{{}}}", env);

    let start_idx = content.find(&begin_tag)?;
    let after_begin = &content[start_idx + begin_tag.len()..];
    let end_idx = after_begin.find(&end_tag)?;

    Some(clean_latex(&after_begin[..end_idx]))
}

fn clean_latex(s: &str) -> String {
    let s = s
        .replace("\\texttt{", "")
        .replace("\\textbf{", "")
        .replace("\\textit{", "");
    let s = s.replace(['}', '{'], "");
    let s = s.replace("\\\\", " ").replace('\n', " ");
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(super) fn format_title_from_stem(stem: &str) -> String {
    if stem.eq_ignore_ascii_case("grpo") {
        return "Group Relative Policy Optimization (GRPO)".to_string();
    }
    stem.replace('_', " ")
        .split_whitespace()
        .map(|word| {
            let mut c = word.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

pub(super) fn encode_uri_component(s: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::with_capacity(s.len());
    for byte in s.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(byte & 0x0F)]));
        }
    }
    encoded
}
