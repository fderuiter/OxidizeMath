//! Zero-dependency BibTeX parser, persistent-ID (`doi` / `eprint`) validator, and citation extractors.

mod extract;
mod parser;

pub use extract::{
    extract_doc_bib_keys, extract_doc_spec_ids, extract_tex_cite_keys, validate_math_spec_markdown,
};
pub use parser::{BibtexDatabase, BibtexEntry, is_valid_bibtex_key};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bibtex_entries_adjacent_blocks_and_validates_persistent_ids() {
        let sample = r#"
% Leading comment
@article{alpha2024,
  author = {Ada Lovelace},
  title = {{Nested {Braces} Title}},
  doi = {10.1000/alpha.1}
}@misc{beta2025,
  title = "Quoted Value",
  eprint = {2501.12345}
}
@book{legacy1990,
  author = {No Persistent Id},
  title = {Legacy Book},
  year = {1990}
}
"#;
        let db = BibtexDatabase::parse(sample).expect("parse sample bib");
        assert_eq!(db.entries().len(), 3);
        assert!(db.verify_cited_key("alpha2024").is_ok());
        assert!(db.verify_cited_key("beta2025").is_ok());
        let missing_id = db.verify_cited_key("legacy1990").unwrap_err();
        assert!(missing_id.contains("doi") && missing_id.contains("eprint"));
        assert!(db.verify_cited_key("unknown_key").is_err());
        assert!(db.verify_cited_key("Prose Citation (2024)").is_err());
    }

    #[test]
    fn extracts_tex_and_doc_citations_accurately() {
        let tex = "\\cite{k1, k2} % \\cite{ignored_comment}\n\\citep[see][p.~2]{k3}\n";
        assert_eq!(extract_tex_cite_keys(tex), vec!["k1", "k2", "k3"]);

        let doc = "References: `@vaswani2017attention`, [bib:dormand1980family], `spec:transformer_v1`, docs/math_specs/biology_v1.md.";
        assert_eq!(
            extract_doc_bib_keys(doc),
            vec!["vaswani2017attention", "dormand1980family"]
        );
        assert_eq!(
            extract_doc_spec_ids(doc),
            vec!["transformer_v1", "biology_v1"]
        );
    }
}
