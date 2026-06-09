use crate::frontmatter::{DATETIME_FORMAT, frontmatter_date};
use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A single category associated with one or more notes.
#[derive(Debug, Serialize, Deserialize, Clone, JsonSchema, Eq, PartialEq, Hash)]
#[serde(transparent)]
pub struct NoteCategory {
    /// The title of the category.
    pub title: String,
}

/// Private struct used for parsing frontmatter of a note.
#[derive(Debug, Serialize, Deserialize)]
struct NoteFrontmatter {
    title: String,
    #[serde(with = "frontmatter_date")]
    datetime: DateTime<Utc>,
    categories: Option<Vec<NoteCategory>>,
}

/// A single note, consisting of metadata and body content.
#[derive(Debug, Serialize, Deserialize)]
pub struct Note {
    /// The title of the note
    pub title: String,

    /// The datetime of the note, used for filename generation and filtering.
    pub datetime: DateTime<Utc>,

    /// Optional categories associated with the note.
    pub categories: Option<Vec<NoteCategory>>,

    /// The body content of the note, following the frontmatter.
    pub body: String,
}

impl Note {
    /// Generate a filename for the note based on its datetime, in the format "YYYY-MM-DD_HH-MM-SS.md".
    pub fn filename(&self) -> String {
        let datetime_str = self.datetime.format(DATETIME_FORMAT).to_string();
        format!("{}.md", datetime_str)
    }

    /// Render the frontmatter of the note as a YAML string.
    pub fn frontmatter(&self) -> String {
        let frontmatter = NoteFrontmatter {
            title: self.title.clone(),
            datetime: self.datetime,
            categories: self.categories.as_ref().map(|cats| {
                cats.iter()
                    .map(|cat| NoteCategory {
                        title: cat.title.clone(),
                    })
                    .collect()
            }),
        };

        format! {"---\n{}---\n", yaml_serde::to_string(&frontmatter).expect("Failed to serialize frontmatter")}
    }

    /// Return the full content of the note, including frontmatter and body,
    /// as it would be rendered in a note file.
    pub fn content(&self) -> String {
        format!("{}\n{}", self.frontmatter(), self.body)
    }

    /// Return the full content of the note, including frontmatter and body,
    /// as it would be rendered in a note file.
    ///
    /// Consumes the Note instance.
    pub fn into_content(self) -> String {
        format!("{}\n{}", self.frontmatter(), self.body)
    }

    /// Parse a note from its full content, which includes frontmatter and body.
    ///
    /// Used by new_from_content and new_into_content functions to construct Note
    /// instances from content strings.
    fn parse_content(
        content: &str,
    ) -> anyhow::Result<(String, DateTime<Utc>, Option<Vec<NoteCategory>>, &str)> {
        let rest = content
            .strip_prefix("---\n")
            .ok_or_else(|| anyhow::anyhow!("missing frontmatter"))?;
        let end = rest
            .find("\n---")
            .ok_or_else(|| anyhow::anyhow!("missing end of frontmatter"))?;

        let frontmatter = yaml_serde::from_str::<NoteFrontmatter>(&rest[..end])?;
        let body = &rest[end + "\n---".len()..];

        let title = frontmatter.title;
        let datetime = frontmatter.datetime;
        let categories = frontmatter.categories;

        Ok((title, datetime, categories, body))
    }

    /// Create a Note instance from a content string reference, which includes frontmatter and body.
    pub fn from_str(content: &str) -> anyhow::Result<Self> {
        let (title, datetime, categories, body_slice) = Self::parse_content(content)?;

        Ok(Note {
            title,
            datetime,
            categories,
            body: body_slice.to_string(), // Clones the body slice here
        })
    }

    /// Create a Note instance from a content string, which includes frontmatter and body.
    pub fn from_string(mut content: String) -> anyhow::Result<Self> {
        let (title, datetime, categories, body_slice) = Self::parse_content(&content)?;

        // Zero-copy extraction of the body slice from the original content string
        let body_start_idx = body_slice.as_ptr() as usize - content.as_ptr() as usize;
        let body = content.split_off(body_start_idx);

        Ok(Note {
            title,
            datetime,
            categories,
            body,
        })
    }
}
