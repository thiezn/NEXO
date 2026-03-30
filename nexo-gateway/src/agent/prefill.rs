use nexo_ws_schema::Frame;
use sha2::{Digest, Sha256};
use sqlx::SqlitePool;
use std::io;
use std::path::{Path, PathBuf};

// ── Error type ──────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum Error {
    Db(sqlx::Error),
    Io(io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Db(e) => write!(f, "database error: {e}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl From<sqlx::Error> for Error {
    fn from(e: sqlx::Error) -> Self {
        Self::Db(e)
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

// ── Record types ────────────────────────────────────────────────────────────────

pub struct MarkdownFileRecord {
    pub id: String,
    pub category: String,
    pub description: String,
    pub filename: String,
    pub created_at: String,
    pub updated_at: String,
}

pub struct CollectionRecord {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub markdown_ids: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
}

impl From<MarkdownFileRecord> for nexo_ws_schema::MarkdownFileEntry {
    fn from(r: MarkdownFileRecord) -> Self {
        nexo_ws_schema::MarkdownFileEntry {
            id: r.id,
            category: r.category,
            description: r.description,
            filename: r.filename,
            created_at: r.created_at,
            updated_at: r.updated_at,
        }
    }
}

impl From<CollectionRecord> for nexo_ws_schema::CollectionEntry {
    fn from(c: CollectionRecord) -> Self {
        nexo_ws_schema::CollectionEntry {
            id: c.id,
            name: c.name,
            description: c.description,
            markdown_ids: c.markdown_ids,
            created_at: c.created_at,
            updated_at: c.updated_at,
        }
    }
}

// ── SHA utility ─────────────────────────────────────────────────────────────────

/// Compute lowercase hex(SHA-256(combined_content)).
pub fn compute_sha(combined: &str) -> String {
    format!("{:x}", Sha256::digest(combined.as_bytes()))
}

// ── Storage path helpers ────────────────────────────────────────────────────────

pub fn markdown_dir(storage_root: &Path) -> PathBuf {
    storage_root.join("markdown")
}

pub fn markdown_path(storage_root: &Path, id: &str) -> PathBuf {
    markdown_dir(storage_root).join(format!("{id}.md"))
}

// ── Markdown CRUD ───────────────────────────────────────────────────────────────

/// Create a new markdown file: write content to disk and insert metadata row.
/// Returns the new UUID v7 ID.
pub async fn create_markdown(
    pool: &SqlitePool,
    storage_root: &Path,
    category: &str,
    description: &str,
    content: &str,
) -> Result<String, Error> {
    let id = Frame::new_id();
    let filename = format!("{id}.md");
    let path = markdown_path(storage_root, &id);

    std::fs::create_dir_all(markdown_dir(storage_root))?;
    std::fs::write(&path, content)?;

    let result = sqlx::query(
        "INSERT INTO markdown_files (id, category, description, filename) VALUES (?, ?, ?, ?)",
    )
    .bind(&id)
    .bind(category)
    .bind(description)
    .bind(&filename)
    .execute(pool)
    .await;

    if let Err(e) = result {
        // Best-effort cleanup of the file if the DB insert failed
        let _ = std::fs::remove_file(&path);
        return Err(Error::Db(e));
    }

    Ok(id)
}

/// List all markdown file metadata rows.
pub async fn list_markdown(pool: &SqlitePool) -> Result<Vec<MarkdownFileRecord>, Error> {
    let rows: Vec<(String, String, String, String, String, String)> = sqlx::query_as(
        "SELECT id, category, description, filename, created_at, updated_at
         FROM markdown_files ORDER BY created_at ASC",
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|(id, category, description, filename, created_at, updated_at)| {
            MarkdownFileRecord { id, category, description, filename, created_at, updated_at }
        })
        .collect())
}

/// Delete a markdown file: remove the disk file and the metadata row.
/// Returns true if the row existed.
pub async fn delete_markdown(
    pool: &SqlitePool,
    storage_root: &Path,
    id: &str,
) -> Result<bool, Error> {
    let row: Option<(String,)> =
        sqlx::query_as("SELECT filename FROM markdown_files WHERE id = ?")
            .bind(id)
            .fetch_optional(pool)
            .await?;

    let Some((filename,)) = row else {
        return Ok(false);
    };

    sqlx::query("DELETE FROM markdown_files WHERE id = ?")
        .bind(id)
        .execute(pool)
        .await?;

    let path = markdown_dir(storage_root).join(&filename);
    if let Err(e) = std::fs::remove_file(&path) {
        if e.kind() != io::ErrorKind::NotFound {
            return Err(Error::Io(e));
        }
    }

    Ok(true)
}

// ── Collection CRUD ─────────────────────────────────────────────────────────────

/// Create a prefill collection with an ordered list of markdown file IDs.
/// Returns the new UUID v7 collection ID.
pub async fn create_collection(
    pool: &SqlitePool,
    name: &str,
    description: Option<&str>,
    markdown_ids: &[String],
) -> Result<String, Error> {
    let id = Frame::new_id();

    sqlx::query("INSERT INTO prefill_collections (id, name, description) VALUES (?, ?, ?)")
        .bind(&id)
        .bind(name)
        .bind(description)
        .execute(pool)
        .await?;

    for (position, markdown_id) in markdown_ids.iter().enumerate() {
        sqlx::query(
            "INSERT INTO prefill_collection_items (collection_id, position, markdown_id)
             VALUES (?, ?, ?)",
        )
        .bind(&id)
        .bind(position as i64)
        .bind(markdown_id)
        .execute(pool)
        .await?;
    }

    Ok(id)
}

/// List all prefill collections with their ordered markdown file IDs.
pub async fn list_collections(pool: &SqlitePool) -> Result<Vec<CollectionRecord>, Error> {
    let rows: Vec<(String, String, Option<String>, String, String, Option<String>)> =
        sqlx::query_as(
            "SELECT pc.id, pc.name, pc.description, pc.created_at, pc.updated_at, pci.markdown_id
             FROM prefill_collections pc
             LEFT JOIN prefill_collection_items pci ON pci.collection_id = pc.id
             ORDER BY pc.created_at ASC, pc.id, pci.position ASC",
        )
        .fetch_all(pool)
        .await?;

    let mut records: Vec<CollectionRecord> = Vec::new();
    for (id, name, description, created_at, updated_at, markdown_id) in rows {
        if records.last().map(|r| r.id == id).unwrap_or(false) {
            if let Some(mid) = markdown_id {
                records.last_mut().unwrap().markdown_ids.push(mid);
            }
        } else {
            records.push(CollectionRecord {
                id,
                name,
                description,
                markdown_ids: markdown_id.into_iter().collect(),
                created_at,
                updated_at,
            });
        }
    }

    Ok(records)
}

/// Delete a collection (cascades to items). Returns true if it existed.
pub async fn delete_collection(pool: &SqlitePool, id: &str) -> Result<bool, Error> {
    let result = sqlx::query("DELETE FROM prefill_collections WHERE id = ?")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

// ── Content assembly ────────────────────────────────────────────────────────────

/// Load each markdown file in a collection in order and concatenate them.
/// Returns `Some((combined_content, sha256_hex))`, or `None` if the collection
/// does not exist.
pub async fn resolve_collection(
    pool: &SqlitePool,
    storage_root: &Path,
    collection_id: &str,
) -> Result<Option<(String, String)>, Error> {
    // A single LEFT JOIN distinguishes "not found" (0 rows) from "empty collection"
    // (1 row with NULL filename).
    let rows: Vec<(Option<String>,)> = sqlx::query_as(
        "SELECT mf.filename
         FROM prefill_collections c
         LEFT JOIN prefill_collection_items pci ON pci.collection_id = c.id
         LEFT JOIN markdown_files mf ON mf.id = pci.markdown_id
         WHERE c.id = ?
         ORDER BY pci.position ASC",
    )
    .bind(collection_id)
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        return Ok(None);
    }

    let mut parts: Vec<String> = Vec::new();
    for filename in rows.into_iter().filter_map(|(f,)| f) {
        let path = markdown_dir(storage_root).join(&filename);
        let content = std::fs::read_to_string(&path).map_err(|e| {
            tracing::warn!("Failed to read markdown file '{}': {e}", path.display());
            Error::Io(e)
        })?;
        parts.push(content);
    }

    let combined = parts.join("\n\n");
    let sha = compute_sha(&combined);

    Ok(Some((combined, sha)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn compute_sha_is_deterministic() {
        let s1 = compute_sha("hello world");
        let s2 = compute_sha("hello world");
        assert_eq!(s1, s2);
    }

    #[test]
    fn compute_sha_differs_for_different_content() {
        assert_ne!(compute_sha("hello"), compute_sha("world"));
    }

    #[test]
    fn compute_sha_is_64_hex_chars() {
        let s = compute_sha("test");
        assert_eq!(s.len(), 64);
        assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn create_and_list_markdown(pool: SqlitePool) {
        let tmp = TempDir::new().unwrap();
        let id = create_markdown(&pool, tmp.path(), "identity", "Soul file", "# SOUL\nBe helpful.")
            .await
            .unwrap();
        assert!(!id.is_empty());

        let path = markdown_path(tmp.path(), &id);
        assert!(path.exists());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "# SOUL\nBe helpful.");

        let files = list_markdown(&pool).await.unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].id, id);
        assert_eq!(files[0].category, "identity");
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn delete_markdown_removes_file_and_row(pool: SqlitePool) {
        let tmp = TempDir::new().unwrap();
        let id =
            create_markdown(&pool, tmp.path(), "skill", "Coding assistant", "# CODING")
                .await
                .unwrap();

        let path = markdown_path(tmp.path(), &id);
        assert!(path.exists());

        let deleted = delete_markdown(&pool, tmp.path(), &id).await.unwrap();
        assert!(deleted);
        assert!(!path.exists());

        let files = list_markdown(&pool).await.unwrap();
        assert!(files.is_empty());
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn delete_nonexistent_markdown_returns_false(pool: SqlitePool) {
        let tmp = TempDir::new().unwrap();
        let deleted = delete_markdown(&pool, tmp.path(), "no-such-id").await.unwrap();
        assert!(!deleted);
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn create_collection_and_resolve(pool: SqlitePool) {
        let tmp = TempDir::new().unwrap();
        let id1 =
            create_markdown(&pool, tmp.path(), "identity", "Soul", "I am helpful.")
                .await
                .unwrap();
        let id2 =
            create_markdown(&pool, tmp.path(), "skill", "Coder", "I write Rust.")
                .await
                .unwrap();

        let col_id = create_collection(&pool, "default", None, &[id1, id2]).await.unwrap();
        assert!(!col_id.is_empty());

        let (combined, sha) = resolve_collection(&pool, tmp.path(), &col_id)
            .await
            .unwrap()
            .unwrap();

        assert!(combined.contains("I am helpful."));
        assert!(combined.contains("I write Rust."));
        assert_eq!(sha, compute_sha(&combined));
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn resolve_nonexistent_collection_returns_none(pool: SqlitePool) {
        let tmp = TempDir::new().unwrap();
        let result = resolve_collection(&pool, tmp.path(), "no-such-id").await.unwrap();
        assert!(result.is_none());
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn delete_collection_returns_true(pool: SqlitePool) {
        let col_id = create_collection(&pool, "test", None, &[]).await.unwrap();
        let deleted = delete_collection(&pool, &col_id).await.unwrap();
        assert!(deleted);

        let cols = list_collections(&pool).await.unwrap();
        assert!(cols.is_empty());
    }
}
