use sqlx::Executor;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::path::Path;
use std::str::FromStr;

/// A client for interacting with the SQLite database.
#[derive(Debug, Clone)]
pub struct DbClient {
    /// The SQLite connection pool for database operations.
    pool: SqlitePool,
}

impl DbClient {
    /// Create a database client backed by the default gateway SQLite database.
    pub fn new() -> Self {
        let pool = Self::pool_options().connect_lazy_with(Self::default_options());
        Self { pool }
    }

    /// Create a database client backed by a SQLite database file path.
    ///
    /// Creates the parent directory if needed and configures SQLite to create
    /// the database file when it does not exist yet.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Filesystem path to the SQLite database file.
    pub fn from_path(db_path: &Path) -> crate::Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let options = SqliteConnectOptions::new()
            .filename(db_path)
            .create_if_missing(true)
            .foreign_keys(true);
        let pool = Self::pool_options().connect_lazy_with(options);
        Ok(Self { pool })
    }

    /// Create a database client from an existing SQLite pool.
    ///
    /// # Arguments
    ///
    /// * `pool` - The preconfigured SQLite connection pool to wrap.
    pub fn from_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Return the underlying SQLite connection pool.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Generate a UTC RFC3339 timestamp with timezone information for persisted rows.
    pub(crate) fn current_timestamp() -> String {
        chrono::Utc::now().to_rfc3339()
    }

    /// Apply the schema from `db_schema/schema.sql` to the connected database.
    pub async fn initialize_schema(&self) -> crate::Result {
        self.pool
            .execute(include_str!("../../../db_schema/schema.sql"))
            .await?;
        Ok(())
    }

    fn default_options() -> SqliteConnectOptions {
        SqliteConnectOptions::from_str("sqlite://nexo.db")
            .expect("Failed to parse the default database URL")
            .create_if_missing(true)
            .foreign_keys(true)
    }

    fn pool_options() -> SqlitePoolOptions {
        SqlitePoolOptions::new()
            .max_connections(5)
            .after_connect(|connection, _meta| {
                Box::pin(async move {
                    connection.execute("PRAGMA foreign_keys = ON;").await?;
                    connection.execute("PRAGMA journal_mode = WAL;").await?;
                    connection.execute("PRAGMA synchronous = NORMAL;").await?;
                    connection.execute("PRAGMA busy_timeout = 5000;").await?;
                    connection.execute("PRAGMA cache_size = -64000;").await?;
                    Ok(())
                })
            })
    }
}
