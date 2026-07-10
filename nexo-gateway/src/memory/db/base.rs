use sqlx::Executor;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
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
        let options = SqliteConnectOptions::from_str("sqlite://nexo.db")
            .expect("Failed to parse the default database URL")
            .create_if_missing(true)
            .foreign_keys(true);
        let pool = Self::pool_options().connect_lazy_with(options);
        Self { pool }
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
        self.pool.execute(include_str!("../../../db_schema/schema.sql")).await?;
        Ok(())
    }

    fn pool_options() -> SqlitePoolOptions {
        SqlitePoolOptions::new().max_connections(5).after_connect(|connection, _meta| {
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
