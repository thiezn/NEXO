use sqlx::sqlite::SqlitePool;

/// A client for interacting with the SQLite database.
#[derive(Debug, Clone)]
pub struct DbClient {
    /// The SQLite connection pool for database operations.
    pool: SqlitePool,
}

impl DbClient {
    pub fn new() -> Self {
        let pool = SqlitePool::connect_lazy("sqlite://nexo.db")
            .expect("Failed to connect to the database");
        Self { pool }
    }
}
