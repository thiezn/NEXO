use nexo_ws_schema::SchemaSection;

/// Generate WebSocket protocol schemas and write them to stdout or a file.
///
/// # Errors
///
/// Returns an error when the output path cannot be written.
pub async fn run(section: SchemaSection, output: Option<String>) -> cli_helpers::Result {
    let json = nexo_ws_schema::schema_json(section);

    match output {
        Some(path) => {
            tokio::fs::write(&path, &json).await?;
            tracing::info!("Schema written to {path}");
        }
        None => println!("{json}"),
    }
    Ok(())
}
