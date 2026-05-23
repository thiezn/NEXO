use nexo_ws_schema::SchemaSection;

/// Generate WebSocket protocol schemas and write them to stdout or a file.
///
/// # Errors
///
/// Returns an error when the output path cannot be written.
pub fn run_schema(section: SchemaSection, output: Option<&str>) -> cli_helpers::Result {
    let json = nexo_ws_schema::schema_json(section);

    match output {
        Some(path) => {
            std::fs::write(path, &json)?;
            tracing::info!("Schema written to {path}");
        }
        None => println!("{json}"),
    }
    Ok(())
}
