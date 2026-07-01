use nexo_ws_schema::SchemaSection;

/// Run the `schema` command, which outputs the JSON schema for a given section.
pub fn run(section: SchemaSection, output: Option<&str>) -> cli_helpers::Result {
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
