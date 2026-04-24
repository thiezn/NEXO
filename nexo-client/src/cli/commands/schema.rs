use nexo_ws_schema::SchemaSection;

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
