use crate::cli::SchemaTarget;
use nexo_ws_schema::SchemaSection;

pub fn run_schema(target: SchemaTarget, output: Option<&str>) -> utl_helpers::Result {
    let section = match target {
        SchemaTarget::All => SchemaSection::All,
        SchemaTarget::Frames => SchemaSection::Frames,
        SchemaTarget::Connect => SchemaSection::Connect,
        SchemaTarget::Methods => SchemaSection::Methods,
        SchemaTarget::Events => SchemaSection::Events,
        SchemaTarget::Errors => SchemaSection::Errors,
    };

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
