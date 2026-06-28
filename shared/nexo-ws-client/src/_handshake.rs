use crate::connection::NexoConnection;
use crate::error::{Error, Result};
use nexo_core::ClientKind;
use nexo_ws_schema::{ConnectParams, Frame, HelloOk, Method, PROTOCOL_VERSION};
use tracing::info;

/// Perform the connect handshake with the gateway.
///
/// Sends a `connect` request with the given params and waits for a `hello-ok` response.
/// Validates the negotiated protocol version.
pub async fn perform_handshake(conn: &mut NexoConnection, role: ClientKind) -> Result<HelloOk> {
    let request_id = Frame::new_id();
    let frame = Frame::Request {
        id: request_id.clone(),
        method: Method::Connect,
        params: serde_json::to_value(&role)?,
    };

    conn.send_frame(&frame).await?;
    tracing::debug!("Sent connect request (id={request_id})");

    let response = conn.recv_frame().await?;
    // .ok_or_else(|| Error::Handshake("Connection closed before hello-ok".into()))?;

    match response {
        Frame::Response {
            id,
            ok: true,
            payload: Some(payload),
            ..
        } if id == request_id => {
            let hello: HelloOk = serde_json::from_value(payload)?;

            if hello.protocol < role.min_protocol || hello.protocol > role.max_protocol {
                return Err(Error::Protocol(nexo_ws_schema::Error::ProtocolMismatch {
                    min: role.min_protocol,
                    max: role.max_protocol,
                    server: hello.protocol,
                }));
            }

            info!(
                protocol = hello.protocol,
                tick_interval_ms = hello.policy.tick_interval_ms,
                "Handshake complete"
            );
            Ok(hello)
        }
        Frame::Response {
            ok: false, error, ..
        } => {
            let msg = error
                .map(|e| format!("{}: {}", e.code, e.message))
                .unwrap_or_else(|| "Unknown error".into());
            Err(Error::Handshake(msg))
        }
        other => Err(Error::Handshake(format!(
            "Unexpected response frame: {other:?}"
        ))),
    }
}

/// Build default `ConnectParams` for a node-role capability host.
pub fn default_node_connect_params(
    client_id: &str,
    version: &str,
    platform: nexo_ws_schema::Platform,
    device_id: &str,
    capabilities: Vec<String>,
    commands: Vec<String>,
    models: Vec<String>,
) -> ConnectParams {
    ConnectParams {
        min_protocol: PROTOCOL_VERSION,
        max_protocol: PROTOCOL_VERSION,
        client: nexo_ws_schema::ClientInfo {
            id: client_id.to_string(),
            version: version.to_string(),
            platform,
        },
        role: nexo_ws_schema::ConnectionRole::Node,
        scopes: vec![],
        capabilities,
        commands,
        models,
        locale: Some("en-US".to_string()),
        user_agent: Some(format!("NEXO-{client_id}/{version}")),
        device: Some(nexo_ws_schema::DeviceInfo {
            id: device_id.to_string(),
        }),
    }
}

/// Build default `ConnectParams` for a user-role CLI client.
pub fn default_user_connect_params(
    client_id: &str,
    version: &str,
    platform: nexo_ws_schema::Platform,
    device_id: &str,
) -> ConnectParams {
    ConnectParams {
        min_protocol: PROTOCOL_VERSION,
        max_protocol: PROTOCOL_VERSION,
        client: nexo_ws_schema::ClientInfo {
            id: client_id.to_string(),
            version: version.to_string(),
            platform,
        },
        role: nexo_ws_schema::ConnectionRole::User,
        scopes: vec![
            nexo_ws_schema::Scope::UserRead,
            nexo_ws_schema::Scope::UserWrite,
        ],
        capabilities: vec![],
        commands: vec![],
        models: vec![],
        locale: Some("en-US".to_string()),
        user_agent: Some(format!("NEXO-{client_id}/{version}")),
        device: Some(nexo_ws_schema::DeviceInfo {
            id: device_id.to_string(),
        }),
    }
}
