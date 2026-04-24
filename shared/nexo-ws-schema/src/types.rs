use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Connection role: user (control-plane client) or node (capability host).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Node,
}

impl Role {
    pub fn as_str(&self) -> &str {
        match self {
            Self::User => "user",
            Self::Node => "node",
        }
    }
}

/// Authorization scopes for user-role connections.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
pub enum Scope {
    #[serde(rename = "user.read")]
    UserRead,
    #[serde(rename = "user.write")]
    UserWrite,
    #[serde(rename = "user.admin")]
    UserAdmin,
}

/// Platform the client is running on.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Platform {
    Macos,
    Ios,
    Linux,
    Windows,
    Mortimmy,
}

impl Platform {
    /// Detect the current platform from `std::env::consts::OS`.
    pub fn current() -> Self {
        match std::env::consts::OS {
            "macos" => Self::Macos,
            "ios" => Self::Ios,
            "linux" => Self::Linux,
            "windows" => Self::Windows,
            "mortimmy" => Self::Mortimmy,
            _ => Self::Macos,
        }
    }
}

/// Client identity included in the connect handshake.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ClientInfo {
    pub id: String,
    pub version: String,
    pub platform: Platform,
}

/// Stable device identity for pairing.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct DeviceInfo {
    pub id: String,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn role_serialization() {
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(serde_json::to_string(&Role::Node).unwrap(), "\"node\"");
    }

    #[test]
    fn role_deserialization() {
        let r: Role = serde_json::from_str("\"node\"").unwrap();
        assert_eq!(r, Role::Node);
    }

    #[test]
    fn scope_serialization() {
        assert_eq!(
            serde_json::to_string(&Scope::UserRead).unwrap(),
            "\"user.read\""
        );
        assert_eq!(
            serde_json::to_string(&Scope::UserWrite).unwrap(),
            "\"user.write\""
        );
        assert_eq!(
            serde_json::to_string(&Scope::UserAdmin).unwrap(),
            "\"user.admin\""
        );
    }

    #[test]
    fn scope_deserialization() {
        let s: Scope = serde_json::from_str("\"user.write\"").unwrap();
        assert_eq!(s, Scope::UserWrite);
    }

    #[test]
    fn platform_roundtrip() {
        for platform in [
            Platform::Macos,
            Platform::Ios,
            Platform::Linux,
            Platform::Windows,
            Platform::Mortimmy,
        ] {
            let json = serde_json::to_string(&platform).unwrap();
            let decoded: Platform = serde_json::from_str(&json).unwrap();
            assert_eq!(platform, decoded);
        }
    }

    #[test]
    fn client_info_serialization() {
        let info = ClientInfo {
            id: "cli".into(),
            version: "1.0.0".into(),
            platform: Platform::Macos,
        };
        let json = serde_json::to_value(&info).unwrap();
        assert_eq!(json["id"], "cli");
        assert_eq!(json["platform"], "macos");
    }

    #[test]
    fn device_info_roundtrip() {
        let device = DeviceInfo {
            id: "abc123".into(),
        };
        let json = serde_json::to_string(&device).unwrap();
        let decoded: DeviceInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(device, decoded);
    }
}
