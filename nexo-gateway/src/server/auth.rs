/// Validate the X-NEXO-AUTH header from the HTTP upgrade request.
pub fn validate_auth(headers: &http::HeaderMap, expected_token: &str) -> bool {
    headers
        .get(nexo_ws_schema::AUTH_HEADER)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v == expected_token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_auth_accepted() {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            nexo_ws_schema::AUTH_HEADER,
            http::HeaderValue::from_static(nexo_ws_schema::AUTH_TOKEN),
        );
        assert!(validate_auth(&headers, nexo_ws_schema::AUTH_TOKEN));
    }

    #[test]
    fn invalid_auth_rejected() {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            nexo_ws_schema::AUTH_HEADER,
            http::HeaderValue::from_static("wrong_token"),
        );
        assert!(!validate_auth(&headers, nexo_ws_schema::AUTH_TOKEN));
    }

    #[test]
    fn missing_auth_rejected() {
        let headers = http::HeaderMap::new();
        assert!(!validate_auth(&headers, nexo_ws_schema::AUTH_TOKEN));
    }
}
