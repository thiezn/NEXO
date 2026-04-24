use crate::error::{Error, Result};
use std::fmt::Display;
use std::str::FromStr;

/// Prompt for text input with an optional default value.
pub fn text_input(prompt: &str, default: Option<&str>) -> Result<String> {
    let mut input = dialoguer::Input::<String>::new().with_prompt(prompt);
    if let Some(d) = default {
        input = input.default(d.to_string());
    }
    input
        .interact_text()
        .map_err(|e| Error::Io(format!("Input failed: {e}")))
}

/// Prompt for text input that is validated to be non-empty.
pub fn text_input_required(prompt: &str) -> Result<String> {
    dialoguer::Input::<String>::new()
        .with_prompt(prompt)
        .validate_with(|input: &String| -> std::result::Result<(), &str> {
            if input.trim().is_empty() {
                Err("Value cannot be empty")
            } else {
                Ok(())
            }
        })
        .interact_text()
        .map_err(|e| Error::Io(format!("Input failed: {e}")))
}

/// Prompt user to select one item from a list. Returns the selected index.
pub fn select(prompt: &str, items: &[&str], default: Option<usize>) -> Result<usize> {
    let mut sel = dialoguer::Select::new().with_prompt(prompt).items(items);
    if let Some(d) = default {
        sel = sel.default(d);
    }
    sel.interact()
        .map_err(|e| Error::Io(format!("Selection failed: {e}")))
}

/// Prompt for yes/no confirmation with an optional default.
pub fn confirm(prompt: &str, default: Option<bool>) -> Result<bool> {
    let mut c = dialoguer::Confirm::new().with_prompt(prompt);
    if let Some(d) = default {
        c = c.default(d);
    }
    c.interact()
        .map_err(|e| Error::Io(format!("Confirm failed: {e}")))
}

/// Prompt for a numeric value with an optional default.
/// Works with any type that implements `FromStr + Display`.
pub fn number_input<T>(prompt: &str, default: Option<T>) -> Result<T>
where
    T: FromStr + Display + Clone,
    T::Err: Display,
{
    let mut input = dialoguer::Input::<String>::new().with_prompt(prompt);
    if let Some(ref d) = default {
        input = input.default(d.to_string());
    }
    let raw = input
        .validate_with(|v: &String| -> std::result::Result<(), String> {
            v.parse::<T>()
                .map(|_| ())
                .map_err(|e| format!("Invalid number: {e}"))
        })
        .interact_text()
        .map_err(|e| Error::Io(format!("Input failed: {e}")))?;

    raw.parse::<T>()
        .map_err(|e| Error::Other(format!("Parse failed: {e}")))
}

/// Prompt for a password/secret (hidden input).
pub fn password_input(prompt: &str) -> Result<String> {
    dialoguer::Password::new()
        .with_prompt(prompt)
        .interact()
        .map_err(|e| Error::Io(format!("Password input failed: {e}")))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    #[test]
    fn number_parse_validation_rejects_non_numeric() {
        let result = "abc".parse::<u16>();
        assert!(result.is_err());
    }

    #[test]
    fn number_parse_validation_accepts_valid() {
        let result = "6969".parse::<u16>();
        assert_eq!(result.unwrap(), 6969);
    }

    #[test]
    fn number_parse_u64_accepts_large_value() {
        let result = "15000".parse::<u64>();
        assert_eq!(result.unwrap(), 15000);
    }
}
