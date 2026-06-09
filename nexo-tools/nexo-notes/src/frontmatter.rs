use chrono::{DateTime, NaiveDateTime, Utc};

pub(crate) const DATETIME_FORMAT: &str = "%Y-%m-%dT%H:%M:%S";

/// Custom serde serialization module for matching the exact datetime format
pub(crate) mod frontmatter_date {
    use super::{DATETIME_FORMAT, DateTime, NaiveDateTime, Utc};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(date: &DateTime<Utc>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = format!("{}", date.format(DATETIME_FORMAT));
        serializer.serialize_str(&s)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<DateTime<Utc>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let naive_dt =
            NaiveDateTime::parse_from_str(&s, DATETIME_FORMAT).map_err(serde::de::Error::custom)?;

        Ok(DateTime::<Utc>::from_naive_utc_and_offset(naive_dt, Utc))
    }
}
