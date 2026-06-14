A good way to think about this in Rust is:

* **Library boundaries should usually expose typed errors**
* **Application / binary boundaries can erase into a generic error**
* **`?` works smoothly if you implement `From<OtherError>` for your crate’s error**
* **`thiserror` helps you build typed errors**
* **`anyhow` helps you consume/report errors conveniently at the top level**

That’s the core pattern.

***

# Recommended structure

## 1) Every library crate defines its own `Error`

Yes — your instinct is right.

Each library crate should usually have:

* a crate-specific `Error` enum
* a crate-specific `Result<T>` alias

Example:

```rust
// in my_crate/src/error.rs
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("I/O error")]
    Io(#[from] std::io::Error),

    #[error("failed to parse input")]
    Parse(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
```

Then throughout the crate:

```rust
use crate::error::Result;

pub fn load_stuff() -> Result<()> {
    let _text = std::fs::read_to_string("file.txt")?;
    Ok(())
}
```

Because of `#[from]`, the `?` operator automatically converts `std::io::Error` or `serde_json::Error` into your crate’s `Error`.

***

## 2) A crate that depends on other crates wraps those errors as variants

If `crate_b` depends on `crate_a`, then `crate_b::Error` should typically contain a variant for `crate_a::Error`.

Example:

```rust
// crate_a
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("database failed")]
    Db,
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn handle_thing() -> Result<()> {
    Err(Error::Db)
}
```

```rust
// crate_b
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("crate_a failed")]
    CrateA(#[from] crate_a::Error),

    #[error("other problem")]
    Other,
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn my_func() -> Result<()> {
    crate_a::handle_thing()?; // automatic conversion via #[from]
    Ok(())
}
```

That gives you exactly the ergonomics you want:

```rust
fn my_func() -> Result<()> {
    other_crate1::handle_thing()?;
    other_crate2::handle_thing()?;
    Ok(())
}
```

as long as your local error type has `From<other_crate1::Error>` and `From<other_crate2::Error>` implemented — and `thiserror` generates those for you with `#[from]`.

***

# The most practical pattern

Here’s the pattern I recommend for a workspace with multiple libs and a few bins:

***

## For library crates: use `thiserror`

Use **typed errors** in each library:

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("network request failed")]
    Network(#[from] reqwest::Error),

    #[error("config error: {0}")]
    Config(String),

    #[error("dependency crate failed")]
    Dep(#[from] dep_crate::Error),
}
```

### Why this is good for libraries

A library’s error type is part of its API contract.

Consumers may want to:

* match on error variants
* distinguish validation failures from I/O failures
* retry on some errors but not others
* map internal crate errors into higher-level errors

`anyhow::Error` hides those distinctions.

So for library code, **`thiserror` is usually the right fit**.

***

## For binary crates: use `anyhow` at the top boundary

For binaries, you usually don’t need highly structured matching at the top level. You want:

* easy propagation
* good context
* readable output

That is exactly what `anyhow` is for.

Example:

```rust
use anyhow::{Context, Result};

fn main() -> Result<()> {
    run().context("application startup failed")
}

fn run() -> Result<()> {
    my_library::do_work().context("failed while doing work")?;
    Ok(())
}
```

If `my_library::do_work()` returns `Result<T, my_library::Error>`, it can still be used with `?` in an `anyhow::Result<T>` function, because `anyhow` can wrap any error implementing `std::error::Error`.

So the interop is excellent.

***

# How `thiserror` and `anyhow` fit together

They are not competing solutions so much as **complementary layers**.

## `thiserror`

Use when you want to **define** an error type.

It helps you write:

* `Display`
* `std::error::Error`
* `From<...>` conversions
* source chaining

## `anyhow`

Use when you want to **consume** errors without caring about their exact type.

It gives you:

* a single application-level error type
* `.context(...)` / `.with_context(...)`
* good failure chains for CLI apps or services

### Typical layering

```text
leaf libraries      -> typed errors (`thiserror`)
mid-level libraries -> typed errors (`thiserror`) that wrap lower-level errors
binary/app entry    -> `anyhow::Result<()>`
```

This is by far the most common ergonomic design.

***

# What if a crate is both a library and a binary?

You mentioned:

> a lot of my binary crates are simultaneously also libraries. The binary is just a small wrapper around the library implementation

That’s very common, and the solution is straightforward:

* the **library part** should expose typed errors (`thiserror`)
* the **binary part** should use `anyhow` (or even plain `Result<(), my_lib::Error>` if simple)

Example package layout:

```text
my_app/
  src/
    lib.rs
    error.rs
    main.rs
```

### `src/error.rs`

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("config error: {0}")]
    Config(String),

    #[error("I/O failed")]
    Io(#[from] std::io::Error),

    #[error("dependency failed")]
    Dep(#[from] other_lib::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
```

### `src/lib.rs`

```rust
pub mod error;

use error::Result;

pub fn run_app_logic() -> Result<()> {
    other_lib::do_it()?;
    Ok(())
}
```

### `src/main.rs`

```rust
use anyhow::{Context, Result};

fn main() -> Result<()> {
    my_app::run_app_logic().context("application failed")?;
    Ok(())
}
```

This is a very clean separation:

* `lib.rs` remains reusable, typed, explicit
* `main.rs` remains tiny and ergonomic

***

# When to avoid exposing dependency errors directly

A subtle design choice:

## Option A — directly wrap dependency errors

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("dependency failed")]
    Dep(#[from] other_lib::Error),
}
```

### Pros

* easy
* minimal boilerplate
* nice `?` ergonomics

### Cons

* your public API now exposes `other_lib::Error`
* changing dependency or dependency error shape becomes breaking API churn

***

## Option B — translate dependency errors into your own domain variants

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to initialize storage")]
    StorageInit,

    #[error("failed to read user profile")]
    ProfileRead,

    #[error("internal error")]
    Internal {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
```

Then:

```rust
pub fn my_func() -> Result<()> {
    other_crate::handle_thing()
        .map_err(|source| Error::Internal { source: Box::new(source) })?;
    Ok(())
}
```

### Pros

* your library API is more stable
* errors reflect your domain rather than dependency internals

### Cons

* extra boilerplate
* slightly less convenient than plain `#[from]`

***

## Practical rule of thumb

Use direct wrapping when:

* the dependency is an implementation detail you’re okay exposing
* the crates are internal to one workspace
* you want fast development and ergonomics

Translate into your own domain errors when:

* your library is public/stable
* semver/API cleanliness matters
* you don’t want your public API tied to dependency internals

For many internal workspace crates, direct wrapping is perfectly fine.

***

# How `?` actually works here

The `?` operator in a function returning:

```rust
Result<T, MyError>
```

can only propagate another error if Rust can convert it into `MyError`, usually via:

```rust
impl From<OtherError> for MyError
```

`thiserror` provides that automatically with:

```rust
#[error("...")]
Other(#[from] other_crate::Error)
```

So if your function calls two crates:

```rust
fn my_func() -> Result<(), Error> {
    other_crate1::handle_thing()?;
    other_crate2::handle_thing()?;
    Ok(())
}
```

your `Error` enum needs both conversions:

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("other_crate1 failed")]
    Crate1(#[from] other_crate1::Error),

    #[error("other_crate2 failed")]
    Crate2(#[from] other_crate2::Error),
}
```

That’s the mechanism.

***

# Suggested patterns for different crate layers

## Leaf crate pattern

A leaf crate mostly wraps system / third-party errors:

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error")]
    Io(#[from] std::io::Error),

    #[error("json parse error")]
    Json(#[from] serde_json::Error),

    #[error("invalid format: {0}")]
    InvalidFormat(String),
}
```

***

## Middle-layer crate pattern

A crate composes lower-level crates:

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("auth failed")]
    Auth(#[from] auth_lib::Error),

    #[error("storage failed")]
    Storage(#[from] storage_lib::Error),

    #[error("invalid request: {0}")]
    InvalidRequest(String),
}
```

***

## Binary crate pattern

The binary just adds context and prints nicely:

```rust
use anyhow::{Context, Result};

fn main() -> Result<()> {
    run().context("top-level execution failed")
}

fn run() -> Result<()> {
    service_lib::serve().context("service failed")?;
    Ok(())
}
```

***

# If you want to use only one crate

You said you’d prefer to use only one if possible.

## If you choose only `thiserror`

This is totally viable.

* libraries use typed errors
* binaries can return `Result<(), my_lib::Error>` or some top-level app error enum

Example:

```rust
fn main() -> Result<(), my_lib::Error> {
    my_lib::run()
}
```

This is simplest if your binary is tiny.

### Downsides

You lose:

* easy `.context(...)`
* ergonomic ad hoc error aggregation in binaries
* very convenient top-level reporting

***

## If you choose only `anyhow`

This can work for **private/internal code**, but I would be cautious for reusable libraries.

### Downsides for libraries

* callers can’t pattern match meaningfully
* API becomes less self-documenting
* hidden error contracts
* harder to distinguish expected vs unexpected failures

So: **only-anyhow is fine for apps, not ideal for libraries**.

***

# My recommendation for your setup

Given your description — multiple library crates, some binary wrappers, and some packages that are both library + binary — I would use:

## Best overall approach

* **Library crates:** `thiserror`
* **Binary entrypoints (`main.rs`):** `anyhow`

That means:

* each library has a crate-specific `Error`
* each higher-level library wraps lower-level crate errors with `#[from]`
* each binary uses `anyhow::Result<()>`
* the binary adds user-facing context with `.context(...)`

This gives you:

* explicit reusable library APIs
* easy `?` propagation across many crates
* pleasant top-level CLI/service ergonomics

***

# A concrete end-to-end example

## `other_crate1`

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("thing 1 failed")]
    Thing1Failed,
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn handle_thing() -> Result<()> {
    Err(Error::Thing1Failed)
}
```

## `other_crate2`

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("thing 2 failed")]
    Thing2Failed,
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn handle_thing() -> Result<()> {
    Err(Error::Thing2Failed)
}
```

## `my_lib`

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("other crate 1 error")]
    OtherCrate1(#[from] other_crate1::Error),

    #[error("other crate 2 error")]
    OtherCrate2(#[from] other_crate2::Error),

    #[error("my own domain error: {0}")]
    Domain(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn my_func() -> Result<()> {
    other_crate1::handle_thing()?;
    other_crate2::handle_thing()?;
    Ok(())
}
```

## `my_bin`

```rust
use anyhow::{Context, Result};

fn main() -> Result<()> {
    my_lib::my_func().context("failed to run my_func")
}
```

That’s clean, idiomatic, and scalable.

***

# Extra tips

## Use `#[error(transparent)]` for pass-through wrappers

If a variant is just forwarding another error:

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Other(#[from] other_crate::Error),
}
```

This preserves the wrapped error’s display output nicely.

Use it when you don’t need a custom message for that variant.

***

## Add domain variants, not just wrapped variants

Don’t make everything “some other crate failed.”  
Keep your own meaningful variants too:

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("user id was missing")]
    MissingUserId,

    #[error("storage failed")]
    Storage(#[from] storage::Error),
}
```

That makes the crate API more expressive.

***

## Consider API stability for public crates

If a crate is public and semver-sensitive, exposing dependency errors directly may become awkward. In those cases, consider translating into your own variants instead of re-exporting dependency details.

For internal workspace crates, I’d optimize for ergonomics unless you already know you need strict abstraction.

***

# Short answer

If I were designing your workspace, I’d do this:

* **Each library crate:** define `pub enum Error` + `pub type Result<T>`
* **Use `thiserror`** in those libraries
* **Wrap lower-level crate errors as variants** with `#[from]` so `?` works automatically
* **At the binary boundary**, use `anyhow::Result<()>` and `.context(...)`
* **For packages that are both lib + bin**, the library side stays typed, and the `main.rs` side uses `anyhow`

That’s the idiomatic “best of both worlds” setup.

If you want, I can also give you:

1. a **recommended workspace-wide error template** for all crates, or
2. a **small example workspace layout** with 2 libs and 1 binary showing this pattern end-to-end.
