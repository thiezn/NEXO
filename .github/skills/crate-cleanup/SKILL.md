---
name: crate-cleanup
description: Guidelines for cleaning up and refactoring Rust crates.
disable-model-invocation: true
---

Your task is to clean up, simplify and refactor the described Rust crate or crates.

This may involve removing unused code, simplifying complex functions, improving naming, and ensuring consistency with project conventions. Focus on improving readability and maintainability.

Leverage the /rust skill for any specific questions about Rust code style, best practices, and project-specific guidelines.

Start with launching two sub agents to analyze the crate(s):

- Naming conventions, missing docstring / Documentation for functions, modules and/or parameters, and module layout
- Code complexity, redundancy, and readability
- Missing or incorrect tests

Use the findings to come up with a concrete refactoring plan.

The focus should be on improving the specified crate, but you are encouraged to make cross-cutting improvements across the codebase if you identify common issues or patterns that can be improved in multiple places.

Do not worry about backwards compatibility or breaking changes, as long as the code is improved according to the project's Rust guidelines. This is a personal project and we can break things to make them better.

When evaluating the codebase or code selections, prioritize readable, idiomatic Rust over overly dense or compact "code golf" solutions.

## Simplification Directives

1. **Leverage Language Idioms**: Replace verbose imperative structures with functional iterator chains (`map`, `filter`, `and_then`, `collect`) where it improves readability.
2. **Error Handling & Option Management**: Clean up redundant `match` or `if let` blocks on `Result` and `Option` types by using idiomatic methods like `.unwrap_or_default()`, `.as_ref()`, or the `?` propagation operator.
3. **Pattern Matching**: Simplify overly complex nested conditionals into flat, guard-clause-driven pattern matching.
4. **Ownership and Borrowing**: Elide unnecessary `.clone()` or `.to_owned()` invocations if references can be held instead. Clean up explicit lifetimes if lifetime elision rules allow it.
5. **Redundant Logic**: Remove temporary variables that add zero semantic value. Consolidate overlapping boolean checks.
