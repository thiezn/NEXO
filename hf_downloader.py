#!/usr/bin/env python3
"""HuggingFace model inspector for NEXO manifests.

Fetches file metadata, configs, and generates Rust manifest snippets
from HuggingFace model repositories.

Usage:
    python hf_downloader.py inspect <repo_id> [--filter PATTERN] [--sha256] [--pretty]
    python hf_downloader.py config <repo_id> [--file NAME] [--all] [--pretty]
    python hf_downloader.py manifest <repo_id> --files F [F ...] [--component-map K=V ...] [--pretty]
    python hf_downloader.py verify --manifest-json PATH [--pretty]

Repo IDs are things like 'openai/whisper-large-v3'
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError  # type: ignore

# ── Token resolution ─────────────────────────────────────────────────────────


def load_token() -> str | None:
    token_path = Path(__file__).resolve().parent / "hugging_token.txt"
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            return token
    return os.environ.get("HF_TOKEN")


# ── Helpers ──────────────────────────────────────────────────────────────────


def format_human_size(n: int) -> str:
    for unit, threshold in [("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]:
        if n >= threshold:
            return f"{n / threshold:.2f} {unit}"
    return f"{n} B"


def format_rust_size(n: int) -> str:
    return f"{n:_}"


def make_sha256_literal(sha: str | None) -> str:
    if sha is None:
        return "None"
    return f'Some("{sha}")'


def output_json(data: dict[str, Any], pretty: bool = False) -> None:
    indent = 2 if pretty else None
    json.dump(data, sys.stdout, indent=indent, ensure_ascii=False)
    sys.stdout.write("\n")


# ── inspect ──────────────────────────────────────────────────────────────────


def cmd_inspect(args: Any) -> None:
    token = load_token()
    api = HfApi(token=token)

    try:
        info = api.model_info(args.repo_id, files_metadata=True)
    except GatedRepoError:
        output_json({"error": f"Gated repo '{args.repo_id}' — token required or access not granted"})
        sys.exit(1)
    except RepositoryNotFoundError:
        output_json({"error": f"Repository '{args.repo_id}' not found"})
        sys.exit(1)

    gated = bool(info.gated)
    files = []

    if not info.siblings:
        output_json({"error": f"No files found in repo '{args.repo_id}'"})
        sys.exit(1)

    for sibling in sorted(info.siblings, key=lambda s: s.rfilename):
        fname = sibling.rfilename
        if args.filter and not fnmatch.fnmatch(fname, args.filter):
            continue

        is_lfs = sibling.lfs is not None
        sha256 = None
        if args.sha256 and is_lfs and sibling.lfs:
            sha256 = sibling.lfs.sha256

        size = sibling.size or 0
        entry = {
            "filename": fname,
            "size_bytes": size,
            "size_human": format_human_size(size),
            "is_lfs": is_lfs,
        }
        if args.sha256:
            entry["sha256"] = sha256
        files.append(entry)

    total = sum(f["size_bytes"] for f in files)
    output_json(
        {
            "repo_id": args.repo_id,
            "gated": gated,
            "total_size_bytes": total,
            "total_size_human": format_human_size(total),
            "file_count": len(files),
            "files": files,
        },
        pretty=args.pretty,
    )


# ── config ───────────────────────────────────────────────────────────────────

STANDARD_CONFIGS = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
]


def cmd_config(args: Any) -> None:
    token = load_token()

    if args.all:
        filenames = STANDARD_CONFIGS
    else:
        filenames = [args.file]

    configs = {}
    errors = []

    for fname in filenames:
        try:
            local_path = hf_hub_download(args.repo_id, fname, token=token)
            with open(local_path) as f:
                configs[fname] = json.load(f)
        except EntryNotFoundError:
            configs[fname] = None
            errors.append(f"{fname}: not found in repo")
        except GatedRepoError:
            output_json({"error": f"Gated repo '{args.repo_id}' — token required or access not granted"})
            sys.exit(1)
        except RepositoryNotFoundError:
            output_json({"error": f"Repository '{args.repo_id}' not found"})
            sys.exit(1)
        except json.JSONDecodeError:
            configs[fname] = None
            errors.append(f"{fname}: not valid JSON")

    result = {"repo_id": args.repo_id, "configs": configs}
    if errors:
        result["_errors"] = errors

    output_json(result, pretty=args.pretty)


# ── manifest ─────────────────────────────────────────────────────────────────


def cmd_manifest(args: Any) -> None:
    token = load_token()
    api = HfApi(token=token)

    try:
        info = api.model_info(args.repo_id, files_metadata=True)
    except (GatedRepoError, RepositoryNotFoundError) as e:
        output_json({"error": str(e)})
        sys.exit(1)

    if not info.siblings:
        output_json({"error": f"No files found in repo '{args.repo_id}'"})
        sys.exit(1)

    gated = bool(info.gated)
    sibling_map = {s.rfilename: s for s in info.siblings}

    # Parse component map
    component_map = {}
    if args.component_map:
        for pair in args.component_map:
            if "=" not in pair:
                print(f"Warning: ignoring invalid component-map entry '{pair}' (expected key=Value)", file=sys.stderr)
                continue
            key, val = pair.split("=", 1)
            component_map[key] = val

    enum_name = args.component_enum
    files_data = []
    rust_lines = []
    errors = []

    for fname in args.files:
        sibling = sibling_map.get(fname)
        if sibling is None:
            errors.append(f"{fname}: not found in repo")
            continue

        size = sibling.size or 0
        sha256 = None
        if args.sha256 and sibling.lfs:
            sha256 = sibling.lfs.sha256

        # Resolve component name from map or filename stem
        stem = Path(fname).stem.split("-")[0].split(".")[0]
        component_name = component_map.get(stem, stem.title())

        files_data.append(
            {
                "filename": fname,
                "size_bytes": size,
                "sha256": sha256,
                "component": component_name,
            }
        )

        rust_lines.append("ModelFile {")
        rust_lines.append(f"    component: {enum_name}::{component_name},")
        rust_lines.append(f'    hf_repo: "{args.repo_id}".to_string(),')
        rust_lines.append(f'    hf_filename: "{fname}".to_string(),')
        rust_lines.append(f"    size_bytes: {format_rust_size(size)},")
        rust_lines.append(f"    gated: {'true' if gated else 'false'},")
        rust_lines.append(f"    sha256: {make_sha256_literal(sha256)},")
        rust_lines.append("},")

    result = {
        "repo_id": args.repo_id,
        "gated": gated,
        "files": files_data,
        "rust_code": "\n".join(rust_lines),
    }
    if errors:
        result["_errors"] = errors

    output_json(result, pretty=args.pretty)


# ── verify ───────────────────────────────────────────────────────────────────


def cmd_verify(args: Any) -> None:
    token = load_token()
    api = HfApi(token=token)

    with open(args.manifest_json) as f:
        entries = json.load(f)

    # Group entries by repo to minimize API calls
    by_repo: dict[str, list[dict]] = {}
    for entry in entries:
        repo = entry["hf_repo"]
        by_repo.setdefault(repo, []).append(entry)

    results = []
    repo_errors = []

    for repo_id, repo_entries in by_repo.items():
        try:
            info = api.model_info(repo_id, files_metadata=True)
        except (GatedRepoError, RepositoryNotFoundError) as e:
            repo_errors.append(f"{repo_id}: {e}")
            for entry in repo_entries:
                results.append(
                    {
                        "hf_repo": repo_id,
                        "hf_filename": entry["hf_filename"],
                        "status": "repo_error",
                        "error": str(e),
                    }
                )
            continue

        if not info.siblings:
            output_json({"error": f"No files found in repo '{repo_id}'"})
            sys.exit(1)

        sibling_map = {s.rfilename: s for s in info.siblings}

        for entry in repo_entries:
            fname = entry["hf_filename"]
            expected_size = entry.get("expected_size_bytes")
            expected_sha = entry.get("expected_sha256")

            sibling = sibling_map.get(fname)
            if sibling is None:
                results.append(
                    {
                        "hf_repo": repo_id,
                        "hf_filename": fname,
                        "status": "missing",
                        "expected_size": expected_size,
                    }
                )
                continue

            actual_size = sibling.size or 0
            actual_sha = sibling.lfs.sha256 if sibling.lfs else None

            r = {
                "hf_repo": repo_id,
                "hf_filename": fname,
                "status": "ok",
                "expected_size": expected_size,
                "actual_size": actual_size,
            }

            if expected_size is not None and expected_size != actual_size:
                r["status"] = "size_mismatch"
                r["diff_bytes"] = actual_size - expected_size
                r["diff_human"] = format_human_size(abs(actual_size - expected_size))

            if expected_sha and actual_sha and expected_sha != actual_sha:
                r["status"] = "sha256_mismatch"
                r["expected_sha256"] = expected_sha
                r["actual_sha256"] = actual_sha

            results.append(r)

    passed = sum(1 for r in results if r["status"] == "ok")
    failed = len(results) - passed

    result = {
        "total_checked": len(results),
        "passed": passed,
        "failed": failed,
        "results": results,
    }
    if repo_errors:
        result["_repo_errors"] = repo_errors

    output_json(result, pretty=args.pretty)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = ArgumentParser(description="HuggingFace model inspector for NEXO manifests")
    shared = ArgumentParser(add_help=False)
    shared.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_inspect = subparsers.add_parser("inspect", parents=[shared], help="List files in a HF repo with exact sizes")
    p_inspect.add_argument("repo_id", help="HuggingFace repo (e.g. openai/whisper-large-v3)")
    p_inspect.add_argument("--filter", help="Glob pattern to filter files (e.g. '*.safetensors')")
    p_inspect.add_argument("--sha256", action="store_true", help="Include SHA-256 hashes from LFS")

    # config
    p_config = subparsers.add_parser("config", parents=[shared], help="Fetch config JSON files from a HF repo")
    p_config.add_argument("repo_id", help="HuggingFace repo")
    p_config.add_argument("--file", default="config.json", help="Config filename (default: config.json)")
    p_config.add_argument("--all", action="store_true", help="Fetch all standard config files")

    # manifest
    p_manifest = subparsers.add_parser("manifest", parents=[shared], help="Generate Rust ModelFile snippet")
    p_manifest.add_argument("repo_id", help="HuggingFace repo")
    p_manifest.add_argument("--files", nargs="+", required=True, help="Files to include")
    p_manifest.add_argument("--component-map", nargs="+", help="Stem=Component pairs (e.g. model=Transformer)")
    p_manifest.add_argument("--component-enum", default="Component", help="Rust enum name (default: Component)")
    p_manifest.add_argument("--sha256", action="store_true", help="Include SHA-256 hashes")

    # verify
    p_verify = subparsers.add_parser("verify", parents=[shared], help="Verify manifest data against HF")
    p_verify.add_argument("--manifest-json", required=True, help="JSON file with entries to verify")

    args = parser.parse_args()

    commands = {
        "inspect": cmd_inspect,
        "config": cmd_config,
        "manifest": cmd_manifest,
        "verify": cmd_verify,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
