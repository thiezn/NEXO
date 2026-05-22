#!/usr/bin/env python3
"""HuggingFace model inspector for NEXO manifests.

Fetches file metadata, configs, and generates Rust manifest snippets
from HuggingFace model repositories.

Usage:
    python hf_downloader.py inspect <repo_id> [--filter PATTERN] [--sha256] [--pretty]
    python hf_downloader.py tree <repo_id> [--pretty]
    python hf_downloader.py config <repo_id> [--file NAME] [--all] [--pretty]
    python hf_downloader.py manifest <repo_id> --files F [F ...] [--component-map K=V ...] [--pretty]
    python hf_downloader.py autodetect <repo_id> [--pretty]
    python hf_downloader.py verify --manifest-json PATH [--pretty]

Repo IDs are things like 'openai/whisper-large-v3'
"""

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import fnmatch
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError  # type: ignore

# ── Token resolution ─────────────────────────────────────────────────────────


def load_token() -> str | None:
    # 1. Environment variable
    if token := os.environ.get("HF_TOKEN", "").strip():
        return token

    # 2. Project root hugging_token.txt
    project_root = Path(__file__).resolve().parents[3]
    for name in ("hugging_token.txt", "hf_token.txt"):
        token_path = project_root / name
        if token_path.exists():
            token = token_path.read_text().strip()
            if token:
                return token

    # 3. ~/.nexo/hf_token.txt
    home = Path.home()
    nexo_token = home / ".nexo" / "hf_token.txt"
    if nexo_token.exists():
        token = nexo_token.read_text().strip()
        if token:
            return token

    # 4. huggingface-cli cached token
    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token()
    except Exception:
        return None


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


def get_repo_info(api: HfApi, repo_id: str) -> Any:
    """Fetch model info with clear error messages."""
    try:
        return api.model_info(repo_id, files_metadata=True)
    except GatedRepoError:
        output_json(
            {
                "error": f"Gated repo '{repo_id}' — set HF_TOKEN or request access at https://huggingface.co/{repo_id}",
                "hint": "export HF_TOKEN=hf_... or place token in hugging_token.txt at project root",
            }
        )
        sys.exit(1)
    except RepositoryNotFoundError:
        output_json(
            {
                "error": f"Repository '{repo_id}' not found",
                "hint": f"Check the repo ID at https://huggingface.co/{repo_id}",
            }
        )
        sys.exit(1)


# ── inspect ──────────────────────────────────────────────────────────────────


def cmd_inspect(args: Any) -> None:
    api = HfApi(token=load_token(), endpoint="https://hf-mirror.com")
    info = get_repo_info(api, args.repo_id)

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


# ── tree ─────────────────────────────────────────────────────────────────────


def cmd_tree(args: Any) -> None:
    """Show repo directory structure with file counts and sizes per directory."""
    api = HfApi(token=load_token(), endpoint="https://hf-mirror.com")
    info = get_repo_info(api, args.repo_id)

    if not info.siblings:
        output_json({"error": f"No files found in repo '{args.repo_id}'"})
        sys.exit(1)

    dirs: dict[str, dict[str, Any]] = {}
    for sibling in info.siblings:
        fname = sibling.rfilename
        parts = PurePosixPath(fname)
        parent = str(parts.parent) if str(parts.parent) != "." else "/"
        size = sibling.size or 0

        if parent not in dirs:
            dirs[parent] = {"files": [], "total_bytes": 0}
        dirs[parent]["files"].append({"name": parts.name, "size": size})
        dirs[parent]["total_bytes"] += size

    tree = []
    for dir_path in sorted(dirs.keys()):
        d = dirs[dir_path]
        tree.append(
            {
                "directory": dir_path,
                "file_count": len(d["files"]),
                "total_size": format_human_size(d["total_bytes"]),
                "total_bytes": d["total_bytes"],
                "files": sorted(d["files"], key=lambda f: f["name"]),
            }
        )

    output_json(
        {
            "repo_id": args.repo_id,
            "gated": bool(info.gated),
            "directories": tree,
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

    result: dict[str, Any] = {"repo_id": args.repo_id, "configs": configs}
    if errors:
        result["_errors"] = errors

    output_json(result, pretty=args.pretty)


# ── autodetect ───────────────────────────────────────────────────────────────

# Patterns to auto-classify files into components
COMPONENT_PATTERNS: list[tuple[str, str, list[str]]] = [
    # (component_name, description, glob_patterns)
    (
        "transformer",
        "Model weights (transformer/diffusion)",
        [
            "*.safetensors",
            "model*.safetensors",
            "diffusion_pytorch_model*.safetensors",
        ],
    ),
    (
        "text_encoder",
        "Text encoder weights",
        [
            "text_encoder/*.safetensors",
            "text_encoder/**/*.safetensors",
        ],
    ),
    (
        "vae",
        "VAE weights",
        [
            "vae/*.safetensors",
            "vae/**/*.safetensors",
        ],
    ),
    (
        "tokenizer",
        "Tokenizer files",
        [
            "tokenizer.json",
            "tokenizer/*.json",
            "text_encoder/tokenizer.json",
            "**/tokenizer.json",
        ],
    ),
    (
        "config",
        "Configuration files",
        [
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
        ],
    ),
]


def classify_file(fname: str) -> str | None:
    """Try to classify a file into a component category."""
    parts = PurePosixPath(fname)

    # Subdirectory-based classification (most reliable)
    if len(parts.parts) >= 2:
        subdir = parts.parts[0]
        ext = parts.suffix
        if subdir == "transformer" and ext == ".safetensors":
            return "transformer"
        if subdir == "text_encoder" and ext == ".safetensors":
            return "text_encoder"
        if subdir == "vae" and ext == ".safetensors":
            return "vae"
        if subdir == "tokenizer" and parts.name == "tokenizer.json":
            return "tokenizer"
        if subdir == "text_encoder" and parts.name == "tokenizer.json":
            return "tokenizer"

    # Root-level classification
    name = parts.name
    if name == "tokenizer.json":
        return "tokenizer"
    if name == "config.json":
        return "config"
    if name.endswith(".safetensors") and parts.parent == PurePosixPath("."):
        return "model"

    return None


def cmd_autodetect(args: Any) -> None:
    """Auto-detect model components from repo file structure."""
    api = HfApi(token=load_token(), endpoint="https://hf-mirror.com")
    info = get_repo_info(api, args.repo_id)

    if not info.siblings:
        output_json({"error": f"No files found in repo '{args.repo_id}'"})
        sys.exit(1)

    gated = bool(info.gated)
    components: dict[str, list[dict[str, Any]]] = {}
    unclassified = []

    for sibling in sorted(info.siblings, key=lambda s: s.rfilename):
        fname = sibling.rfilename
        size = sibling.size or 0
        sha256 = sibling.lfs.sha256 if sibling.lfs else None

        category = classify_file(fname)
        entry = {
            "filename": fname,
            "size_bytes": size,
            "size_human": format_human_size(size),
            "sha256": sha256,
        }

        if category:
            components.setdefault(category, []).append(entry)
        else:
            unclassified.append(entry)

    # Determine if sharded
    for comp_name, comp_files in components.items():
        safetensor_count = sum(1 for f in comp_files if f["filename"].endswith(".safetensors"))
        for f in comp_files:
            f["is_shard"] = safetensor_count > 1 and f["filename"].endswith(".safetensors")

    # Compute totals
    total_model_bytes = sum(
        f["size_bytes"] for files in components.values() for f in files if f["filename"].endswith(".safetensors")
    )

    # Generate suggested Rust component mapping
    rust_suggestions = []
    for comp_name, comp_files in sorted(components.items()):
        safetensors = [f for f in comp_files if f["filename"].endswith(".safetensors")]
        is_sharded = len(safetensors) > 1
        rust_comp = "ModelShard" if is_sharded else "Model"
        if comp_name == "tokenizer":
            rust_comp = "Tokenizer"
        elif comp_name == "config":
            rust_comp = "Config"
        elif comp_name in ("text_encoder", "vae") and not is_sharded:
            rust_comp = comp_name.title().replace("_", "")
        for f in comp_files:
            rust_suggestions.append(
                {
                    "filename": f["filename"],
                    "component": rust_comp,
                    "size_bytes": f["size_bytes"],
                    "sha256": f["sha256"],
                    "gated": gated,
                }
            )

    output_json(
        {
            "repo_id": args.repo_id,
            "gated": gated,
            "total_model_size": format_human_size(total_model_bytes),
            "total_model_bytes": total_model_bytes,
            "components": components,
            "unclassified": unclassified if unclassified else None,
            "rust_manifest_suggestion": rust_suggestions,
        },
        pretty=args.pretty,
    )


# ── manifest ─────────────────────────────────────────────────────────────────


def cmd_manifest(args: Any) -> None:
    token = load_token()
    api = HfApi(token=token, endpoint="https://hf-mirror.com")
    info = get_repo_info(api, args.repo_id)

    if not info.siblings:
        output_json({"error": f"No files found in repo '{args.repo_id}'"})
        sys.exit(1)

    gated = bool(info.gated)
    sibling_map = {s.rfilename: s for s in info.siblings}

    # Expand glob patterns in --files
    expanded_files = []
    for pattern in args.files:
        if any(c in pattern for c in "*?["):
            matches = sorted(s.rfilename for s in info.siblings if fnmatch.fnmatch(s.rfilename, pattern))
            if not matches:
                print(f"Warning: glob '{pattern}' matched no files", file=sys.stderr)
            expanded_files.extend(matches)
        else:
            expanded_files.append(pattern)

    # Parse component map
    component_map: dict[str, str] = {}
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

    for fname in expanded_files:
        sibling = sibling_map.get(fname)
        if sibling is None:
            errors.append(f"{fname}: not found in repo")
            continue

        size = sibling.size or 0
        sha256 = None
        if args.sha256 and sibling.lfs:
            sha256 = sibling.lfs.sha256

        # Resolve component: explicit map > auto-classify > filename stem
        stem = Path(fname).stem.split("-")[0].split(".")[0]
        component_name = component_map.get(stem)
        if component_name is None:
            auto = classify_file(fname)
            if auto:
                component_name = auto.title().replace("_", "")
            else:
                component_name = stem.title()

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
        rust_lines.append("    hf_repo: repo.clone(),")
        rust_lines.append(f'    hf_filename: "{fname}".to_string(),')
        rust_lines.append(f"    size_bytes: {format_rust_size(size)},")
        rust_lines.append(f"    gated: {'true' if gated else 'false'},")
        rust_lines.append(f"    sha256: {make_sha256_literal(sha256)},")
        rust_lines.append("},")

    result: dict[str, Any] = {
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
    api = HfApi(token=token, endpoint="https://hf-mirror.com")

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

            r: dict[str, Any] = {
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

    result: dict[str, Any] = {
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

    # tree
    p_tree = subparsers.add_parser("tree", parents=[shared], help="Show repo directory structure")
    p_tree.add_argument("repo_id", help="HuggingFace repo")

    # config
    p_config = subparsers.add_parser("config", parents=[shared], help="Fetch config JSON files from a HF repo")
    p_config.add_argument("repo_id", help="HuggingFace repo")
    p_config.add_argument("--file", default="config.json", help="Config filename (default: config.json)")
    p_config.add_argument("--all", action="store_true", help="Fetch all standard config files")

    # autodetect
    p_auto = subparsers.add_parser("autodetect", parents=[shared], help="Auto-detect model components")
    p_auto.add_argument("repo_id", help="HuggingFace repo")

    # manifest
    p_manifest = subparsers.add_parser("manifest", parents=[shared], help="Generate Rust ModelFile snippet")
    p_manifest.add_argument("repo_id", help="HuggingFace repo")
    p_manifest.add_argument("--files", nargs="+", required=True, help="Files or glob patterns to include")
    p_manifest.add_argument("--component-map", nargs="+", help="Stem=Component pairs (e.g. model=Transformer)")
    p_manifest.add_argument("--component-enum", default="AiComponent", help="Rust enum name (default: AiComponent)")
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
        "tree": cmd_tree,
        "autodetect": cmd_autodetect,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
