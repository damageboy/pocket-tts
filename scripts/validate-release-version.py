#!/usr/bin/env python3
"""Reject release tags that disagree with the publishable Cargo workspace."""

from pathlib import Path
import json
import re
import sys
import tomllib


NUMERIC_IDENTIFIER = r"(?:0|[1-9][0-9]*)"
PRERELEASE_IDENTIFIER = rf"(?:{NUMERIC_IDENTIFIER}|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)"
BUILD_IDENTIFIER = r"[0-9A-Za-z-]+"
RELEASE_TAG = re.compile(
    rf"v{NUMERIC_IDENTIFIER}\.{NUMERIC_IDENTIFIER}\.{NUMERIC_IDENTIFIER}"
    rf"(?:-{PRERELEASE_IDENTIFIER}(?:\.{PRERELEASE_IDENTIFIER})*)?"
    rf"(?:\+{BUILD_IDENTIFIER}(?:\.{BUILD_IDENTIFIER})*)?"
)


def validate(root: Path, tag: str) -> str:
    if not RELEASE_TAG.fullmatch(tag):
        raise ValueError(f"Invalid release version tag: {tag!r}")
    workspace = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))["workspace"]
    version = workspace["package"]["version"]
    if tag != f"v{version}":
        raise ValueError(f"Release tag {tag} does not match Cargo.toml version {version}")

    members = {}
    for member in workspace["members"]:
        manifest_path = root / member / "Cargo.toml"
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        package = manifest["package"]
        actual = package["version"]
        if isinstance(actual, dict) and actual.get("workspace") is True:
            actual = version
        if actual != version:
            raise ValueError(f"{manifest_path}: package version {actual} differs from {version}")
        members[package["name"]] = manifest

    for name, manifest in members.items():
        dependency_sets = [manifest.get("dependencies", {}), manifest.get("build-dependencies", {})]
        for target in manifest.get("target", {}).values():
            dependency_sets.extend([target.get("dependencies", {}), target.get("build-dependencies", {})])
        for dependencies in dependency_sets:
            for dependency_name, dependency in dependencies.items():
                if isinstance(dependency, dict) and dependency.get("workspace"):
                    dependency = workspace.get("dependencies", {}).get(dependency_name, {})
                package_name = dependency.get("package", dependency_name) if isinstance(dependency, dict) else dependency_name
                requirement = dependency.get("version") if isinstance(dependency, dict) else dependency
                if package_name in members and requirement != version:
                    raise ValueError(f"{name}: publishable dependency {package_name} must use version {version}")

    lock = tomllib.loads((root / "Cargo.lock").read_text(encoding="utf-8"))
    local_packages = {p["name"]: p["version"] for p in lock["package"] if "source" not in p}
    for name in members:
        if local_packages.get(name) != version:
            raise ValueError(f"Cargo.lock: {name} must use version {version}")
    return version


if __name__ == "__main__":
    try:
        if len(sys.argv) not in (2, 3):
            raise ValueError("Usage: validate-release-version.py vX.Y.Z [wasm-package-directory]")
        version = validate(Path(__file__).resolve().parent.parent, sys.argv[1])
        if len(sys.argv) == 3:
            metadata = Path(sys.argv[2]) / "package.json"
            if json.loads(metadata.read_text(encoding="utf-8")).get("version") != version:
                raise ValueError(f"{metadata}: WASM package version must be {version}")
    except (ValueError, OSError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    print(f"Release version verified: {version}")
