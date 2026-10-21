"""Usage: python3 -m research {groups,compare,check,overview,find,section,catalog}."""

import argparse
import json
import os
import sys
from pathlib import Path

from research.comparison import EvidenceError, compare, digest, load_registry, overview, render
from research.documents import index_markdown, inventory, search, section
from research.library import load_library, select_experiments, experiment_markdown, load_saved_result, saved_result_markdown


REPOSITORY = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY)
    parser.add_argument("--runs-root", type=Path, default=os.environ.get("CONTEXT_STR_RUNS_ROOT"))
    parser.add_argument("--artifacts-root", type=Path, default=os.environ.get("CONTEXT_STR_ARTIFACTS_ROOT"))
    parser.add_argument("--registry", type=Path, help="Comparison registry (default: research/experiments.json under repo root)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("groups", help="List available curated comparisons, without reading run directories")
    comparison = commands.add_parser("compare", help="Read one pinned protocol group; never launch inference")
    comparison.add_argument("group")
    comparison.add_argument("--format", choices=("markdown", "json", "csv"), default="markdown")
    comparison.add_argument("--sort", choices=("f1", "precision", "recall", "tn0", "tp"))
    comparison.add_argument("--output", type=Path, help="Write a NEW export file; existing files are protected")
    commands.add_parser("check", help="Validate all registered groups; report every failure")
    summary = commands.add_parser("overview", help="Render the curated model overview as separate tables")
    summary.add_argument("--output", type=Path, help="Write a NEW Markdown export; existing files are protected")
    summary.add_argument("--check", type=Path, help="Verify an existing overview without writing")
    find = commands.add_parser("find", help="Search evidence documents, excluding sample caches")
    find.add_argument("query")
    find.add_argument("--limit", type=int, default=20)
    read = commands.add_parser("section", help="Read one original Markdown section")
    read.add_argument("query")
    read.add_argument("--file", help="Disambiguate with a path substring")
    catalog = commands.add_parser("catalog", help="Build or verify the bounded audit inventory")
    catalog.add_argument("--check", action="store_true", help="Check both generated index files without writing")
    catalog.add_argument("--source-only", action="store_true", help="Index repository documentation only, excluding local/external run artifacts")
    library = commands.add_parser("experiments", help="Browse experiments by family; no external data needed")
    library.add_argument("--family")
    library.add_argument("--query")
    library.add_argument("--format", choices=("markdown", "json"), default="markdown")
    experiment = commands.add_parser("experiment", help="Show an experiment's guide, code, configs, tests and results")
    experiment.add_argument("id")
    saved = commands.add_parser("results", help="Read a verified saved numerical summary without restoring raw runs")
    saved.add_argument("group")
    saved.add_argument("--format", choices=("markdown", "json", "csv"), default="markdown")
    args = parser.parse_args(argv)
    roots = dict(repository=args.repo_root.resolve(), runs=args.runs_root, external_artifacts=args.artifacts_root)
    try:
        if args.command in ("experiments", "experiment"):
            library = load_library(args.repo_root)
            if args.command == "experiment":
                rows = [row for row in library["experiments"] if row["id"] == args.id]
                if len(rows) != 1:
                    raise EvidenceError("Unknown experiment; run 'python3 -m research experiments'")
                print(experiment_markdown(rows[0]), end="")
            else:
                rows = select_experiments(library, args.family, args.query)
                if args.format == "json":
                    print(json.dumps(rows, indent=2, ensure_ascii=False))
                else:
                    print("| Family | Experiment ID | Guide | Numerical groups |\n| --- | --- | --- | --- |")
                    for row in rows:
                        print("| {} | {} | {} | {} |".format(row["family"], row["id"], row["guide"], ", ".join(row["result_groups"]) or "See guide"))
            return 0
        if args.command == "results":
            report = load_saved_result(args.repo_root, args.group)
            print(saved_result_markdown(report) if args.format == "markdown" else render(report, args.format), end="")
            return 0
        if args.command == "find":
            if args.limit < 1 or not args.query.strip():
                raise EvidenceError("Search requires a nonempty query and positive limit")
            hits = search(roots, args.query, args.limit)
            print("\n".join(hits) if hits else "No matches in the selected evidence scope")
            return 0
        if args.command == "section":
            print(section(roots, args.query, args.file), end="")
            return 0
        if args.command == "catalog":
            if args.source_only:
                roots = dict(repository=args.repo_root.resolve(), source_only=True)
            index = inventory(roots)
            outputs = {
                args.repo_root / "docs/AUDIT_INDEX.json": json.dumps(index, indent=2, ensure_ascii=False) + "\n",
                args.repo_root / "docs/AUDIT_INDEX.md": index_markdown(index, roots),
            }
            if args.check:
                stale = [str(p) for p, content in outputs.items() if not p.is_file() or p.read_text(encoding="utf-8") != content]
                if stale:
                    raise EvidenceError("Stale/missing audit index; regenerate with the SAME root options: " + ", ".join(stale))
            else:
                for path, content in outputs.items():
                    path.write_text(content, encoding="utf-8")
            print("{} evidence documents; catalog {}".format(len(index["entries"]), "verified" if args.check else "written"))
            return 0
        registry_path = args.registry or args.repo_root / "research/experiments.json"
        registry = load_registry(registry_path)
        if args.command == "groups":
            for group in registry["groups"]:
                print("{}\t{}\t{}\t{}".format(group["id"], group["split"], group["metric"], group["title"]))
            return 0
        if args.command == "check":
            failures = []
            for group in registry["groups"]:
                try:
                    report = compare(group, roots)
                    print("PASS {} ({} rows)".format(group["id"], len(report["rows"])))
                except (EvidenceError, OSError, ValueError, KeyError, TypeError) as exc:
                    failures.append(group["id"])
                    print("FAIL {}: {}".format(group["id"], exc), file=sys.stderr)
            print("{} passed; {} failed".format(len(registry["groups"]) - len(failures), len(failures)))
            return 1 if failures else 0
        if args.command == "overview":
            if args.output and args.check:
                raise EvidenceError("Choose either --output or --check")
            output = overview(registry, roots)
            if args.check:
                if args.check.read_text(encoding="utf-8") != output:
                    raise EvidenceError("Stale overview; render a fresh export and review the difference")
                print("Overview verified: " + str(args.check))
            elif args.output:
                with args.output.open("x", encoding="utf-8", newline="") as stream:
                    stream.write(output)
                print("Exported " + str(args.output))
            else:
                print(output, end="")
            return 0
        selected = [group for group in registry["groups"] if group["id"] == args.group]
        if not selected:
            raise EvidenceError("Unknown group '{}'; run 'python3 -m research groups'".format(args.group))
        report = compare(selected[0], roots)
        report["registry_sha256"] = digest(registry_path.read_bytes())
        if args.sort:
            report["rows"].sort(key=lambda row: (row[args.sort] is not None, row[args.sort] or 0), reverse=True)
        output = render(report, args.format)
        if args.output:
            # Exclusive creation protects source reports, registries and previous exports.
            with args.output.open("x", encoding="utf-8", newline="") as stream:
                stream.write(output)
            print("Exported " + str(args.output))
        else:
            print(output, end="")
        return 0
    except (EvidenceError, OSError, ValueError, KeyError, TypeError) as exc:
        print("Evidence error: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
