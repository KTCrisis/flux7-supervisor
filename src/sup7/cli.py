"""CLI entry point for sup7."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from sup7.config import load_config
from sup7.runner import SupervisorRunner


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx logs every request URL at INFO; the Cloudflare one carries the account id
    logging.getLogger("httpx").setLevel(logging.WARNING)


def cmd_start(args: argparse.Namespace) -> None:
    config = load_config(args.config)

    setup_logging(args.verbose)

    runner = SupervisorRunner(config)
    asyncio.run(runner.start())


def cmd_status(args: argparse.Namespace) -> None:
    from mesh7 import AgentMesh

    config = load_config(args.config)

    mesh = AgentMesh(url=config.mesh.url, agent=config.mesh.agent_id)
    mesh_ok = mesh.is_healthy()
    print(f"flux7-mesh ({config.mesh.url}): {'ok' if mesh_ok else 'unreachable'}")

    if config.memory.enabled:
        try:
            from mem7 import Mem7

            m = Mem7(config.memory.url, token=config.memory.token)
            mem_ok = m.health()
            print(f"flux7-memory ({config.memory.url}): {'ok' if mem_ok else 'unreachable'}")
        except Exception:
            print(f"flux7-memory ({config.memory.url}): unreachable")
    else:
        print("flux7-memory: disabled")

    ev = config.evaluator
    print(f"evaluator: {ev.provider}" + (f" ({ev.model})" if ev.provider != "none" else ""))
    print(f"rules: {len(config.rules)}")

    if not mesh_ok:
        sys.exit(1)


def cmd_bench_replay(args: argparse.Namespace) -> None:
    from collections import Counter

    from sup7 import bench

    keywords = list(args.exclude)
    if args.exclude_file:
        with open(args.exclude_file) as f:
            keywords += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    with open(args.traces, errors="ignore") as f:
        # allowed repositories are the project for target_zone
        home = os.path.expanduser("~")
        projects = [os.path.expanduser(p) for p in args.project] if args.project else \
            [os.path.join(home, r) for r in (args.allow_repo or [])]
        sel = bench.select(f, keywords, agents=args.agent, allow_repos=args.allow_repo, project_dirs=projects)
    cases = sel.cases[: args.limit] if args.limit else sel.cases

    print(f"{len(sel.cases)} cases kept, {sum(sel.excluded.values())} calls excluded")
    for kw, n in sel.excluded.most_common():
        print(f"  excluded by {kw!r}: {n}")
    for why, n in sel.skipped.most_common():
        print(f"  skipped ({why}): {n}")
    print("by policy label:", dict(Counter(c.label for c in cases)))
    print("by tool:", dict(Counter(c.context.tool for c in cases).most_common(10)))
    if args.dry_run:
        return

    setup_logging(args.verbose)
    config = load_config(args.config)
    ev = bench.evaluator_config(config, args.provider)
    kept = bench.done(args.out) if args.resume else []
    ids = {r["trace_id"] for r in kept}
    todo = [c for c in cases if c.trace_id not in ids]
    if args.resume:
        print(f"resume: {len(kept)} results kept, {len(todo)} to replay")
    with open(args.out, "w") as out:
        for r in kept:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
        results = asyncio.run(bench.replay(
            todo, ev, config.evaluator.confidence_threshold, args.concurrency, out))
    print(bench.report(kept + results))
    print(f"results: {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sup7",
        description="flux7-supervisor — L1 evaluation agent for flux7-mesh",
    )
    parser.add_argument(
        "-c", "--config",
        default="sup7.yaml",
        help="config file path (default: sup7.yaml)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("start", help="start the supervisor poll loop")
    sub.add_parser("status", help="check mesh and memory connectivity")
    bench = sub.add_parser("bench", help="offline evaluation of traced calls")
    bench_sub = bench.add_subparsers(dest="bench_command")
    rp = bench_sub.add_parser("replay", help="replay mesh7 traces through an evaluator, resolve nothing")
    rp.add_argument("--traces", required=True, help="mesh7 traces JSONL")
    rp.add_argument("--out", default="replay.jsonl", help="results JSONL (default: replay.jsonl)")
    rp.add_argument("--provider", help="provider of the configured chain to use (default: the configured evaluator)")
    rp.add_argument("--exclude", action="append", default=[], metavar="KEYWORD",
                    help="drop any call whose tool name or parameters contain this keyword (repeatable)")
    rp.add_argument("--exclude-file", help="file with one exclusion keyword per line")
    rp.add_argument("--allow-repo", action="append", metavar="DIR",
                    help="allowlist: keep only calls naming ~/DIR (repeatable); query tools may go without a path")
    rp.add_argument("--project", action="append", metavar="DIR",
                    help="project directory given to the evaluator (repeatable; default: the allowed repos)")
    rp.add_argument("--agent", action="append", help="keep only this agent (repeatable)")
    rp.add_argument("--limit", type=int, help="replay at most this many cases")
    rp.add_argument("--concurrency", type=int, default=4)
    rp.add_argument("--resume", action="store_true",
                    help="keep the results already in --out, replay only the missing and failed cases")
    rp.add_argument("--dry-run", action="store_true", help="count what would be sent and excluded, send nothing")

    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "bench" and args.bench_command == "replay":
        cmd_bench_replay(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
