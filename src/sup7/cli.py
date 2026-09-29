"""CLI entry point for sup7."""

from __future__ import annotations

import argparse
import asyncio
import logging
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

    print(f"evaluator: {config.evaluator.provider} ({config.evaluator.model})")
    print(f"rules: {len(config.rules)}")

    if not mesh_ok:
        sys.exit(1)


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

    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
