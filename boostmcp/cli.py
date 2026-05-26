import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="boostmcp")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="Start MCP + Web UI server")
    args = parser.parse_args()
    if args.command == "serve":
        print("boostmcp serve not implemented yet", file=sys.stderr)
        sys.exit(1)
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
