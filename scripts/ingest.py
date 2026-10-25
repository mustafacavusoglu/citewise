"""Index a folder of documents into the running API via POST /ingest.

Usage:
    uv run --env-file .env python -m scripts.ingest data/sample \\
        [--url http://localhost:8000] [--glob "**/*"] [--key KEY]

.md/.markdown/.txt files are sent as text; .json files must already be an
ingest body ({"text", "source", "metadata"?}). The API requires a Bearer key:
pass --key or set INGEST_KEY (the .env file above provides it).
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
_H1 = re.compile(r"^# +(.+?)\s*$", re.MULTILINE)


def title_of(text: str, stem: str) -> str:
    """First Markdown '# ' heading, else the file stem."""
    m = _H1.search(text)
    return m.group(1) if m else stem


def build_body(path: Path, root: Path) -> dict | None:
    """Ingest body for one file; None if the file is empty or not a supported type."""
    suffix = path.suffix.lower()
    if suffix not in TEXT_SUFFIXES | {".json"}:
        return None
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return None
    if suffix == ".json":
        body = json.loads(raw)
        if not isinstance(body, dict) or not {"text", "source"} <= body.keys():
            raise ValueError("JSON file needs 'text' and 'source' keys")
        return body
    return {
        "text": raw,
        "source": path.relative_to(root).as_posix(),
        "metadata": {"title": title_of(raw, path.stem)},
    }


def post(url: str, key: str, body: dict) -> dict:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    req = urllib.request.Request(f"{url}/ingest", json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def main() -> int:
    ap = argparse.ArgumentParser(description="Index a folder into the running API.")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--glob", default="**/*")
    ap.add_argument("--key", default=os.getenv("INGEST_KEY"), help="default: $INGEST_KEY")
    args = ap.parse_args()
    if not args.key:
        print("Set INGEST_KEY (see .env.example) or pass --key", file=sys.stderr)
        return 2
    if not args.folder.is_dir():
        print(f"not a folder: {args.folder}", file=sys.stderr)
        return 2

    url = args.url.rstrip("/")
    total = failed = 0
    for path in sorted(p for p in args.folder.glob(args.glob) if p.is_file()):
        name = path.relative_to(args.folder).as_posix()
        try:
            body = build_body(path, args.folder)
            if body is None:
                continue
            chunks = post(url, args.key, body)["chunks"]
        except urllib.error.HTTPError as e:
            if e.code == 401:
                print(f"FAIL {name}: unauthorized: INGEST_KEY does not match the server's")
            else:
                print(f"FAIL {name}: HTTP {e.code} {e.read().decode(errors='replace')}")
            failed += 1
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            print(f"FAIL {name}: {e}")
            failed += 1
        else:
            print(f"ok   {name}: {chunks} chunks")
            total += chunks
    print(f"total: {total} chunks, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
