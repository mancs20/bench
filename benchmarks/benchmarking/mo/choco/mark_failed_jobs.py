#!/usr/bin/env python3
import sys
import json
import shutil
from pathlib import Path


def infer_paths(argv):
    """
    Path logic:

    - No args:
        joblog = ./jobs.log
        json_dir = ../
    - 1 arg:
        joblog = argv[1]
        json_dir = joblog.parent.parent
    - 2 args:
        joblog = argv[1]
        json_dir = argv[2]
    """
    if len(argv) == 1:
        joblog = Path("jobs.log").resolve()
        json_dir = joblog.parent.parent  # parent of choco/ → /pathjson
    elif len(argv) == 2:
        joblog = Path(argv[1]).resolve()
        json_dir = joblog.parent.parent
    elif len(argv) == 3:
        joblog = Path(argv[1]).resolve()
        json_dir = Path(argv[2]).resolve()
    else:
        print(f"Usage: {argv[0]} [JOBLOG_PATH [JSON_DIR]]", file=sys.stderr)
        sys.exit(1)

    if not joblog.exists():
        print(f"Error: joblog file not found: {joblog}", file=sys.stderr)
        sys.exit(1)

    if not json_dir.exists():
        print(f"Error: JSON directory not found: {json_dir}", file=sys.stderr)
        sys.exit(1)

    return joblog, json_dir


def main():
    joblog_path, json_dir = infer_paths(sys.argv)

    print(f"Using joblog: {joblog_path}")
    print(f"Using JSON dir: {json_dir}")

    # 1) Build set of (benchmark, problem, instance, front_generator) that have errors
    error_keys = set()

    # error_patterns = [
    #     'Exception in thread "main"',
    #     "CANCELLED AT",
    #     "error",  # generic, can be tuned later
    # ]
    error_patterns = [
        'Exception in thread "main"',
        "CANCELLED AT",
        "error",  # generic, can be tuned later
    ]

    for json_path in json_dir.glob("*.json"):
        try:
            text = json_path.read_text(errors="ignore")
        except Exception as e:
            print(f"Warning: could not read {json_path}: {e}", file=sys.stderr)
            continue

        # Quick filter: skip files with no obvious error markers
        lower_text = text.lower()
        if not any(pat.lower() in lower_text for pat in error_patterns):
            continue

        # Find the statistics line to get benchmark/problem/instance/front_generator
        stats = None
        for line in text.splitlines():
            if '"type": "statistics"' in line:
                try:
                    msg = json.loads(line)
                    stats = msg.get("statistics", {})
                except Exception:
                    pass
                break

        if not stats:
            continue

        key = (
            stats.get("benchmark"),
            stats.get("problem"),
            stats.get("instance"),
            stats.get("front_generator"),
        )

        if all(key):
            error_keys.add(key)

    if not error_keys:
        print("No error keys detected from JSON logs; nothing to change.", file=sys.stderr)
        return

    print("Detected error keys (benchmark, problem, instance, front_generator):")
    for k in sorted(error_keys):
        print("  ", k)

    # 2) Read jobs.log, mark matching lines, and rewrite file
    if not joblog_path.exists():
        print(f"Error: joblog not found: {joblog_path}", file=sys.stderr)
        sys.exit(1)

    # Create backup first
    backup_path = joblog_path.with_suffix(joblog_path.suffix + ".bak")
    print(f"Creating backup: {backup_path}")
    shutil.copy2(joblog_path, backup_path)

    # Read all lines
    with joblog_path.open() as f:
        lines = f.readlines()

    # Rewrite jobs.log in-place
    with joblog_path.open("w") as out:
        if not lines:
            print("Warning: joblog is empty.", file=sys.stderr)
            return

        # Keep header as-is
        header = lines[0]
        out.write(header)

        for line in lines[1:]:
            line_stripped = line.rstrip("\n")
            if not line_stripped.strip():
                out.write(line)
                continue

            parts = line_stripped.split("\t", 8)  # up to 9 columns: 0..8
            if len(parts) < 9:
                # Not a normal job line, just write it back
                out.write(line)
                continue

            seq, host, start, runtime, send, recv, exitval, sig, cmd = parts

            mark_fail = False

            for (bench, prob, inst, fg) in error_keys:
                pattern = f'"{bench}" "{prob}" "{inst}"'
                # We match both the (bench,prob,inst) triple and the front generator
                if pattern in cmd and f" {fg} " in cmd:
                    mark_fail = True
                    break

            if mark_fail:
                exitval = "1"  # mark as failed

            new_parts = [seq, host, start, runtime, send, recv, exitval, sig, cmd]
            out.write("\t".join(new_parts) + "\n")

    print("jobs.log updated. To restore backup if needed:")
    print(f"  rm {joblog_path.name}")
    print(f"  mv {backup_path.name} {joblog_path.name}")


if __name__ == "__main__":
    main()
