import sys
from datetime import datetime, timezone, timedelta


def parse_jobs_log(path):
    jobs = []
    with open(path) as f:
        lines = f.readlines()

    if not lines or lines[0].strip().split()[0] != "Seq":
        raise ValueError("jobs.log does not appear to have a valid header")

    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) < 8:
            continue
        try:
            start_time = float(parts[2])
            runtime = float(parts[3])
            exitval = int(parts[6])
            command = parts[8] if len(parts) >= 9 else ""
            jobs.append((start_time, runtime, exitval, command))
        except ValueError:
            continue
    return jobs


def parse_time_string(time_str):
    """Parses 'HH:MM' into (hour, minute)."""
    parts = time_str.strip().split(":")
    if len(parts) != 2:
        raise ValueError("Time must be in 'HH:MM' format")
    return int(parts[0]), int(parts[1])


def summarize_jobs(jobs, timeout_threshold=5400, target_time_str=None, time_window_minutes=1, timezone_offset_hours=1):
    total = len(jobs)
    all_success = all(j[2] == 0 for j in jobs)

    allowed_excess_time = 30
    over_timeout = [j for j in jobs if j[1] > timeout_threshold + allowed_excess_time]

    lux_offset = timedelta(hours=timezone_offset_hours)
    jobs_around_time = []

    if target_time_str:
        target_hour, target_minute = parse_time_string(target_time_str)
        window = timedelta(minutes=time_window_minutes)

        for j in jobs:
            start_dt = datetime.fromtimestamp(j[0], tz=timezone.utc) + lux_offset
            target_dt = start_dt.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
            if abs(start_dt - target_dt) <= window:
                jobs_around_time.append((start_dt, j[1], j[3]))  # time, duration, command

    print(f"🧮 Total jobs logged: {total}")
    print(f"✅ All jobs successful (Exitval=0): {'Yes' if all_success else 'No'}")
    print(f"⏱️ Jobs exceeding {timeout_threshold} seconds: {len(over_timeout)}")
    for job in over_timeout:
        start = datetime.fromtimestamp(job[0], tz=timezone.utc) + lux_offset
        print(f"  - Start: {start}, Runtime: {job[1]:.2f} sec")

    if target_time_str:
        print(f"🕒 Jobs around {target_time_str} ±{time_window_minutes} min: {len(jobs_around_time)}")
        for start, duration, cmd in jobs_around_time:
            print(f"  - Start: {start}, Duration: {duration:.2f} sec")

    return {
        "total_jobs": total,
        "all_successful": all_success,
        "over_timeout": over_timeout,
        "jobs_around_time": jobs_around_time,
        "all_jobs": jobs
    }


def analyze_jobs_log(
    log_path,
    timeout_threshold=5400,
    target_time_str=None,
    time_window_minutes=1,
    timezone_offset_hours=1
):
    jobs = parse_jobs_log(log_path)
    return summarize_jobs(
        jobs,
        timeout_threshold=timeout_threshold,
        target_time_str=target_time_str,
        time_window_minutes=time_window_minutes,
        timezone_offset_hours=timezone_offset_hours
    )


# ✅ Command-line usage
if __name__ == "__main__":
    if len(sys.argv) < 1 or len(sys.argv) > 4:
        print("\nUsage:")
        print("  python3 review_logs.py <path_to_jobs.log> [timeout_seconds] [conflicting_time 'HH:MM']")
        print("\nExamples:")
        print("  python3 review_logs.py jobs.log")
        print("  python3 review_logs.py jobs.log 5400")
        print("  python3 review_logs.py jobs.log 5400 14:17")
        sys.exit(1)

    log_path = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 5400
    conflicting_time = sys.argv[3] if len(sys.argv) > 3 else None

    jobs = parse_jobs_log(log_path)
    summarize_jobs(jobs, timeout, conflicting_time)

