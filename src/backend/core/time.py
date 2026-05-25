from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def timestamp_for_run_id() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")