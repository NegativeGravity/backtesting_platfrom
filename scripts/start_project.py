#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

DEFAULT_MARKET_CONFIG = "configs/market_data.docker.yaml"
DEFAULT_SPLIT_POLICY = "recommended"

REQUIRED_IMAGES = [
    "trading-platform-backend",
    "trading-platform-frontend",
]


# ANSI Color codes for terminal formatting
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class CommandError(RuntimeError):
    pass


def get_timestamp() -> str:
    # Returns current time formatted as [HH:MM:SS]
    return f"[{datetime.datetime.now().strftime('%H:%M:%S')}]"


def log_step(message: str) -> None:
    # Print a major step with Cyan color and bold text
    print(f"\n{Colors.CYAN}{Colors.BOLD}{get_timestamp()} ==> {message}{Colors.RESET}", flush=True)


def log_ok(message: str) -> None:
    # Print a success message with Green color
    print(f"{Colors.GREEN}{get_timestamp()} ✓ {message}{Colors.RESET}", flush=True)


def log_warn(message: str) -> None:
    # Print a warning message with Yellow color
    print(f"{Colors.YELLOW}{get_timestamp()} ⚠ {message}{Colors.RESET}", flush=True)


def run_command(
        command: list[str],
        *,
        check: bool = True,
) -> subprocess.CompletedProcess:
    cmd_str = ' '.join(command)
    # Highlight the command being executed in Blue
    print(f"{Colors.BLUE}{get_timestamp()} $ {cmd_str}{Colors.RESET}", flush=True)

    start_time = time.time()
    result = subprocess.run(command, check=False)
    elapsed_time = time.time() - start_time

    if check and result.returncode != 0:
        raise CommandError(
            f"Command failed with exit code {result.returncode}: {cmd_str}"
        )

    # Log the duration of the command
    print(f"{Colors.BLUE}{get_timestamp()} ⏱  Finished in {elapsed_time:.2f}s{Colors.RESET}", flush=True)
    return result


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def docker_compose(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return run_command(["docker", "compose", *args], check=check)


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return run_command(["docker", *args], check=check)


def ensure_project_root() -> Path:
    root = Path.cwd()

    if not (root / "docker-compose.yml").exists():
        raise FileNotFoundError(
            "docker-compose.yml not found. Run this script from the project root."
        )

    return root


def ensure_env_file(root: Path) -> None:
    env_path = root / ".env"
    env_example = root / ".env.example"

    if env_path.exists():
        return

    if env_example.exists():
        shutil.copyfile(env_example, env_path)
        log_ok(".env created from .env.example")
        return

    env_path.write_text(
        "\n".join(
            [
                "FRONTEND_CONTEXT=./src/frontend",
                "FRONTEND_PORT=3000",
                "BACKEND_PORT=8000",
                "QUESTDB_VERSION=latest",
                "QUESTDB_HTTP_PORT=9000",
                "QUESTDB_PG_PORT=8812",
                "QUESTDB_ILP_PORT=9009",
                "REDIS_PORT=6379",
                "VITE_API_BASE_URL=http://localhost:8000",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log_ok(".env created")


def wait_http(url: str, *, retries: int = 30, delay_seconds: float = 2.0) -> bool:
    for _ in range(retries):
        try:
            with urlopen(url, timeout=3) as response:
                if 200 <= response.status < 500:
                    return True
        except (URLError, TimeoutError, OSError):
            time.sleep(delay_seconds)

    return False


def check_required_tools() -> None:
    log_step("Checking required tools")

    if not command_exists("docker"):
        raise RuntimeError("Docker is not installed or not available in PATH.")

    docker_compose("version")
    log_ok("Docker Compose is available")


def image_exists(image_name: str) -> bool:
    result = docker("image", "inspect", image_name, check=False)
    return result.returncode == 0


def check_ready_images() -> None:
    log_step("Checking ready Docker images")

    missing_images: list[str] = []

    for image_name in REQUIRED_IMAGES:
        if image_exists(image_name):
            log_ok(f"Image exists: {image_name}")
        else:
            missing_images.append(image_name)
            log_warn(f"Image missing: {image_name}")

    if missing_images:
        raise RuntimeError(
            "Some required images are missing. Build them first:\n"
            "docker compose build backend frontend\n\n"
            f"Missing images: {', '.join(missing_images)}"
        )


def start_services(*, build: bool, pull: bool, use_ready_images: bool) -> None:
    if pull:
        log_step("Pulling Docker images")
        docker_compose("pull")

    if use_ready_images and not build:
        check_ready_images()

    log_step("Starting Docker services")

    args = ["up", "-d"]

    if build:
        args.append("--build")
    elif use_ready_images:
        args.append("--no-build")

    docker_compose(*args)

    log_step("Current containers")
    docker_compose("ps", check=False)


def wait_services() -> None:
    log_step("Waiting for backend")
    if wait_http("http://localhost:8000/health", retries=20):
        log_ok("Backend health endpoint is ready")
    elif wait_http("http://localhost:8000/docs", retries=20):
        log_ok("Backend docs endpoint is ready")
    else:
        log_warn("Backend did not respond yet. Check: docker compose logs -f backend")

    log_step("Waiting for frontend")
    if wait_http("http://localhost:3000", retries=20):
        log_ok("Frontend is ready")
    else:
        log_warn("Frontend did not respond yet. Check: docker compose logs -f frontend")

    log_step("Waiting for QuestDB")
    if wait_http("http://localhost:9000", retries=25):
        log_ok("QuestDB UI is ready")
    else:
        log_warn("QuestDB did not respond yet. Check: docker compose logs -f questdb")


def run_backend_python(args: list[str]) -> None:
    docker_compose("run", "--rm", "backend", "python", *args)


def run_data_pipeline(config: str, split_policy: str, *, force_download: bool) -> None:
    log_step("Downloading Binance BTCUSDT 1h data")
    download_command = [
        "scripts/download_binance_klines.py",
        "--config",
        config,
        "--workers",
        "8",
    ]

    if force_download:
        download_command.append("--force")

    run_backend_python(download_command)

    log_step("Ingesting market data into QuestDB")
    run_backend_python(
        [
            "scripts/ingest_questdb_klines.py",
            "--config",
            config,
            "--truncate",
        ]
    )

    log_step("Warming Redis market-data cache")
    run_backend_python(
        [
            "scripts/build_redis_cache.py",
            "--config",
            config,
            "--split-policy",
            split_policy,
        ]
    )

    log_step("Verifying dataset splits")
    run_backend_python(
        [
            "scripts/verify_dataset_splits.py",
            "--config",
            config,
            "--split-policy",
            split_policy,
        ]
    )


def train_ml_momentum(config: str, split_policy: str) -> None:
    log_step("Training ML Momentum model")
    run_backend_python(
        [
            "scripts/train_ml_momentum_model.py",
            "--market-config",
            config,
            "--split-policy",
            split_policy,
        ]
    )


def train_ml_regime(config: str, split_policy: str) -> None:
    log_step("Training ML Regime Meta Label model")
    run_backend_python(
        [
            "scripts/train_ml_regime_meta_label_model.py",
            "--market-config",
            config,
            "--split-policy",
            split_policy,
        ]
    )


def train_dl_temporal(
        config: str,
        split_policy: str,
        *,
        epochs: int,
        sequence_length: int,
        batch_size: int,
) -> None:
    log_step("Training DL Temporal Fusion Momentum model")
    run_backend_python(
        [
            "scripts/train_dl_temporal_fusion_momentum_model.py",
            "--market-config",
            config,
            "--split-policy",
            split_policy,
            "--epochs",
            str(epochs),
            "--sequence-length",
            str(sequence_length),
            "--batch-size",
            str(batch_size),
        ]
    )


def train_models(
        config: str,
        split_policy: str,
        *,
        train_ml: bool,
        train_regime: bool,
        train_dl: bool,
        epochs: int,
        sequence_length: int,
        batch_size: int,
) -> None:
    if train_ml:
        train_ml_momentum(config, split_policy)

    if train_regime:
        train_ml_regime(config, split_policy)

    if train_dl:
        train_dl_temporal(
            config,
            split_policy,
            epochs=epochs,
            sequence_length=sequence_length,
            batch_size=batch_size,
        )


def print_final_message() -> None:
    print(
        f"""{Colors.BOLD}{Colors.GREEN}
Project is up and running!

Frontend:     http://localhost:3000
Backend Docs: http://localhost:8000/docs
QuestDB:      http://localhost:9000
Redis:        localhost:6379

Useful commands:

docker compose logs -f
docker compose logs -f backend
docker compose logs -f frontend
docker compose down

Generated model artifacts will be in:
outputs/models/
{Colors.RESET}""".strip(),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start the trading platform Docker stack using ready images by default. "
            "Optionally build images, run data pipeline, and train ML/DL models."
        )
    )

    parser.add_argument(
        "--build",
        action="store_true",
        help="Build Docker images before starting. If omitted, ready images are used.",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull Docker images before starting.",
    )
    parser.add_argument(
        "--skip-start",
        action="store_true",
        help="Do not start Docker services.",
    )
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Do not wait for service healthchecks.",
    )
    parser.add_argument(
        "--no-ready-image-check",
        action="store_true",
        help="Do not check whether backend/frontend images already exist.",
    )

    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run full data pipeline: download, ingest QuestDB, warm Redis, verify.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force redownload and reprocess Binance monthly zip files.",
    )
    parser.add_argument(
        "--market-config",
        default=DEFAULT_MARKET_CONFIG,
        help=f"Market data config path inside container. Default: {DEFAULT_MARKET_CONFIG}",
    )
    parser.add_argument(
        "--split-policy",
        default=DEFAULT_SPLIT_POLICY,
        choices=[
            "recommended",
            "user_requested_no_overlap",
            "user_requested_with_leakage_warning",
        ],
        help="Dataset split policy.",
    )

    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train all models: ml_momentum, ml_regime_meta_label, dl_temporal_fusion_momentum.",
    )
    parser.add_argument(
        "--train-ml",
        action="store_true",
        help="Train only ML Momentum.",
    )
    parser.add_argument(
        "--train-regime",
        action="store_true",
        help="Train only ML Regime Meta Label.",
    )
    parser.add_argument(
        "--train-dl",
        action="store_true",
        help="Train only DL Temporal Fusion Momentum.",
    )

    parser.add_argument(
        "--dl-epochs",
        type=int,
        default=25,
        help="DL training epochs.",
    )
    parser.add_argument(
        "--dl-sequence-length",
        type=int,
        default=128,
        help="DL sequence length.",
    )
    parser.add_argument(
        "--dl-batch-size",
        type=int,
        default=128,
        help="DL batch size.",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Use ready images, run pipeline, and train all models.",
    )
    parser.add_argument(
        "--full-build",
        action="store_true",
        help="Build images, run pipeline, and train all models.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.full:
        args.pipeline = True
        args.train_all = True

    if args.full_build:
        args.build = True
        args.pipeline = True
        args.train_all = True

    root = ensure_project_root()

    try:
        check_required_tools()
        ensure_env_file(root)

        if not args.skip_start:
            start_services(
                build=args.build,
                pull=args.pull,
                use_ready_images=not args.no_ready_image_check,
            )

        if not args.skip_wait:
            wait_services()

        if args.pipeline:
            run_data_pipeline(
                args.market_config,
                args.split_policy,
                force_download=args.force_download,
            )

        train_ml = args.train_all or args.train_ml
        train_regime = args.train_all or args.train_regime
        train_dl = args.train_all or args.train_dl

        if train_ml or train_regime or train_dl:
            train_models(
                args.market_config,
                args.split_policy,
                train_ml=train_ml,
                train_regime=train_regime,
                train_dl=train_dl,
                epochs=args.dl_epochs,
                sequence_length=args.dl_sequence_length,
                batch_size=args.dl_batch_size,
            )

        print_final_message()
        return 0

    except KeyboardInterrupt:
        print(f"\n{Colors.RED}Interrupted by user.{Colors.RESET}", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n{Colors.RED}{Colors.BOLD}ERROR: {exc}{Colors.RESET}", file=sys.stderr)
        print(f"\n{Colors.YELLOW}Debug commands:{Colors.RESET}", file=sys.stderr)
        print("docker compose ps", file=sys.stderr)
        print("docker compose logs -f backend", file=sys.stderr)
        print("docker compose logs -f frontend", file=sys.stderr)
        print("docker compose logs -f questdb", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())