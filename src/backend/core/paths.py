from __future__ import annotations

from pathlib import Path


class UnsafePathError(ValueError):
    pass


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_root(root: str | Path | None) -> Path:
    if root is None:
        return get_project_root().resolve()

    root_path = Path(root)
    if root_path.is_absolute():
        return root_path.resolve()

    return (get_project_root() / root_path).resolve()


def resolve_project_path(
    path_value: str | Path,
    *,
    root: str | Path | None = None,
    allow_absolute: bool = True,
    must_exist: bool = False,
) -> Path:
    raw = Path(path_value)
    allowed_root = _resolve_root(root)

    candidate = raw.resolve() if raw.is_absolute() else (get_project_root() / raw).resolve()

    if raw.is_absolute() and not allow_absolute:
        raise UnsafePathError(f"Absolute paths are not allowed: {path_value}")

    if not _is_relative_to(candidate, allowed_root):
        raise UnsafePathError(f"Path escapes allowed root: {path_value}")

    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)

    return candidate


def resolve_config_path(path_value: str | Path, *, must_exist: bool = True) -> Path:
    return resolve_project_path(
        path_value,
        root=get_project_root() / "configs",
        allow_absolute=True,
        must_exist=must_exist,
    )


def resolve_model_artifact_path(path_value: str | Path, *, must_exist: bool = True) -> Path:
    candidate = resolve_project_path(path_value, allow_absolute=True, must_exist=must_exist)
    allowed_roots = [
        get_project_root() / "outputs" / "models",
        get_project_root() / "models",
        get_project_root() / "artifacts",
    ]

    if not any(_is_relative_to(candidate, root.resolve()) for root in allowed_roots):
        raise UnsafePathError(f"Model artifact path is outside allowed model roots: {path_value}")

    return candidate
