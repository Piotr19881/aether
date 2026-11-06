"""Compare Aether representations using model-specific tokenizer profiles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.ai_view import (
    Encoder as ProfileEncoder,
    ProfileUnavailable,
    TokenizerProfile,
    create_encoder,
    default_profile,
    generate_ai_view,
    load_profiles,
)
from src.codemap import build_codemap, render_active_slice, render_map
from src.frontend import parse_source


class Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class ProfileResult:
    profile: TokenizerProfile
    counts: tuple[tuple[str, int], ...] = ()
    unavailable_reason: str = ""



def count_tokens(text: str, encoder: Encoder | None = None) -> int:
    """Return exact token count using tiktoken's o200k_base encoding."""
    if encoder is None:
        try:
            import tiktoken
        except ImportError as error:
            raise RuntimeError(
                "The benchmark requires tiktoken. Install it with "
                "'python -m pip install -r requirements.txt'."
            ) from error
        encoder = tiktoken.get_encoding("o200k_base")
    return len(encoder.encode(text))


def savings_percent(baseline: int, candidate: int) -> float:
    """Compute token savings relative to a baseline count."""
    return 0.0 if baseline == 0 else (baseline - candidate) * 100.0 / baseline


def benchmark(
    python_source: str,
    human_source: str,
    ai_source: str,
    graph_slice: str,
    encoder: Encoder | None = None,
) -> list[tuple[str, int, float]]:
    """Measure all representations and return counts plus savings vs `.hl`."""
    human_count = count_tokens(human_source, encoder)
    counts = [
        ("Python", count_tokens(python_source, encoder)),
        (".hl human", human_count),
        (".ai-sem", count_tokens(ai_source, encoder)),
        (".map + active node", count_tokens(graph_slice, encoder)),
    ]
    return [
        (label, count, savings_percent(human_count, count))
        for label, count in counts
    ]


def format_report(results: list[tuple[str, int, float]]) -> str:
    rows = [(label, count, savings) for label, count, savings in results]
    label_width = max(len(label) for label, _, _ in rows)
    lines = [
        "Tokenizer: tiktoken / o200k_base",
        "",
        f"{'View':<{label_width}}  Tokens  Savings vs .hl",
    ]
    lines.append(f"{'-' * label_width}  ------  ---------------")
    for label, count, percent in rows:
        lines.append(f"{label:<{label_width}}  {count:>6}  {percent:>16.1f}%")
    return "\n".join(lines)


def benchmark_profile(
    profile: TokenizerProfile,
    encoder: ProfileEncoder,
    python_source: str,
    human_source: str,
    graph_slice: str,
    tree,
) -> ProfileResult:
    """Measure equivalent views and optimize aliases against one profile encoder."""
    ai_view = generate_ai_view(tree, profile, encoder)
    counts = (
        ("Python", len(encoder.encode(python_source))),
        (".hl", len(encoder.encode(human_source))),
        (".ai", len(encoder.encode(ai_view))),
        (".map + active", len(encoder.encode(graph_slice))),
    )
    return ProfileResult(profile, counts)


def benchmark_profiles(
    profiles: list[TokenizerProfile],
    python_source: str,
    human_source: str,
    graph_slice: str,
    tree,
    encoders: dict[str, ProfileEncoder] | None = None,
) -> list[ProfileResult]:
    """Benchmark each available tokenizer independently; never substitute tokenizers."""
    results = []
    supplied = encoders or {}
    for profile in profiles:
        try:
            encoder = supplied.get(profile.name) or create_encoder(profile)
        except ProfileUnavailable as error:
            results.append(ProfileResult(profile, unavailable_reason=str(error)))
            continue
        results.append(
            benchmark_profile(
                profile,
                encoder,
                python_source,
                human_source,
                graph_slice,
                tree,
            )
        )
    return results


def format_profile_report(results: list[ProfileResult]) -> str:
    """Format profile-specific counts and savings against each profile's `.hl`."""
    lines = [
        "Tokenizer profile results (no tokenizer is substituted for another):",
        "",
        "Profile                 Tokenizer / status                         Python   .hl   .ai   .map   AI save",
        "----------------------  -----------------------------------------  -------  ----  ----  -----  --------",
    ]
    for result in results:
        profile = result.profile
        if result.unavailable_reason:
            lines.append(
                f"{profile.name:<22}  {result.unavailable_reason}"
            )
            continue
        counts = dict(result.counts)
        human_count = counts[".hl"]
        saving = savings_percent(human_count, counts[".ai"])
        lines.append(
            f"{profile.name:<22}  {profile.backend}:{profile.tokenizer:<31} "
            f"{counts['Python']:>7}  {human_count:>4}  {counts['.ai']:>4}  "
            f"{counts['.map + active']:>5}  {saving:>7.1f}%"
        )
        if "proxy" in profile.name:
            lines.append(f"  NOTE: {profile.description}")
    return "\n".join(lines)


def run(
    source_path: Path,
    python_path: Path,
    active_node: str,
    *,
    encoder: Encoder | None = None,
    profile_directory: Path | None = None,
    profile_names: set[str] | None = None,
) -> str:
    """Generate profile-specific views, map, and per-tokenizer measurements."""
    with source_path.open("r", encoding="utf-8", newline="") as source_file:
        source = source_file.read()
    with python_path.open("r", encoding="utf-8", newline="") as python_file:
        python_source = python_file.read()
    frontend = parse_source(source)
    errors = [
        diagnostic
        for diagnostic in frontend.diagnostics
        if diagnostic.severity == "error"
    ]
    if errors:
        joined = "; ".join(
            f"{item.code} line {item.span.line}: {item.message}"
            for item in errors
        )
        raise ValueError(f"Cannot benchmark invalid Aether source: {joined}")
    code_map = build_codemap(frontend.ast)
    graph_slice = render_active_slice(code_map, active_node)
    map_text = render_map(code_map)
    source_path.with_suffix(".map").write_text(map_text, encoding="utf-8")
    profiles = load_profiles(profile_directory) if profile_directory else load_profiles()
    if profile_names is not None:
        unknown = profile_names - {profile.name for profile in profiles}
        if unknown:
            raise ValueError(f"Unknown tokenizer profile(s): {', '.join(sorted(unknown))}")
        profiles = [profile for profile in profiles if profile.name in profile_names]

    supplied_encoders = (
        {profile.name: encoder for profile in profiles}
        if encoder is not None
        else None
    )
    results = benchmark_profiles(
        profiles,
        python_source,
        source,
        graph_slice,
        frontend.ast,
        supplied_encoders,
    )
    for result in results:
        if result.unavailable_reason:
            continue
        view_encoder = (
            supplied_encoders.get(result.profile.name)
            if supplied_encoders is not None
            else None
        )
        if view_encoder is None and not result.unavailable_reason:
            view_encoder = create_encoder(result.profile)
        ai_view = generate_ai_view(frontend.ast, result.profile, view_encoder)
        output_path = (
            source_path.with_suffix(".ai-sem")
            if result.profile.name == default_profile().name
            else source_path.with_suffix(f".ai-{result.profile.name}")
        )
        output_path.write_text(ai_view, encoding="utf-8")
    return format_profile_report(results)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "examples" / "sensor.hl",
        help="Aether source file to benchmark",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=project_root / "examples" / "sensor.py",
        help="Equivalent Python source file",
    )
    parser.add_argument(
        "--active",
        default="f:normalize_voltage",
        help="Active codemap node id (default: f:normalize_voltage)",
    )
    parser.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        help="Profile name to benchmark (repeatable; default: all profiles)",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="Directory containing tokenizer_profile JSON files",
    )
    args = parser.parse_args()
    print(
        run(
            args.source,
            args.python,
            args.active,
            profile_directory=args.profile_dir,
            profile_names=set(args.profiles) if args.profiles else None,
        )
    )


if __name__ == "__main__":
    main()
