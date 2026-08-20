from __future__ import annotations

from pathlib import Path

import config


def sanitize_text(value: str) -> str:
    return (
        str(value)
        .replace("\r", " ")
        .replace("\n", " ")
        .replace('"', "")
        .replace(",", ";")
        .strip()
    )


def load_event_name_map(symbol: str, timeframe: str) -> dict[str, str]:
    path = config.runtime_event_details_path(symbol, timeframe)
    if not path.exists():
        path = config.terminal_runtime_event_details_path(symbol, timeframe)
    if not path.exists():
        return {}

    event_map: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        next(handle, None)
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",", 7)
            if len(parts) < 8:
                continue
            bar_time, _event_time, event_name, _country, _imp, _max_imp, _count, _offset = parts
            event_map[bar_time] = sanitize_text(event_name)
    return event_map


def repair_transaction_file(path: Path, event_map: dict[str, str]) -> bool:
    original_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not original_lines:
        return False

    repaired_lines = [original_lines[0]]
    changed = False

    for line in original_lines[1:]:
        if not line.strip():
            continue

        parts = line.split(",", 14)
        if len(parts) < 15:
            repaired_lines.append(line)
            continue

        comment = parts[14]
        bar_time = parts[0]

        if comment.startswith('"') or "\r" in comment or "\n" in comment:
            comment = event_map.get(bar_time, comment)
            changed = True

        sanitized_comment = sanitize_text(comment)
        if sanitized_comment != parts[14]:
            changed = True
        parts[14] = sanitized_comment
        repaired_lines.append(",".join(parts))

    if changed:
        try:
            path.write_text("\n".join(repaired_lines) + "\n", encoding="utf-8")
        except PermissionError:
            repaired_path = path.with_name(path.stem + "_repaired.csv")
            repaired_path.write_text("\n".join(repaired_lines) + "\n", encoding="utf-8")
            print(f"Locked file preserved, repaired copy written to: {repaired_path}")
    return changed


def repair_symbol(symbol: str) -> list[Path]:
    timeframe = config.resolve_pair_timeframe(symbol)
    event_map = load_event_name_map(symbol, timeframe)
    repaired: list[Path] = []
    for path in sorted(config.BACKTEST_DIR.glob(f"{symbol}_{timeframe}_*_transactions.csv")):
        if repair_transaction_file(path, event_map):
            repaired.append(path)
    return repaired


def main() -> None:
    repaired = repair_symbol("GBPUSD")
    if not repaired:
        print("No GBPUSD transaction files required repair.")
        return
    print("Repaired GBPUSD transaction files:")
    for path in repaired:
        print(path)


if __name__ == "__main__":
    main()
