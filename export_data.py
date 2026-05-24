"""
Export F1 season data → static JSON for the Next.js dashboard.

Uses fastf1.ergast (HTTP API, ~5s/season) — no telemetry, no live backend.
Run:  ./venv/Scripts/python.exe export_data.py
Output: web/public/data/<year>.json  +  web/public/data/index.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastf1.ergast import Ergast

from src.utils.config import DRIVER_COLORS, TEAM_COLORS, SUPPORTED_YEARS

OUT_DIR = Path(__file__).parent / "web" / "public" / "data"
_PAGE_SIZE = 100
_MAX_OFFSET = 2000
_TEAM_FALLBACK = "#9AA0AA"


def _paginate(fetch_method, season: int) -> dict[int, pd.DataFrame]:
    """Paginate Ergast results, merging round partials across pages."""
    parts: dict[int, list[pd.DataFrame]] = {}
    race_names: dict[int, str] = {}
    circuit_names: dict[int, str] = {}
    seen: set[int] = set()
    offset = 0
    while offset <= _MAX_OFFSET and offset not in seen:
        seen.add(offset)
        try:
            resp = fetch_method(season=season, limit=_PAGE_SIZE, offset=offset)
            desc, contents = resp.description, resp.content
        except Exception:
            break
        if desc is None or len(desc) == 0 or contents is None or len(contents) == 0:
            break
        total = sum(len(c) for c in contents)
        if total == 0:
            break
        for desc_row, df in zip(desc.itertuples(index=False), contents):
            try:
                rnd = int(desc_row.round)
            except (AttributeError, TypeError, ValueError):
                continue
            if df is None or len(df) == 0:
                continue
            parts.setdefault(rnd, []).append(df.copy())
            if rnd not in race_names:
                race_names[rnd] = str(getattr(desc_row, "raceName", f"Round {rnd}"))
                circuit_names[rnd] = str(getattr(desc_row, "circuitName", ""))
        if total < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE

    out: dict[int, pd.DataFrame] = {}
    for rnd, dfs in parts.items():
        df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)
        if "driverCode" in df.columns:
            df = df.drop_duplicates(subset=["driverCode"], keep="first")
        df = df.reset_index(drop=True)
        df["_round"] = rnd
        df["_raceName"] = race_names.get(rnd, f"Round {rnd}")
        df["_circuitName"] = circuit_names.get(rnd, "")
        out[rnd] = df
    return out


def _num(v, default=0.0):
    try:
        return float(v) if pd.notna(v) else default
    except (TypeError, ValueError):
        return default


def _int(v):
    try:
        return int(v) if pd.notna(v) else None
    except (TypeError, ValueError):
        return None


def _is_dnf(status: str) -> bool:
    return not (status == "Finished" or "Lap" in status)


def _result_label(pos, points, dnf) -> str:
    if dnf:
        return "DNF"
    if pos == 1:
        return "Win"
    if pos is not None and pos <= 3:
        return "Podium"
    if points and points > 0:
        return "Points"
    return "Finished"


def build_season(year: int) -> dict | None:
    E = Ergast()
    races = _paginate(E.get_race_results, year)
    quali = _paginate(E.get_qualifying_results, year)
    sprints = _paginate(E.get_sprint_results, year)
    if not races:
        return None

    rounds = sorted(races.keys())
    round_meta = [
        {
            "round": r,
            "name": str(races[r]["_raceName"].iloc[0]),
            "circuit": str(races[r]["_circuitName"].iloc[0]),
        }
        for r in rounds
    ]

    # Poles per driver (quali P1)
    poles: dict[str, int] = {}
    for df in quali.values():
        if df is None or "driverCode" not in df.columns:
            continue
        for _, row in df.iterrows():
            if _int(row.get("position")) == 1:
                poles[row["driverCode"]] = poles.get(row["driverCode"], 0) + 1

    drivers: dict[str, dict] = {}
    constructors: dict[str, dict] = {}
    recent_rows: list[dict] = []
    recent_cut = rounds[-6:] if len(rounds) >= 6 else rounds

    for r in rounds:
        df = races[r]
        if "driverCode" not in df.columns:
            continue
        sprint_df = sprints.get(r)
        for _, row in df.iterrows():
            code = row.get("driverCode")
            if not isinstance(code, str) or not code:
                continue
            given = str(row.get("givenName", "")) if pd.notna(row.get("givenName")) else ""
            family = str(row.get("familyName", "")) if pd.notna(row.get("familyName")) else ""
            name = f"{given} {family}".strip() or code
            team = str(row.get("constructorName", "")) if pd.notna(row.get("constructorName")) else ""
            pos = _int(row.get("position"))
            pts = _num(row.get("points"))
            status = str(row.get("status", "")) if pd.notna(row.get("status")) else ""
            dnf = _is_dnf(status)
            fl_rank = _int(row.get("fastestLapRank"))

            # Sprint points for this driver/round
            sprint_pts = 0.0
            if sprint_df is not None and "driverCode" in sprint_df.columns:
                m = sprint_df[sprint_df["driverCode"] == code]
                if len(m) > 0:
                    sprint_pts = _num(m.iloc[0].get("points"))

            d = drivers.setdefault(code, {
                "code": code, "name": name, "team": team,
                "color": DRIVER_COLORS.get(code, _TEAM_FALLBACK),
                "points": 0.0, "wins": 0, "podiums": 0,
                "poles": poles.get(code, 0), "fastestLaps": 0, "dnfs": 0,
                "racesEntered": 0, "number": _int(row.get("number")),
                "perRound": {r: 0.0 for r in rounds},
            })
            total_pts = pts + sprint_pts
            d["points"] += total_pts
            d["perRound"][r] = total_pts
            d["racesEntered"] += 1
            d["team"] = team
            d["color"] = DRIVER_COLORS.get(code, TEAM_COLORS.get(team, _TEAM_FALLBACK))
            if dnf:
                d["dnfs"] += 1
            if pos == 1:
                d["wins"] += 1
            if pos is not None and pos <= 3:
                d["podiums"] += 1
            if fl_rank == 1 and pos is not None and pos <= 10:
                d["fastestLaps"] += 1

            c = constructors.setdefault(team, {
                "team": team, "color": TEAM_COLORS.get(team, _TEAM_FALLBACK),
                "points": 0.0, "wins": 0,
            })
            c["points"] += total_pts
            if pos == 1:
                c["wins"] += 1

            if r in recent_cut:
                recent_rows.append({
                    "round": r,
                    "gp": str(df["_raceName"].iloc[0]),
                    "code": code, "name": name, "team": team,
                    "color": DRIVER_COLORS.get(code, TEAM_COLORS.get(team, _TEAM_FALLBACK)),
                    "position": pos, "points": total_pts,
                    "status": "Success" if not dnf else "Failure",
                    "result": _result_label(pos, total_pts, dnf),
                })

    # Finalize driver list: cumulative + perRound arrays aligned to rounds
    driver_list = []
    for code, d in drivers.items():
        per = [round(d["perRound"][r], 1) for r in rounds]
        cum, run = [], 0.0
        for r in rounds:
            run += d["perRound"][r]
            cum.append(round(run, 1))
        driver_list.append({
            "code": d["code"], "name": d["name"], "team": d["team"],
            "color": d["color"], "number": d["number"],
            "points": round(d["points"], 1),
            "wins": d["wins"], "podiums": d["podiums"], "poles": d["poles"],
            "fastestLaps": d["fastestLaps"], "dnfs": d["dnfs"],
            "racesEntered": d["racesEntered"],
            "perRound": per, "cumulative": cum,
        })
    driver_list.sort(key=lambda x: x["points"], reverse=True)

    constructor_list = sorted(
        ({"team": c["team"], "color": c["color"],
          "points": round(c["points"], 1), "wins": c["wins"]}
         for c in constructors.values()),
        key=lambda x: x["points"], reverse=True,
    )

    # Recent: most recent round first, best finish first; cap 16
    recent_rows.sort(key=lambda x: (-x["round"], x["position"] if x["position"] else 99))
    recent = recent_rows[:16]

    total_points = round(sum(d["points"] for d in driver_list), 1)
    return {
        "year": year,
        "rounds": round_meta,
        "drivers": driver_list,
        "constructors": constructor_list,
        "recent": recent,
        "totals": {
            "drivers": len(driver_list),
            "rounds": len(rounds),
            "totalPoints": total_points,
            "completedRounds": len(rounds),
        },
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exported = []
    for year in SUPPORTED_YEARS:
        print(f"Fetching {year} …", flush=True)
        season = build_season(year)
        if season is None:
            print(f"  ! no data for {year}, skipping")
            continue
        (OUT_DIR / f"{year}.json").write_text(
            json.dumps(season, indent=None, separators=(",", ":")), encoding="utf-8"
        )
        exported.append(year)
        d = season["drivers"][0]
        print(f"  ok {year}: {len(season['drivers'])} drivers, "
              f"{len(season['rounds'])} rounds, leader {d['code']} {d['points']}pts")

    (OUT_DIR / "index.json").write_text(
        json.dumps({"years": sorted(exported, reverse=True),
                    "default": max(exported) if exported else None}),
        encoding="utf-8",
    )
    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
