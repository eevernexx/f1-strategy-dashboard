"""
Season Data Loader (Ergast API)
================================
Fetch full-season race/qualifying/sprint results via fastf1.ergast.Ergast.

Why Ergast (bukan FastF1 session.load):
- session.load() butuh ~3-5 min/season untuk 22-24 races (telemetry overhead).
- Ergast = HTTP API, ~5 detik/season untuk results saja (no telemetry/laps).
- Cocok untuk Mode B (Season H2H) yang cuma butuh hasil per race.

Pagination:
- Public Ergast API hard-cap di ~100 row per request.
- 2024 season = 24 races × ~20 drivers = ~480 rows → butuh 5 request via offset.
- Helper `_paginate_results` handle dedup overlap antar page.

Caching:
- @st.cache_data ttl=86400 (1 hari) karena data historis tidak berubah.
- session_id key = (year, type) — caching independen per kombinasi.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

# Ergast wrapper. Import di module-level supaya error import muncul early.
try:
    from fastf1.ergast import Ergast
    _ERGAST_AVAILABLE = True
except Exception:
    Ergast = None     # type: ignore[assignment]
    _ERGAST_AVAILABLE = False


# Hard cap Ergast per-request — discovered via API testing.
_PAGE_SIZE = 100
# Max offset to try (safety bound). 2000 rows = 100 races × 20 drivers, lebih
# dari cukup untuk satu season.
_MAX_OFFSET = 2000


def _ergast() -> "Ergast | None":
    """Construct Ergast client; return None kalau library tidak tersedia."""
    if not _ERGAST_AVAILABLE:
        return None
    try:
        return Ergast()
    except Exception:
        return None


# ── Generic paginator ────────────────────────────────────────────────────────

def _paginate_results(
    fetch_method,
    season: int,
) -> dict[int, pd.DataFrame]:
    """
    Paginate via offset, MERGE round partials antar page.

    `fetch_method` = bound method seperti `Ergast.get_race_results`.
    Returns dict {round: DataFrame} dengan baris per-driver per-race.

    Note: Ergast hard-cap = 100 row/page. Sebuah round (≈20 driver) bisa
    ter-split di boundary antar page (mis. round 6 partial di akhir page 1 +
    sisanya di awal page 2). Strategy: kumpulkan partials per-round, lalu
    concat + dedupe by (round, driverCode) di akhir.
    """
    # Stage 1: kumpulkan SEMUA partial pages per round
    parts: dict[int, list[pd.DataFrame]] = {}
    race_names: dict[int, str] = {}
    circuit_ids: dict[int, str] = {}
    circuit_names: dict[int, str] = {}
    seen_offsets: set[int] = set()
    offset = 0

    while offset <= _MAX_OFFSET and offset not in seen_offsets:
        seen_offsets.add(offset)
        try:
            resp = fetch_method(
                season=season, limit=_PAGE_SIZE, offset=offset,
            )
        except Exception:
            # Network / API failure → return what we have so far
            break

        try:
            desc = resp.description
            contents = resp.content
        except AttributeError:
            break

        if desc is None or len(desc) == 0 or contents is None or len(contents) == 0:
            break

        total_rows = sum(len(c) for c in contents)
        if total_rows == 0:
            break

        for desc_row, content_df in zip(desc.itertuples(index=False), contents):
            try:
                rnd = int(desc_row.round)
            except (AttributeError, TypeError, ValueError):
                continue
            if content_df is None or len(content_df) == 0:
                continue
            df = content_df.copy() if hasattr(content_df, "copy") else pd.DataFrame(content_df)
            parts.setdefault(rnd, []).append(df)
            # Race + circuit metadata dari deskripsi (sama untuk semua partial)
            if rnd not in race_names:
                race_names[rnd] = str(getattr(desc_row, "raceName", ""))
                circuit_ids[rnd] = str(getattr(desc_row, "circuitId", ""))
                circuit_names[rnd] = str(getattr(desc_row, "circuitName", ""))

        # Stop kalau page ini lebih kecil dari _PAGE_SIZE → udah habis
        if total_rows < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE

    # Stage 2: concat partials + dedupe per round
    out: dict[int, pd.DataFrame] = {}
    for rnd, dfs in parts.items():
        if len(dfs) == 1:
            df = dfs[0]
        else:
            try:
                df = pd.concat(dfs, ignore_index=True)
            except Exception:
                # Fallback: ambil yang paling banyak rows
                df = max(dfs, key=len)
        # Dedupe by driverCode (kalau ada baris duplikat dari overlap pagination)
        if "driverCode" in df.columns:
            df = df.drop_duplicates(subset=["driverCode"], keep="first")
        df = df.reset_index(drop=True)
        df["_round"]        = rnd
        df["_raceName"]     = race_names.get(rnd, f"Round {rnd}")
        df["_circuitId"]    = circuit_ids.get(rnd, "")
        df["_circuitName"]  = circuit_names.get(rnd, "")
        out[rnd] = df

    return out


# ── Cached season fetchers ────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_season_races(season: int) -> dict[int, pd.DataFrame]:
    """
    Race results untuk seluruh season {round: df}.
    df schema: position, points, grid, status, driverCode, constructorName, ...
    Plus annotated: _round, _raceName.
    """
    E = _ergast()
    if E is None:
        return {}
    return _paginate_results(E.get_race_results, season)


@st.cache_data(ttl=86400, show_spinner=False)
def get_season_qualifying(season: int) -> dict[int, pd.DataFrame]:
    """
    Qualifying results untuk seluruh season {round: df}.
    df schema: position, Q1, Q2, Q3, driverCode, ...
    """
    E = _ergast()
    if E is None:
        return {}
    return _paginate_results(E.get_qualifying_results, season)


@st.cache_data(ttl=86400, show_spinner=False)
def get_season_sprints(season: int) -> dict[int, pd.DataFrame]:
    """
    Sprint results untuk season. Hanya round-round yang punya sprint.
    df schema: position, points, grid, status, driverCode, ...

    Sprint rules berubah lintas season:
    - 2022: sprint defines grid (no separate points sebagian).
    - 2023+: standalone, points P1-P8 (8/7/6/5/4/3/2/1).
    Kita tetap return apapun yang dikasih Ergast (dia handle scoring perubahan).
    """
    E = _ergast()
    if E is None:
        return {}
    return _paginate_results(E.get_sprint_results, season)


@st.cache_data(ttl=86400, show_spinner=False)
def get_season_drivers(season: int) -> list[dict]:
    """
    List of drivers in season — sorted by driverCode (abbr).
    Returns list of dict {code, full_name, team}.

    Source: union dari driverCode di race results + fallback ke
    Ergast.get_driver_info kalau race kosong.
    """
    races = get_season_races(season)
    drivers: dict[str, dict] = {}
    for rnd_df in races.values():
        if rnd_df is None or len(rnd_df) == 0:
            continue
        if "driverCode" not in rnd_df.columns:
            continue
        for _, row in rnd_df.iterrows():
            code = row.get("driverCode")
            if not isinstance(code, str) or len(code) == 0:
                continue
            if code in drivers:
                continue
            given = row.get("givenName", "") if pd.notna(row.get("givenName")) else ""
            family = row.get("familyName", "") if pd.notna(row.get("familyName")) else ""
            full = f"{given} {family}".strip() or code
            team = row.get("constructorName", "") if pd.notna(row.get("constructorName")) else ""
            drivers[code] = {
                "code":      code,
                "full_name": full,
                "team":      str(team),
            }

    return sorted(drivers.values(), key=lambda d: d["code"])


# ── Per-driver season stats aggregation ──────────────────────────────────────

def _race_points_for_driver(
    races: dict[int, pd.DataFrame],
    driver_code: str,
) -> tuple[float, list[tuple[int, float, int | None]]]:
    """
    Return (total_race_points, per_round_points)
    per_round = list of (round, points, finish_pos) untuk round-round di mana
    driver ini compete.
    """
    total = 0.0
    per_round: list[tuple[int, float, int | None]] = []
    for rnd in sorted(races.keys()):
        df = races[rnd]
        if df is None or len(df) == 0 or "driverCode" not in df.columns:
            continue
        match = df[df["driverCode"] == driver_code]
        if len(match) == 0:
            continue
        row = match.iloc[0]
        try:
            pts = float(row.get("points")) if pd.notna(row.get("points")) else 0.0
        except (TypeError, ValueError):
            pts = 0.0
        try:
            pos = (
                int(row.get("position"))
                if pd.notna(row.get("position")) else None
            )
        except (TypeError, ValueError):
            pos = None
        total += pts
        per_round.append((rnd, pts, pos))
    return total, per_round


def _sprint_points_for_driver(
    sprints: dict[int, pd.DataFrame],
    driver_code: str,
) -> tuple[float, dict[int, float]]:
    """Return (total_sprint_points, points_per_round)."""
    total = 0.0
    by_round: dict[int, float] = {}
    for rnd, df in sprints.items():
        if df is None or len(df) == 0 or "driverCode" not in df.columns:
            continue
        match = df[df["driverCode"] == driver_code]
        if len(match) == 0:
            continue
        try:
            pts = (
                float(match.iloc[0].get("points"))
                if pd.notna(match.iloc[0].get("points")) else 0.0
            )
        except (TypeError, ValueError):
            pts = 0.0
        total += pts
        by_round[rnd] = pts
    return total, by_round


def build_driver_season_summary(
    season: int,
    driver_code: str,
) -> dict:
    """
    Aggregate per-driver season stats:
    - total_points (race + sprint)
    - wins, podiums (race only)
    - poles (quali P1)
    - fastest_laps (race fastestLapRank=1 dengan finish ≤ P10 untuk +1 pt bonus)
    - dnfs (status bukan Finished / +N Lap(s))
    - n_races_entered
    Plus cumulative_points: list of (round, cumulative_total) untuk chart.

    Returns dict atau {} kalau driver tidak ditemukan / Ergast unavailable.
    """
    races   = get_season_races(season)
    quali   = get_season_qualifying(season)
    sprints = get_season_sprints(season)

    if not races:
        return {}

    race_pts, race_per_round = _race_points_for_driver(races, driver_code)
    sprint_pts, sprint_by_round = _sprint_points_for_driver(sprints, driver_code)

    # Wins, podiums, FL, DNFs dari race results
    wins = podiums = fastest_laps = dnfs = 0
    for rnd, _, pos in race_per_round:
        # Status check untuk DNF
        df = races.get(rnd)
        if df is None:
            continue
        match = df[df["driverCode"] == driver_code]
        if len(match) == 0:
            continue
        row = match.iloc[0]
        status = str(row.get("status", "")) if pd.notna(row.get("status")) else ""
        # "Finished" / "+N Lap(s)" = classified finish
        if not (status == "Finished" or "Lap" in status):
            dnfs += 1
        if pos is not None:
            if pos == 1:
                wins += 1
            if pos <= 3:
                podiums += 1
        # Fastest lap bonus: rank=1 AND classified position ≤ 10
        try:
            fl_rank = row.get("fastestLapRank")
            if pd.notna(fl_rank) and int(fl_rank) == 1 and pos is not None and pos <= 10:
                fastest_laps += 1
        except (TypeError, ValueError):
            pass

    # Poles dari quali
    poles = 0
    for rnd, df in quali.items():
        if df is None or len(df) == 0 or "driverCode" not in df.columns:
            continue
        match = df[df["driverCode"] == driver_code]
        if len(match) == 0:
            continue
        try:
            pos = (
                int(match.iloc[0].get("position"))
                if pd.notna(match.iloc[0].get("position")) else None
            )
        except (TypeError, ValueError):
            pos = None
        if pos == 1:
            poles += 1

    # Cumulative points trajectory (race + sprint per-round)
    cumulative: list[tuple[int, float]] = []
    running = 0.0
    for rnd in sorted(races.keys()):
        rnd_pts = 0.0
        df = races.get(rnd)
        if df is not None and "driverCode" in df.columns:
            match = df[df["driverCode"] == driver_code]
            if len(match) > 0:
                try:
                    rnd_pts += (
                        float(match.iloc[0].get("points"))
                        if pd.notna(match.iloc[0].get("points")) else 0.0
                    )
                except (TypeError, ValueError):
                    pass
        rnd_pts += sprint_by_round.get(rnd, 0.0)
        running += rnd_pts
        cumulative.append((rnd, running))

    return {
        "code":             driver_code,
        "race_points":      race_pts,
        "sprint_points":    sprint_pts,
        "total_points":     race_pts + sprint_pts,
        "wins":             wins,
        "podiums":          podiums,
        "poles":            poles,
        "fastest_laps":     fastest_laps,
        "dnfs":             dnfs,
        "n_races_entered":  len(race_per_round),
        "cumulative":       cumulative,
    }


# ── Per-GP head-to-head table ────────────────────────────────────────────────

def build_per_gp_h2h(
    season: int,
    driver_a: str,
    driver_b: str,
) -> pd.DataFrame:
    """
    Per-round H2H table:
        Round | GP | Quali A | Quali B | Race A | Race B | Pts A | Pts B | Winner
    Quali/Race = "Pn" string. Winner field: "A" / "B" / "—" untuk race.
    """
    races   = get_season_races(season)
    quali   = get_season_qualifying(season)
    sprints = get_season_sprints(season)

    if not races:
        return pd.DataFrame()

    def _pos(df, code) -> int | None:
        if df is None or len(df) == 0 or "driverCode" not in df.columns:
            return None
        m = df[df["driverCode"] == code]
        if len(m) == 0:
            return None
        try:
            v = m.iloc[0].get("position")
            return int(v) if pd.notna(v) else None
        except (TypeError, ValueError):
            return None

    def _pts(df, code) -> float:
        if df is None or len(df) == 0 or "driverCode" not in df.columns:
            return 0.0
        m = df[df["driverCode"] == code]
        if len(m) == 0:
            return 0.0
        try:
            v = m.iloc[0].get("points")
            return float(v) if pd.notna(v) else 0.0
        except (TypeError, ValueError):
            return 0.0

    rows = []
    for rnd in sorted(races.keys()):
        race_df = races[rnd]
        if race_df is None or len(race_df) == 0:
            continue
        gp_name = (
            str(race_df["_raceName"].iloc[0])
            if "_raceName" in race_df.columns and len(race_df) > 0
            else f"Round {rnd}"
        )

        q_a = _pos(quali.get(rnd), driver_a)
        q_b = _pos(quali.get(rnd), driver_b)
        r_a = _pos(race_df, driver_a)
        r_b = _pos(race_df, driver_b)

        # Race points (race + sprint kalau ada)
        pts_a = _pts(race_df, driver_a) + _pts(sprints.get(rnd), driver_a)
        pts_b = _pts(race_df, driver_b) + _pts(sprints.get(rnd), driver_b)

        # Race winner: lower position wins. None handling: a vs None → other wins.
        if r_a is not None and r_b is not None:
            race_w = driver_a if r_a < r_b else (driver_b if r_b < r_a else "—")
        elif r_a is not None:
            race_w = driver_a
        elif r_b is not None:
            race_w = driver_b
        else:
            race_w = "—"

        # Quali winner
        if q_a is not None and q_b is not None:
            quali_w = driver_a if q_a < q_b else (driver_b if q_b < q_a else "—")
        elif q_a is not None:
            quali_w = driver_a
        elif q_b is not None:
            quali_w = driver_b
        else:
            quali_w = "—"

        rows.append({
            "Round":   rnd,
            "GP":      gp_name,
            "Sprint":  rnd in sprints,
            f"Quali {driver_a}":  f"P{q_a}" if q_a else "—",
            f"Quali {driver_b}":  f"P{q_b}" if q_b else "—",
            f"Race {driver_a}":   f"P{r_a}" if r_a else "—",
            f"Race {driver_b}":   f"P{r_b}" if r_b else "—",
            f"Pts {driver_a}":    pts_a,
            f"Pts {driver_b}":    pts_b,
            "Race winner":        race_w,
            "Quali winner":       quali_w,
            "_q_a": q_a, "_q_b": q_b,
            "_r_a": r_a, "_r_b": r_b,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def compute_h2h_tally(per_gp_df: pd.DataFrame, driver_a: str, driver_b: str) -> dict:
    """
    Win-loss tally untuk Race & Quali — count seberapa sering driver_a finish
    di depan driver_b dan sebaliknya.
    """
    out = {
        "race_a": 0, "race_b": 0, "race_tied": 0,
        "quali_a": 0, "quali_b": 0, "quali_tied": 0,
    }
    if per_gp_df is None or len(per_gp_df) == 0:
        return out

    for _, row in per_gp_df.iterrows():
        ra, rb = row.get("_r_a"), row.get("_r_b")
        qa, qb = row.get("_q_a"), row.get("_q_b")

        if ra is not None and rb is not None:
            if ra < rb:
                out["race_a"] += 1
            elif rb < ra:
                out["race_b"] += 1
            else:
                out["race_tied"] += 1
        # Kalau salah satu None (DNF / DNS) — tidak counted untuk fair compare

        if qa is not None and qb is not None:
            if qa < qb:
                out["quali_a"] += 1
            elif qb < qa:
                out["quali_b"] += 1
            else:
                out["quali_tied"] += 1
    return out


# ── Circuit history (Mode C) ─────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_circuits_for_years(years: tuple[int, ...]) -> dict[str, dict]:
    """
    Map circuit → years/rounds di mana circuit itu dipakai.

    Returns dict {circuitId: {name, locality, years: {year: round}}}.
    circuitId stabil lintas musim (e.g. 'monaco', 'silverstone'), jadi cocok
    sebagai kunci cross-season.

    Source: get_season_races (sudah cached) — baca _circuitId / _circuitName.
    """
    circuits: dict[str, dict] = {}
    for year in years:
        races = get_season_races(year)
        for rnd in sorted(races.keys()):
            df = races[rnd]
            if df is None or len(df) == 0:
                continue
            cid = (
                str(df["_circuitId"].iloc[0])
                if "_circuitId" in df.columns else ""
            )
            if not cid:
                continue
            cname = (
                str(df["_circuitName"].iloc[0])
                if "_circuitName" in df.columns else cid
            )
            gp_name = (
                str(df["_raceName"].iloc[0])
                if "_raceName" in df.columns else cname
            )
            entry = circuits.setdefault(
                cid, {"name": cname, "gp_name": gp_name, "years": {}}
            )
            entry["years"][year] = rnd
    return circuits


def build_circuit_h2h(
    circuit_id: str,
    driver_a: str,
    driver_b: str,
    years: tuple[int, ...],
) -> pd.DataFrame:
    """
    Per-year H2H di satu circuit untuk 2 driver.

    Returns DataFrame, 1 row per year circuit itu dipakai:
        Year | Quali A | Quali B | Race A | Race B | Pts A | Pts B | Status A | Status B
    Plus hidden numeric kolom (_q_a, _r_a, dst) untuk chart & stat.
    'DNC' = did not compete (driver tidak ada di hasil tahun itu).
    """
    circuits = get_circuits_for_years(years)
    if circuit_id not in circuits:
        return pd.DataFrame()

    info = circuits[circuit_id]

    def _extract(df, code) -> dict:
        """Return {pos, pts, status, grid} untuk driver di sebuah results df."""
        empty = {"pos": None, "pts": None, "status": None, "grid": None, "found": False}
        if df is None or len(df) == 0 or "driverCode" not in df.columns:
            return empty
        m = df[df["driverCode"] == code]
        if len(m) == 0:
            return empty
        row = m.iloc[0]
        try:
            pos = int(row.get("position")) if pd.notna(row.get("position")) else None
        except (TypeError, ValueError):
            pos = None
        try:
            pts = float(row.get("points")) if pd.notna(row.get("points")) else 0.0
        except (TypeError, ValueError):
            pts = 0.0
        try:
            grid = int(row.get("grid")) if pd.notna(row.get("grid")) else None
        except (TypeError, ValueError):
            grid = None
        status = str(row.get("status")) if pd.notna(row.get("status")) else None
        return {"pos": pos, "pts": pts, "status": status, "grid": grid, "found": True}

    rows = []
    for year in sorted(info["years"].keys()):
        rnd = info["years"][year]
        races = get_season_races(year)
        quali = get_season_qualifying(year)
        sprints = get_season_sprints(year)

        race_df  = races.get(rnd)
        quali_df = quali.get(rnd)
        sprint_df = sprints.get(rnd)

        ra = _extract(race_df, driver_a)
        rb = _extract(race_df, driver_b)
        qa = _extract(quali_df, driver_a)
        qb = _extract(quali_df, driver_b)
        spa = _extract(sprint_df, driver_a)
        spb = _extract(sprint_df, driver_b)

        # Total points (race + sprint) per driver
        pts_a = (ra["pts"] or 0.0) + (spa["pts"] or 0.0) if ra["found"] else None
        pts_b = (rb["pts"] or 0.0) + (spb["pts"] or 0.0) if rb["found"] else None

        def _posfmt(d) -> str:
            if not d["found"]:
                return "DNC"
            return f"P{d['pos']}" if d["pos"] is not None else "—"

        rows.append({
            "Year":               year,
            "Sprint":             rnd in sprints,
            f"Quali {driver_a}":  _posfmt(qa),
            f"Quali {driver_b}":  _posfmt(qb),
            f"Race {driver_a}":   _posfmt(ra),
            f"Race {driver_b}":   _posfmt(rb),
            f"Pts {driver_a}":    pts_a,
            f"Pts {driver_b}":    pts_b,
            f"Status {driver_a}": ra["status"] or ("DNC" if not ra["found"] else "—"),
            f"Status {driver_b}": rb["status"] or ("DNC" if not rb["found"] else "—"),
            # Hidden numerics
            "_q_a": qa["pos"], "_q_b": qb["pos"],
            "_r_a": ra["pos"], "_r_b": rb["pos"],
            "_found_a": ra["found"], "_found_b": rb["found"],
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_circuit_career_stats(
    circuit_h2h_df: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> dict:
    """
    Per-driver career stats di circuit dari per-year H2H df.
    Returns {driver_code: {best_finish, avg_finish, wins, podiums, dnfs, starts}}.
    """
    out: dict[str, dict] = {}
    if circuit_h2h_df is None or len(circuit_h2h_df) == 0:
        return out

    for code, r_key, found_key in [
        (driver_a, "_r_a", "_found_a"),
        (driver_b, "_r_b", "_found_b"),
    ]:
        finishes: list[int] = []
        wins = podiums = dnfs = starts = 0
        for _, row in circuit_h2h_df.iterrows():
            if not row.get(found_key):
                continue
            starts += 1
            pos = row.get(r_key)
            status_col = f"Status {code}"
            status = str(row.get(status_col, "")) if status_col in row.index else ""
            # DNF detection
            if not (status == "Finished" or "Lap" in status):
                dnfs += 1
            if pos is not None:
                finishes.append(int(pos))
                if pos == 1:
                    wins += 1
                if pos <= 3:
                    podiums += 1
        out[code] = {
            "best_finish": min(finishes) if finishes else None,
            "avg_finish":  (sum(finishes) / len(finishes)) if finishes else None,
            "wins":        wins,
            "podiums":     podiums,
            "dnfs":        dnfs,
            "starts":      starts,
        }
    return out
