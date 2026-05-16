"""
Central config — all constants live here.
Edit this file, not scattered magic strings throughout the codebase.
"""

CACHE_DIR = "data/cache"

SUPPORTED_YEARS = [2022, 2023, 2024]

# F1 rounds per season
F1_ROUNDS = {
    2022: {
        1:  "Bahrain",
        2:  "Saudi Arabia",
        3:  "Australia",
        4:  "Emilia Romagna",
        5:  "Miami",
        6:  "Spain",
        7:  "Monaco",
        8:  "Azerbaijan",
        9:  "Canada",
        10: "United Kingdom",
        11: "Austria",
        12: "France",
        13: "Hungary",
        14: "Belgium",
        15: "Netherlands",
        16: "Italy",
        17: "Singapore",
        18: "Japan",
        19: "United States",
        20: "Mexico",
        21: "Brazil",
        22: "Abu Dhabi",
    },
    2023: {
        1:  "Bahrain",
        2:  "Saudi Arabia",
        3:  "Australia",
        4:  "Azerbaijan",
        5:  "Miami",
        6:  "Monaco",
        7:  "Spain",
        8:  "Canada",
        9:  "Austria",
        10: "United Kingdom",
        11: "Hungary",
        12: "Belgium",
        13: "Netherlands",
        14: "Italy",
        15: "Singapore",
        16: "Japan",
        17: "Qatar",
        18: "United States",
        19: "Mexico",
        20: "Brazil",
        21: "Las Vegas",
        22: "Abu Dhabi",
    },
    2024: {
        1:  "Bahrain",
        2:  "Saudi Arabia",
        3:  "Australia",
        4:  "Japan",
        5:  "China",
        6:  "Miami",
        7:  "Emilia Romagna",
        8:  "Monaco",
        9:  "Canada",
        10: "Spain",
        11: "Austria",
        12: "United Kingdom",
        13: "Hungary",
        14: "Belgium",
        15: "Netherlands",
        16: "Italy",
        17: "Azerbaijan",
        18: "Singapore",
        19: "United States",
        20: "Mexico",
        21: "Brazil",
        22: "Las Vegas",
        23: "Qatar",
        24: "Abu Dhabi",
    },
}

# Backward compat
F1_2024_ROUNDS = F1_ROUNDS[2024]

# Driver colors (hex) — covers 2022, 2023, 2024 rosters.
# Catatan: warna di-map ke "team yang paling sering dipakai driver itu di rentang
# 2022-2024". Contoh: GAS ada di AlphaTauri 2022 lalu Alpine 2023+, kita pakai
# warna Alpine. Untuk 1-driver-1-warna, beberapa pendekatan kompromi.
DRIVER_COLORS = {
    # Red Bull
    "VER": "#3671C6",
    "PER": "#3671C6",
    # Ferrari
    "LEC": "#E8002D",
    "SAI": "#E8002D",
    # McLaren
    "NOR": "#FF8000",
    "PIA": "#FF8000",
    "RIC": "#FF8000",  # 2022/2023 McLaren (post-Renault). Default: McLaren
    # Mercedes
    "HAM": "#27F4D2",
    "RUS": "#27F4D2",
    # Aston Martin
    "ALO": "#358C75",  # 2023+ Aston Martin (sebelumnya Alpine 2022)
    "STR": "#358C75",
    "VET": "#358C75",  # 2022 Aston Martin (retired)
    # Alpine
    "GAS": "#0093CC",  # 2023+ Alpine (sebelumnya AlphaTauri 2022)
    "OCO": "#0093CC",
    # Williams
    "ALB": "#64C4FF",
    "SAR": "#64C4FF",  # 2023-2024
    "LAT": "#64C4FF",  # 2022 Williams (retired after)
    # VCARB / RB / AlphaTauri
    "TSU": "#6692FF",
    "DEV": "#6692FF",  # 2023 AlphaTauri substitute
    "LAW": "#6692FF",  # 2023 AlphaTauri substitute, 2024 RB
    # Haas
    "HUL": "#B6BABD",  # 2023+ Haas
    "MAG": "#B6BABD",
    "MSC": "#B6BABD",  # 2022 Haas
    # Sauber / Alfa Romeo
    "BOT": "#52E252",
    "ZHO": "#52E252",
}

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Ferrari": "#E8002D",
    "McLaren": "#FF8000",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#358C75",
    "Alpine": "#0093CC",
    "Williams": "#64C4FF",
    "RB": "#6692FF",
    "Haas F1 Team": "#B6BABD",
    "Kick Sauber": "#52E252",
}

# Compound tyre colors (official Pirelli palette)
COMPOUND_COLORS = {
    "SOFT":    "#FF3333",
    "MEDIUM":  "#FFD700",
    "HARD":    "#FFFFFF",
    "INTER":   "#39B54A",
    "WET":     "#0067FF",
    "UNKNOWN": "#888888",
}

SESSION_TYPES = ["R", "Q", "FP1", "FP2", "FP3", "SQ"]
SESSION_LABELS = {
    "R":   "Race",
    "Q":   "Qualifying",
    "SQ":  "Sprint Qualifying",
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
}

YEAR = 2024
