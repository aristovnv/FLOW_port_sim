"""Configuration parameters for the port simulation."""

# Ship arrival parameters
SHIPS_DATA = [
    {"name": "CEZANNE", "containers": 2642},
    {"name": "CCNI ANDES", "containers": 2338},
    {"name": "CMA CGM LEO", "containers": 2187},
    {"name": "ONE REINFORCEMENT", "containers": 1752},
    {"name": "POLAR ECUADOR", "containers": 1431},
    {"name": "W KITHIRA", "containers": 1311},
    {"name": "SUAPE EXPRESS", "containers": 1062},
    {"name": "MAERSK ATHABASCA", "containers": 998},
    {"name": "MAERSK SENTOSA", "containers": 896},
    {"name": "MAERSK MONTE LINZOR", "containers": 892},
    {"name": "METHONI", "containers": 742},
    {"name": "TOCONAO", "containers": 621},
    {"name": "LOUIS MAERSK", "containers": 525},
    {"name": "SINE A", "containers": 490},
    {"name": "BSG BAHAMAS", "containers": 454},
    {"name": "SEASPAN EMPIRE", "containers": 317},
    {"name": "MSC UBERTY VIII", "containers": 228},
    {"name": "MSC NIOVI VIII", "containers": 227},
    {"name": "HYUNDAI HONG KONG", "containers": 169},
    {"name": "MSC PARIS", "containers": 120},
    {"name": "MSC TIANSHAN", "containers": 99},
    {"name": "WAN HAI A15", "containers": 91},
    {"name": "MSC TORONTO", "containers": 90},
    {"name": "WAN HAI A13", "containers": 79},
    {"name": "SAVANNAH EXPRESS", "containers": 75},
    {"name": "GJERTRUD MAERSK", "containers": 58},
    {"name": "ODYSSEUS", "containers": 44}
]

# Port configuration
MAX_BERTHS = 3
CRANES_PER_BERTH = 4
CONTAINER_UNLOAD_TIME = {
    "mean": 2.5,  # minutes
    "std": 0.5    # minutes
}
BERTH_TRANSITION_TIME = 60  # minutes (1 hour for ship departure/arrival)

# Container types and dwell times
CONTAINER_TYPES = {
    "dry": {
        "probability": 0.95,
        "dwell_time": {
            "mean": 2.47,  # days
            "std": 1.5     # days
        }
    },
    "reefer": {
        "probability": 0.05,
        "dwell_time": {
            "mean": 2.1,   # days
            "std": 0.8     # days
        }
    }
}

# Modal split
MODAL_SPLIT = {
    "truck": 0.85,
    "train": 0.15
}

# Gate operations
GATE_HOURS = {
    "open": 7,    # 7 AM
    "close": 17,  # 5 PM
    "trucks_per_hour": 500
}

# Train operations
TRAIN_SCHEDULE = {
    "trains_per_day": 6,
    "capacity": 250
}
