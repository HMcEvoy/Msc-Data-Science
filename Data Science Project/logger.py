import csv

LOG_BUFFER = []

CSV_HEADER = [
    "fight_id",
    "actor",
    "action",
    "hit",
    "critical",
    "damage",
    "distance",
    "start_distance",
    "total_rounds",
    "max_hp",
    "hp",
    "ac",
    "attack_bonus",
    "speed",
    "dmg_main",
    "dmg_backup"
]

def append_to_buffer(fight_id, actor, action, hit=None, critical=None,
                     damage=None, distance=None, start_distance=None,
                     total_rounds=None, max_hp=None, hp=None, ac=None,
                     attack_bonus=None, speed=None, dmg_main=None, dmg_backup=None):
    """Store one row in the buffer."""
    LOG_BUFFER.append([
        fight_id, actor, action, hit, critical, damage, distance,
        start_distance, total_rounds, max_hp, hp, ac, attack_bonus,
        speed, dmg_main, dmg_backup
    ])

def save_buffer_to_csv(filename="fight_logs.csv"):
    """Write buffer to CSV and clear it."""
    global LOG_BUFFER
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(LOG_BUFFER)
    print(f"Saved {len(LOG_BUFFER)} rows into {filename}")
    LOG_BUFFER = []

