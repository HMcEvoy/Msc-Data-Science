import pandas as pd

INPUT_CSV = "fight_logs.csv"
OUTPUT_CSV = "fight_features.csv"

df = pd.read_csv(INPUT_CSV)

feature_rows = []

for fight_id, fight_data in df.groupby("fight_id"):

    # Starting distance
    sys_rows = fight_data[fight_data["actor"] == "SYSTEM"]
    starting_distance = (
        float(sys_rows.iloc[0]["start_distance"])
        if "start_distance" in sys_rows.columns
        and not sys_rows["start_distance"].isna().all()
        else None
    )

    # Winner
    winner_rows = fight_data[fight_data["action"] == "winner"]
    winner_name = winner_rows.iloc[0]["actor"] if not winner_rows.empty else None

    # Pre-fight stats
    pre_rows = fight_data[fight_data["action"] == "pre_fight_stats"]
    a_row = pre_rows[pre_rows["actor"] == "A"]
    b_row = pre_rows[pre_rows["actor"] == "B"]

    feature_rows.append({
        "fight_id": fight_id,
        "A_hp": int(a_row["max_hp"].values[0]) if not a_row.empty else None,
        "A_ac": int(a_row["ac"].values[0]) if not a_row.empty else None,
        "A_attack_bonus": int(a_row["attack_bonus"].values[0]) if not a_row.empty else None,
        "A_speed": int(a_row["speed"].values[0]) if not a_row.empty else None,
        "A_dmg_main": a_row["dmg_main"].values[0] if not a_row.empty else None,
        "A_dmg_backup": a_row["dmg_backup"].values[0] if not a_row.empty else None,
        "B_hp": int(b_row["max_hp"].values[0]) if not b_row.empty else None,
        "B_ac": int(b_row["ac"].values[0]) if not b_row.empty else None,
        "B_attack_bonus": int(b_row["attack_bonus"].values[0]) if not b_row.empty else None,
        "B_speed": int(b_row["speed"].values[0]) if not b_row.empty else None,
        "B_dmg_main": b_row["dmg_main"].values[0] if not b_row.empty else None,
        "B_dmg_backup": b_row["dmg_backup"].values[0] if not b_row.empty else None,
        "starting_distance": starting_distance,
        "winner": winner_name
    })

output_df = pd.DataFrame(feature_rows)
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved extracted features to {OUTPUT_CSV}")
print(f"Shape: {output_df.shape}")
