import pandas as pd

INPUT_CSV = "fight_logs.csv"
OUTPUT_CSV = "fight_features.csv"
NUM_FIRST_ACTIONS = 5  # first 5 actions

df = pd.read_csv(INPUT_CSV)

feature_rows = []

for fight_id, fight_data in df.groupby("fight_id"):

    # SYSTEM rows
    sys_rows = fight_data[fight_data["actor"] == "SYSTEM"]
    starting_distance = (float(sys_rows.iloc[0]["start_distance"])
    if "start_distance" in sys_rows.columns
    and not sys_rows["start_distance"].isna().all() else None)
    
    total_rounds = (
        int(sys_rows[sys_rows["action"] == "total_rounds"].iloc[0]["total_rounds"])
        if (sys_rows["action"] == "total_rounds").any()
        else None)

    actors = [a for a in fight_data["actor"].unique() if a != "SYSTEM"]

    for actor in actors:
        actor_data = fight_data[fight_data["actor"] == actor]

        atk_main = (actor_data["action"] == "atk_main").sum()
        atk_backup = (actor_data["action"] == "atk_backup").sum()
        move = actor_data["action"].str.contains("move").sum()
        gain_adv = (actor_data["action"] == "gain_adv").sum()
        inflict_disadv = (actor_data["action"] == "inflict_disadv").sum()
        hits = actor_data["hit"].sum(skipna=True)
        crits = actor_data["critical"].sum(skipna=True)
        total_damage = actor_data["damage"].sum(skipna=True)
        total_attacks = atk_main + atk_backup

        hit_rate = hits / total_attacks if total_attacks else 0
        crit_rate = crits / total_attacks if total_attacks else 0
        dmg_per_attack = total_damage / total_attacks if total_attacks else 0

        first_actions = actor_data["action"].tolist()[:NUM_FIRST_ACTIONS]
        while len(first_actions) < NUM_FIRST_ACTIONS:
            first_actions.append("none")

        feature_rows.append({
            "fight_id": fight_id,
            "actor": actor,
            "starting_distance": starting_distance,
            "total_rounds": total_rounds,
            "atk_main": atk_main,
            "atk_backup": atk_backup,
            "move": move,
            "gain_adv": gain_adv,
            "inflict_disadv": inflict_disadv,
            "hits": hits,
            "crits": crits,
            "total_damage": total_damage,
            "hit_rate": hit_rate,
            "crit_rate": crit_rate,
            "dmg_per_attack": dmg_per_attack,
            "action_1": first_actions[0],
            "action_2": first_actions[1],
            "action_3": first_actions[2],
            "action_4": first_actions[3],
            "action_5": first_actions[4]
        })

output_df = pd.DataFrame(feature_rows)
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved extracted features to {OUTPUT_CSV}")
print(f"Shape: {output_df.shape}")
