from policy import choose_action
from logger import append_to_buffer

def simulate_fight(charA, charB, policyA, policyB, fight_id):
    rounds = 0

    # Record starting distance
    start_distance = abs(charA.position - charB.position)
    append_to_buffer(
        fight_id,
        "SYSTEM",
        "start_distance",
        start_distance=start_distance
    )

    while charA.is_alive() and charB.is_alive() and rounds < 50:
        rounds += 1

        # Reset per-round flags
        charA.inflict_disadv = False
        charB.inflict_disadv = False

        # Turn order
        for actor, enemy, policy in [(charA, charB, policyA), (charB, charA,
                                                               policyB)]:
            if not actor.is_alive() or not enemy.is_alive():
                break

            action = choose_action(policy, actor, enemy)
            dist = abs(actor.position - enemy.position)

            # --------------------------
            # MOVE
            # --------------------------
            if action == 0:
                actor.move_towards(enemy)
                append_to_buffer(fight_id, actor.name, "move",
                                 distance=abs(actor.position - enemy.position))
                continue

            # --------------------------
            # MAIN ATTACK
            # --------------------------
            if action == 1:
                if actor.is_adjacent(enemy):
                    result = actor.attack(enemy, primary=True)
                    append_to_buffer(fight_id, actor.name, "atk_main",
                                     hit=result["hit"],
                                     critical=result["critical"],
                                     damage=result["damage"],
                                     distance=dist)
                else:
                    actor.move_towards(enemy)
                    append_to_buffer(fight_id, actor.name, "move_forced",
                                     distance=abs(actor.position - enemy.position))
                continue

            # --------------------------
            # BACKUP ATTACK
            # --------------------------
            if action == 2:
                if not actor.is_adjacent(enemy):
                    result = actor.attack(enemy, primary=False)
                    append_to_buffer(fight_id, actor.name, "atk_backup",
                                     hit=result["hit"],
                                     critical=result["critical"],
                                     damage=result["damage"],
                                     distance=dist)
                else:
                    result = actor.attack(enemy, primary=True)
                    append_to_buffer(fight_id, actor.name, "atk_main_forced",
                                     hit=result["hit"],
                                     critical=result["critical"],
                                     damage=result["damage"],
                                     distance=dist)
                continue

            # --------------------------
            # GAIN ADVANTAGE
            # --------------------------
            if action == 3 and dist == 0:
                actor.apply_gain_advantage()
                append_to_buffer(fight_id, actor.name, "gain_adv",
                                 distance=dist)
                continue

            # --------------------------
            # INFLICT DISADVANTAGE
            # --------------------------
            if action == 4 and dist == 0:
                actor.apply_inflict_disadvantage()
                append_to_buffer(fight_id, actor.name, "inflict_disadv",
                                 distance=dist)
                continue

    # Record total rounds in a SYSTEM row
    append_to_buffer(fight_id, "SYSTEM", "total_rounds", total_rounds=rounds)

    winner = charA.name if charA.is_alive() else charB.name
    return winner
