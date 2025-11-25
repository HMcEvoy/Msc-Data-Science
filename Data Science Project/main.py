from character import MartialCharacter
from policy import CombatPolicy
from simulate import simulate_fight
from logger import save_buffer_to_csv
import random

NUM_FIGHTS = 10000

def run_batch():
    print(f"Running {NUM_FIGHTS} simulated fights...")

    for fight_id in range(1, NUM_FIGHTS + 1):

        # Randomize A stats
        A_hp = random.randint(30, 50)
        A_ac = random.randint(14, 18)
        A_attack_bonus = random.randint(5, 8)
        A_speed = random.randint(3, 5)
        A_dmg_main = f"1{random.choice(['d8', 'd10', 'd12'])}+4"
        A_dmg_backup = f"1{random.choice(['d6','d8'])}+4"

        # Randomize B stats
        B_hp = random.randint(30, 50)
        B_ac = random.randint(14, 18)
        B_attack_bonus = random.randint(5, 8)
        B_speed = random.randint(3, 5)
        B_dmg_main = f"1{random.choice(['d8', 'd10', 'd12'])}+4"
        B_dmg_backup = f"1{random.choice(['d6','d8'])}+4"

        # Random starting distance
        start_distance = random.randint(0, 10)

        A = MartialCharacter("A", hp=A_hp, ac=A_ac, attack_bonus=A_attack_bonus,
                             dmg_main=A_dmg_main, dmg_backup=A_dmg_backup, speed=A_speed)
        B = MartialCharacter("B", hp=B_hp, ac=B_ac, attack_bonus=B_attack_bonus,
                             dmg_main=B_dmg_main, dmg_backup=B_dmg_backup, speed=B_speed)

        # Set positions
        A.position = 0
        B.position = start_distance

        # Create policies
        policyA = CombatPolicy()
        policyB = CombatPolicy()

        # Simulate fight
        simulate_fight(A, B, policyA, policyB, fight_id)

        if fight_id % 50 == 0:
            print(f"Progress: {fight_id}/{NUM_FIGHTS} fights complete")

    # Save logs
    save_buffer_to_csv()
    print("All fights saved to fight_logs.csv")


if __name__ == "__main__":
    run_batch()
