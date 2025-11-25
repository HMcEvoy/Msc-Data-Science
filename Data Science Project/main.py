from character import MartialCharacter
from policy import CombatPolicy
from simulate import simulate_fight
from logger import save_buffer_to_csv
import random

NUM_FIGHTS = 1000

def run_batch():
    print(f"Running {NUM_FIGHTS} simulated fights...")

    for fight_id in range(1, NUM_FIGHTS + 1):
        A = MartialCharacter("A", hp=45, ac=15, attack_bonus=7, 
                             dmg_main="1d12+4", dmg_backup="1d6+4", speed=4)
        B = MartialCharacter("B", hp=36, ac=18, attack_bonus=6, 
                             dmg_main="1d8+5", dmg_backup="1d6+5", speed=3)

        # Random starting distance
        start_distance = random.randint(0, 10)
        A.position = 0
        B.position = start_distance

        policyA = CombatPolicy()
        policyB = CombatPolicy()

        simulate_fight(A, B, policyA, policyB, fight_id)

        if fight_id % 50 == 0:
            print(f"Progress: {fight_id}/{NUM_FIGHTS} fights complete")

    save_buffer_to_csv()
    print(f"All fights saved to fight_logs.csv")

if __name__ == "__main__":
    run_batch()
