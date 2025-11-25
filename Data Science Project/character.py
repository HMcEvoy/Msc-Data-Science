import d20
import re

def safe_int(value):
    """Extract integer from d20 roll results or strings."""
    if isinstance(value, int):
        return value
    if hasattr(value, "total"):
        value = value.total
    match = re.search(r"-?\d+", str(value))
    if not match:
        raise ValueError(f"Cannot convert d20 output to int: {value}")
    return int(match.group())


class MartialCharacter:
    def __init__(self, name, hp, ac, attack_bonus, dmg_main, dmg_backup,
                 speed):
        self.name = name
        self.max_hp = hp
        self.hp = hp
        self.ac = ac
        self.attack_bonus = attack_bonus
        self.damage_main = dmg_main
        self.damage_backup = dmg_backup
        self.speed = speed
        self.position = 0

        # combat flags
        self.gain_adv = False
        self.inflict_disadv = False

    # ---------------------------------------------
    def is_alive(self):
        return self.hp > 0

    # ---------------------------------------------
    def is_adjacent(self, target):
        """Adjacency = distance == 0 for melee attacks."""
        return abs(self.position - target.position) == 0

    # ---------------------------------------------
    def attack_roll(self, advantage=False, disadvantage=False):
        """d20 roll with advantage/disadvantage logic."""
        if advantage and not disadvantage:
            roll = d20.roll("2d20kh1")
        elif disadvantage and not advantage:
            roll = d20.roll("2d20kl1")
        else:
            roll = d20.roll("1d20")
        return safe_int(roll)

    # ---------------------------------------------
    def roll_damage(self, expr, critical=False):
        """Roll damage. Double dice if critical."""
        dmg1 = safe_int(d20.roll(expr))
        if critical:
            dmg2 = safe_int(d20.roll(expr))
            return dmg1 + dmg2
        return dmg1

    # ---------------------------------------------
    def attack(self, target, primary=True):
        """Resolve attack, including hit/miss and crit logic."""
        advantage = self.gain_adv
        disadvantage = target.inflict_disadv
        self.gain_adv = False  # reset after use

        roll = self.attack_roll(advantage, disadvantage)

        # Natural 1 → automatic miss
        if roll == 1:
            return {"hit": False, "critical": False, "roll": roll, "damage": 0}

        # Natural 20 → critical
        if roll == 20:
            dmg = (self.roll_damage(self.damage_main
                                    if primary
                                    else self.damage_backup,
                                   critical=True))
            target.hp -= dmg
            return {"hit": True, "critical": True, "roll": roll, "damage": dmg}

        # Normal hit calculation
        attack_total = roll + self.attack_bonus
        hit = attack_total >= target.ac

        if hit:
            dmg = (self.roll_damage(self.damage_main
                                    if primary
                                    else self.damage_backup))
            target.hp -= dmg
            return {"hit": True, "critical": False, "roll": attack_total,
                    "damage": dmg}

        return {"hit": False, "critical": False, "roll": attack_total,
                "damage": 0}

    # ---------------------------------------------
    def move_towards(self, target):
        """1D linear movement toward the opponent."""
        if self.position < target.position:
            self.position += min(self.speed, target.position - self.position)
        else:
            self.position -= min(self.speed, self.position - target.position)

    # ---------------------------------------------
    def apply_gain_advantage(self):
        self.gain_adv = True

    def apply_inflict_disadvantage(self):
        self.inflict_disadv = True
