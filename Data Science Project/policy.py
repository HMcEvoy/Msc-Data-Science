import torch
import torch.nn as nn
import torch.nn.functional as F

class CombatPolicy(nn.Module):
    """
    Simple NN outputting 5 possible actions:
    0 = move
    1 = atk_main
    2 = atk_backup
    3 = gain_advantage
    4 = inflict_disadvantage
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 12)
        self.fc2 = nn.Linear(12, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


def choose_action(policy, me, enemy):
    """Convert fight state into NN input vector."""
    vec = torch.tensor([
        me.hp / me.max_hp,
        enemy.hp / enemy.max_hp,
        abs(me.position - enemy.position),
        1.0 if me.gain_adv else 0.0,
        1.0 if enemy.inflict_disadv else 0.0,
        1.0 if me.inflict_disadv else 0.0
    ], dtype=torch.float32)

    probs = policy(vec)
    return torch.multinomial(probs, 1).item()
