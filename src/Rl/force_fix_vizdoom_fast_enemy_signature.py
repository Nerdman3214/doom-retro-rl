from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_force_fast_enemy_signature_fix")
backup.write_text(text)
print(f"Backup saved to {backup}")

new_func = '''    def fast_enemy_reaction_action(self, action_name=None, action=None, *args, **kwargs):
        """
        Reward-only mode:
        Do not override PPO action.

        This accepts both:
        - action_name=...
        - action=...

        so old and new call sites both work.
        """

        if action_name is not None:
            return action_name

        if action is not None:
            return action

        return "move_forward"

'''

text, count = re.subn(
    r"    def fast_enemy_reaction_action\(self, .*?\):.*?(?=\n\n    def main_goal_progress_reward\(self, director_result\):)",
    new_func,
    text,
    flags=re.DOTALL,
)

if count != 1:
    raise SystemExit(f"Expected to replace fast_enemy_reaction_action once, replaced {count}")

p.write_text(text)
print("fast_enemy_reaction_action signature fixed.")
