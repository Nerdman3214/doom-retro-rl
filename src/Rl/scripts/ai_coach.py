"""ai_coach.py — Interactive NLP coach for the DOOM RL agent.

Run from src/Rl/:
    python scripts/ai_coach.py               # analyze most recent clip
    python scripts/ai_coach.py --clip N      # analyze clip_N.pkl
    python scripts/ai_coach.py --all         # walk through every clip

The coach:
  1. Plays a clip and analyses what the agent is doing.
  2. Describes problem patterns in plain English.
  3. Uses an LLM (ollama or OpenAI) for richer commentary if available.
  4. Lets you type free-text feedback, e.g. "stop spinning and push forward".
  5. Maps your words to concrete reward-weight changes.
  6. Saves the adjustments to coach_directives.json (picked up at next training).
  7. Gives tips on how to record your own demos for the agent to learn from.
"""

import glob
import json
import os
import pickle
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.preview_clip import play_clip

CLIPS_DIR       = "clips"
DIRECTIVES_FILE = "coach_directives.json"

# ── reward keyword → {reward_key: multiplier_delta} ───────────────────────
# Positive deltas increase reward (or increase penalty magnitude).
# Format: keyword → dict of reward_key → delta applied to its multiplier.

EFFECTS = {
    # Things the user wants MORE of
    "forward":    {"forward_exploration_bonus": +0.8, "movement_bonus": +0.5},
    "explore":    {"forward_exploration_bonus": +1.0, "new_area_discovered": +0.5,
                   "door_opened": +0.3},
    "push":       {"forward_exploration_bonus": +0.5, "closing_distance_bonus": +0.3},
    "aggressive": {"shoot_when_enemy_visible": +0.8, "accurate_fire_bonus": +0.5},
    "shoot":      {"shoot_when_enemy_visible": +0.8, "accurate_fire_bonus": +0.5},
    "attack":     {"shoot_when_enemy_visible": +0.8, "enemy_hit_flash": +0.3},
    "kill":       {"enemy_killed": +0.5, "shoot_when_enemy_visible": +0.8},
    "aim":        {"accurate_fire_bonus": +1.0, "enemy_centered": +0.5},
    "accurate":   {"accurate_fire_bonus": +1.0},
    "health":     {"medkit_collected": +0.8, "found_health_pack": +0.5,
                   "searching_for_medkit": +0.3},
    "heal":       {"medkit_collected": +0.8, "found_health_pack": +0.5},
    "medkit":     {"medkit_collected": +0.8},
    "ammo":       {"ammo_collected": +0.8, "searching_for_ammo": +0.3},
    "items":      {"medkit_collected": +0.5, "ammo_collected": +0.5,
                   "pickup_flash": +0.3},
    "pickup":     {"medkit_collected": +0.5, "ammo_collected": +0.5,
                   "pickup_flash": +0.3},
    "doors":      {"door_opened": +1.0},
    "door":       {"door_opened": +1.0},
    "survive":    {"smart_retreat": +0.3},
    "alive":      {"smart_retreat": +0.3},
    "dodge":      {"damage_strafe_escape": +0.8, "damage_backward_escape": +0.3},
    "strafe":     {"damage_strafe_escape": +0.8},

    # Things the user wants LESS of — keywords describe BAD behaviour.
    # These always add to the penalty (negation words don't affect them).
    "spinning":   {"rotation_loop_penalty": +1.0},
    "circles":    {"rotation_loop_penalty": +1.0},
    "camping":    {"stuck_penalty": +1.0, "revisited_area_penalty": +0.8},
    "stuck":      {"stuck_penalty": +1.0},
    "walls":      {"wall_shot_penalty": +0.8, "stuck_penalty": +0.3},
    "retreating": {"smart_retreat": -0.5, "damage_backward_escape": -0.3},
    "spamming":   {"interaction_bonus": -0.5},
    "passive":    {"shoot_when_enemy_visible": +0.5},
}

# These keywords describe BAD behaviour — negation has no meaningful effect.
PENALTY_KEYWORDS = {"spinning", "circles", "camping", "stuck", "walls",
                    "retreating", "spamming"}

# Two-word phrases checked before single words.
BIGRAM_EFFECTS = {
    "move forward":  {"forward_exploration_bonus": +0.8, "movement_bonus": +0.5},
    "push forward":  {"forward_exploration_bonus": +1.0, "movement_bonus": +0.5},
    "run away":      {"smart_retreat": -0.5},
    "move backward": {"smart_retreat": -0.3},
    "open doors":    {"door_opened": +1.0},
    "pick up":       {"medkit_collected": +0.5, "ammo_collected": +0.5},
    "stay alive":    {"smart_retreat": +0.3},
    "spin around":   {"rotation_loop_penalty": +1.0},
    "stop spinning": {"rotation_loop_penalty": +1.0},
}

NEGATION_WORDS = {"stop", "dont", "don't", "not", "less", "avoid",
                  "no", "without", "reduce", "decrease", "never"}


# ── clip analysis ──────────────────────────────────────────────────────────

def analyze_clip(path: str) -> dict:
    with open(path, "rb") as f:
        clip = pickle.load(f)

    actions = [action for _, action in clip]
    counts  = Counter(actions)
    total   = max(len(actions), 1)

    shoot_actions = {"shoot", "move_forward_shoot", "turn_left_shoot",
                     "turn_right_shoot", "move_backward_shoot", "strafe_left_shoot",
                     "strafe_right_shoot"}
    rotate_actions = {"turn_left", "turn_right", "turn_left_shoot",
                      "turn_right_shoot"}

    shoot_pct  = sum(counts.get(a, 0) for a in shoot_actions) / total
    fwd_pct    = (counts.get("move_forward", 0) +
                  counts.get("move_forward_shoot", 0)) / total
    back_pct   = (counts.get("move_backward", 0) +
                  counts.get("move_backward_shoot", 0)) / total
    rot_pct    = sum(counts.get(a, 0) for a in rotate_actions) / total
    use_pct    = counts.get("use", 0) / total

    return {
        "total_frames": total,
        "counts":        counts,
        "shoot_pct":     shoot_pct,
        "fwd_pct":       fwd_pct,
        "back_pct":      back_pct,
        "rot_pct":       rot_pct,
        "use_pct":       use_pct,
    }


def detect_issues(a: dict) -> list[str]:
    issues = []
    if a["use_pct"] > 0.15:
        issues.append(
            f"Spamming USE ({a['use_pct']:.0%} of frames) — "
            "likely pressing at walls, not just doors")
    if a["rot_pct"] > 0.40:
        issues.append(
            f"Excessive rotation ({a['rot_pct']:.0%}) — may be confused, "
            "stuck or just tracking enemies poorly")
    if a["back_pct"] > 0.20:
        issues.append(
            f"Frequent backward movement ({a['back_pct']:.0%}) — "
            "retreating too much or avoiding combat")
    if a["fwd_pct"] < 0.10:
        issues.append(
            f"Rarely moving forward ({a['fwd_pct']:.0%}) — "
            "not exploring the map")
    if a["shoot_pct"] < 0.10:
        issues.append(
            f"Shooting very little ({a['shoot_pct']:.0%}) — overly passive")
    return issues


def describe_clip(a: dict) -> str:
    parts = []

    if a["fwd_pct"] > 0.30:
        parts.append("pushing forward and exploring")
    elif a["fwd_pct"] > 0.10:
        parts.append("doing some forward movement")
    else:
        parts.append("barely moving forward")

    if a["shoot_pct"] > 0.40:
        parts.append("firing very frequently")
    elif a["shoot_pct"] > 0.15:
        parts.append("shooting a moderate amount")
    else:
        parts.append("rarely shooting")

    if a["rot_pct"] > 0.40:
        parts.append("turning a lot (possibly confused or tracking enemies)")
    elif a["rot_pct"] > 0.20:
        parts.append("scanning left and right")

    if a["back_pct"] > 0.20:
        parts.append("retreating backward often")

    if a["use_pct"] > 0.15:
        parts.append("pressing USE frequently")

    description = "The agent spent this clip " + ", ".join(parts) + "."

    top3 = a["counts"].most_common(3)
    top_str = ", ".join(f"{act} ({n}×)" for act, n in top3)
    description += f"\nTop actions: {top_str}."

    return description


# ── LLM helpers (optional) ─────────────────────────────────────────────────

def _get_ollama_model() -> str | None:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            models = r.json().get("models", [])
            if models:
                return models[0]["name"]
    except Exception:
        pass
    return None


def _ask_llm(system: str, user_msg: str) -> str | None:
    """Try ollama first, then OpenAI if OPENAI_API_KEY is set."""
    model = _get_ollama_model()
    if model:
        try:
            import requests
            payload = {
                "model":    model,
                "messages": [{"role": "system", "content": system},
                              {"role": "user",   "content": user_msg}],
                "stream":   False,
            }
            r = requests.post("http://localhost:11434/api/chat",
                              json=payload, timeout=15)
            if r.status_code == 200:
                return r.json()["message"]["content"]
        except Exception:
            pass

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": user_msg}],
                max_tokens=400,
            )
            return resp.choices[0].message.content
        except Exception:
            pass

    return None


COACH_SYSTEM = (
    "You are a DOOM game AI trainer. You analyse clips of an RL agent "
    "playing DOOM and give short, practical advice on how to improve its "
    "reward function. Be direct and specific. Max 4 sentences."
)


# ── feedback parsing ───────────────────────────────────────────────────────

def parse_feedback(text: str) -> dict:
    """Return {reward_key: delta} from free-text feedback."""
    words = [w.strip(".,!?;:") for w in text.lower().split()]
    deltas: dict = {}

    # Check bigrams first
    bigram_hits = set()
    for i in range(len(words) - 1):
        phrase = words[i] + " " + words[i + 1]
        if phrase in BIGRAM_EFFECTS:
            window = words[max(0, i - 2):i]
            negated = any(w in NEGATION_WORDS for w in window)
            for k, d in BIGRAM_EFFECTS[phrase].items():
                deltas[k] = deltas.get(k, 0) + (-d if negated else d)
            bigram_hits.update({i, i + 1})

    # Then single words
    for i, word in enumerate(words):
        if i in bigram_hits or word not in EFFECTS:
            continue

        effect = EFFECTS[word]
        negated = (word not in PENALTY_KEYWORDS and
                   any(w in NEGATION_WORDS for w in words[max(0, i - 3):i]))

        for k, d in effect.items():
            deltas[k] = deltas.get(k, 0) + (-d if negated else d)

    return deltas


def explain_deltas(deltas: dict) -> str:
    if not deltas:
        return "  (no recognisable keywords found)"
    lines = []
    for k, d in sorted(deltas.items(), key=lambda x: -abs(x[1])):
        arrow = "▲" if d > 0 else "▼"
        lines.append(f"  {arrow} {k:35s}  {d:+.2f}")
    return "\n".join(lines)


# ── directives persistence ─────────────────────────────────────────────────

def load_directives() -> dict:
    if os.path.exists(DIRECTIVES_FILE):
        with open(DIRECTIVES_FILE) as f:
            return json.load(f)
    return {"version": 1, "reward_multipliers": {}, "feedback_history": []}


def save_directives(directives: dict):
    with open(DIRECTIVES_FILE, "w") as f:
        json.dump(directives, f, indent=2)
    print(f"  Saved → {DIRECTIVES_FILE}")


def apply_deltas(directives: dict, deltas: dict, feedback_text: str):
    mults = directives.setdefault("reward_multipliers", {})
    for k, d in deltas.items():
        current = mults.get(k, 1.0)
        mults[k] = max(0.05, round(current + d, 3))

    from datetime import date
    directives.setdefault("feedback_history", []).append(
        f"{date.today()}: {feedback_text}"
    )


# ── printing helpers ───────────────────────────────────────────────────────

SEPARATOR = "─" * 56


def print_analysis(clip_name: str, a: dict):
    print(f"\n{SEPARATOR}")
    print(f"  Clip: {clip_name}  ({a['total_frames']} frames)")
    print(SEPARATOR)

    total = a["total_frames"]
    for action, count in a["counts"].most_common():
        bar = "█" * min(30, int(count / total * 60))
        print(f"  {action:30s} {count:4d}  {bar}")

    print()
    issues = detect_issues(a)
    if issues:
        print("Issues detected:")
        for iss in issues:
            print(f"  ⚠  {iss}")
    else:
        print("  ✓  No major issues detected.")
    print()


def print_directives(directives: dict):
    mults = directives.get("reward_multipliers", {})
    if not mults:
        print("  (no active directives)")
        return
    print("  Active reward multipliers:")
    for k, v in sorted(mults.items()):
        bar = "█" * int(min(v, 4.0) * 5)
        print(f"    {k:35s}  ×{v:.2f}  {bar}")


# ── interactive session ────────────────────────────────────────────────────

def coach_clip(clip_path: str):
    clip_name = os.path.basename(clip_path)
    a = analyze_clip(clip_path)

    print_analysis(clip_name, a)

    # Natural language description
    description = describe_clip(a)
    issues = detect_issues(a)
    llm_response = _ask_llm(
        COACH_SYSTEM,
        f"Clip summary:\n{description}\n\nIssues:\n" +
        ("\n".join(issues) if issues else "None detected."),
    )
    if llm_response:
        print("AI coach says:")
        for line in llm_response.splitlines():
            print(f"  {line}")
    else:
        print(description)

    print()
    directives = load_directives()

    # Multi-turn feedback loop
    while True:
        print(SEPARATOR)
        print("What should the AI improve? (free text, or: show / tips / quit)")
        print("  Examples: 'push forward and stop spinning'")
        print("            'shoot more aggressively, avoid camping'")
        print("            'learn to pick up health and open doors'")
        feedback = input("> ").strip()

        if not feedback or feedback.lower() == "quit":
            break

        if feedback.lower() == "show":
            print_directives(directives)
            continue

        if feedback.lower() == "tips":
            _print_tips()
            continue

        deltas = parse_feedback(feedback)
        if not deltas:
            print("\n  Couldn't map that to reward changes. Try words like:")
            print("  forward, explore, shoot, dodge, aim, health, doors, ")
            print("  spinning, camping, retreating, spamming, aggressive…")
            print()
            # Ask LLM to translate if available
            suggestion = _ask_llm(
                COACH_SYSTEM,
                f"The user said: '{feedback}'. Rephrase this as advice "
                "using these words: forward, explore, shoot, aim, health, "
                "doors, spinning, camping, retreating, aggressive, dodge.",
            )
            if suggestion:
                print(f"  LLM suggestion: {suggestion}")
                print()
            continue

        print("\nReward changes that will be applied:")
        print(explain_deltas(deltas))

        # Show LLM interpretation if available
        llm_plan = _ask_llm(
            COACH_SYSTEM,
            f"User feedback: '{feedback}'. Reward deltas: {deltas}. "
            "Confirm in one sentence what you will make the agent do differently.",
        )
        if llm_plan:
            print(f"\n  AI confirms: {llm_plan.strip()}")

        print()
        confirm = input("Apply these changes? (Y/n): ").strip().lower()
        if confirm in ("", "y", "yes"):
            apply_deltas(directives, deltas, feedback)
            save_directives(directives)
            print("  ✓ Directives updated. They take effect on the next training run.")
        else:
            print("  Skipped.")

        again = input("\nAnything else to improve? (y/N): ").strip().lower()
        if again != "y":
            break

    print(f"\n{SEPARATOR}")
    print("Current directives saved in:")
    print(f"  {os.path.abspath(DIRECTIVES_FILE)}")
    print("Start training to apply them:")
    print("  python training/train_rl_agent.py")
    print(SEPARATOR)


def _print_tips():
    print(f"""
{SEPARATOR}
  TIPS — how to help the agent learn faster

  1. Record your own gameplay demo (best method):
       python scripts/play_human.py
     Then train the agent to copy your moves:
       python training/train_behavior_clone.py

  2. Import an existing video recording:
       python scripts/import_video_demo.py --video /path/to/recording.mp4
     This lets you label the frames and add them as a demo.

  3. Rate clips to shape the preference model:
       python scripts/rate_clips.py
     Then retrain the preference model:
       python training/train_preference_model.py

  4. Reward keyword cheat-sheet:
       MORE: forward, explore, shoot, aim, health, ammo, doors, dodge
       LESS: spinning, camping, stuck, retreating, spamming
       Use "stop X" or "less X" to reduce a positive behaviour.
{SEPARATOR}""")


# ── entry point ────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if "--all" in args:
        clips = sorted(glob.glob(os.path.join(CLIPS_DIR, "clip_*.pkl")))
        if not clips:
            print("No clips found in clips/.")
            return
        for cp in clips:
            coach_clip(cp)
            ans = input("\nContinue to next clip? (Y/n): ").strip().lower()
            if ans == "n":
                break
        return

    if "--clip" in args:
        idx = args.index("--clip")
        n = args[idx + 1] if idx + 1 < len(args) else "0"
        clip_path = os.path.join(CLIPS_DIR, f"clip_{n}.pkl")
    else:
        # Default: most recently modified clip
        clips = glob.glob(os.path.join(CLIPS_DIR, "clip_*.pkl"))
        if not clips:
            print("No clips found in clips/.")
            return
        clip_path = max(clips, key=os.path.getmtime)

    if not os.path.exists(clip_path):
        print(f"Clip not found: {clip_path}")
        return

    print("\n" + "=" * 56)
    print("  DOOM RL AGENT — AI COACH")
    print("=" * 56)
    play_clip(clip_path)
    coach_clip(clip_path)


if __name__ == "__main__":
    main()
