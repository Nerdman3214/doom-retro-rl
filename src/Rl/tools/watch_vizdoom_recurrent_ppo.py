from pathlib import Path

from sb3_contrib import RecurrentPPO

from env.vizdoom_env import VizDoomEnv


ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "vizdoom_recurrent_ppo_agent.zip"


def unwrap_reset(reset_out):
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


def unwrap_step(step_out):
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = terminated or truncated
        return obs, reward, done, info

    obs, reward, done, info = step_out
    return obs, reward, done, info


def main():
    env = VizDoomEnv(visible=True)

    print(f"[watch_vizdoom] loading {CKPT}")
    model = RecurrentPPO.load(str(CKPT), env=env, device="auto")

    obs = unwrap_reset(env.reset())
    lstm_states = None
    episode_starts = [True]

    ep = 1
    total_reward = 0.0
    step = 0

    print("[watch_vizdoom] watching agent. Ctrl+C to stop.")

    while True:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )

        obs, reward, done, info = unwrap_step(env.step(action))

        total_reward += float(reward)
        step += 1
        episode_starts = [done]

        if step % 100 == 0:
            print(f"[watch_vizdoom] ep={ep} step={step} reward={total_reward:.2f}")

        if done:
            print(f"[watch_vizdoom] episode done ep={ep} steps={step} reward={total_reward:.2f}")
            obs = unwrap_reset(env.reset())
            lstm_states = None
            episode_starts = [True]
            ep += 1
            total_reward = 0.0
            step = 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[watch_vizdoom] stopped")
