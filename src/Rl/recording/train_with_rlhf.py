import subprocess


def run_step(command):

    print(f"\nRunning: {command}\n")

    subprocess.run(command, shell=True)


while True:

    print("\n--- RLHF LOOP START ---\n")

    run_step("python train_rl_agent.py")

    run_step("python build_clip_pairs.py")

    run_step("python compare_clips_auto.py")

    run_step("python train_preference_model.py")

    print("\nLoop complete. Restarting...\n")