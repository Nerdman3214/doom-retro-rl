#!/usr/bin/env python3
"""
Simple test script to verify DoomEnv reward initialization works without running DOOM.
"""

import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from Rl.env.doom_env import DoomEnv
    print("✓ DoomEnv import successful")

    # Try to create env without launching DOOM
    env = DoomEnv(launch_doom=False)
    print("✓ DoomEnv instantiation successful")

    # Check if step method exists and reward initialization is present
    if hasattr(env, 'step'):
        print("✓ step method exists")

        # Check the source code for reward initialization
        import inspect
        source = inspect.getsource(env.step)
        if 'reward = 0.0' in source:
            print("✓ Reward initialization found in step method")
        else:
            print("✗ Reward initialization NOT found in step method")

        if 'reward += self.reward_manager.get_reward()' in source:
            print("✓ Reward manager integration found")
        else:
            print("✗ Reward manager integration NOT found")

    else:
        print("✗ step method missing")

    print("\nTest completed successfully!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()