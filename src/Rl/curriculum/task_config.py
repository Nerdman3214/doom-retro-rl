from dataclasses import dataclass, field
from typing import Dict, List, Optional


ALL_ACTIONS = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "shoot",
    "use",
]


@dataclass
class TaskConfig:
    """
    Small-task curriculum config.

    This lets us train one skill at a time:
    - navigation
    - corridor movement
    - combat
    - navigation + combat
    - items
    - use/doors
    - exit completion
    - secrets later

    The env should use this as the current source of truth for:
    - allowed actions
    - enabled reward groups
    - task success reward
    - timeout penalty
    """

    name: str
    description: str

    allowed_actions: List[str] = field(default_factory=lambda: list(ALL_ACTIONS))

    # Reward group switches.
    enable_navigation: bool = True
    enable_combat: bool = False
    enable_items: bool = False
    enable_use: bool = False
    enable_exit: bool = False
    enable_secrets: bool = False
    enable_reroute: bool = True

    # Disable old mixed shaping unless this task needs it.
    enable_route_bubbles: bool = False
    enable_goal_bubbles: bool = False
    enable_spawn_escape: bool = False

    # Main task targets.
    target_name: str = "level_exit"
    target_x: float = -400.0
    target_y: float = 1296.0
    target_radius: float = 160.0

    # Optional stage requirements.
    required_kills: int = 0
    required_items: int = 0
    required_uses: int = 0

    # Positive rewards.
    reward_task_success: float = 300.0
    reward_new_best_distance: float = 12.0
    reward_distance_progress_scale: float = 1.0 / 16.0
    reward_distance_progress_cap: float = 4.0
    reward_kill: float = 30.0
    reward_item: float = 10.0
    reward_good_shot: float = 20.0
    reward_good_use: float = 20.0
    reward_reroute_success: float = 15.0

    # Negative rewards.
    penalty_timeout: float = -300.0
    penalty_no_progress: float = -8.0
    penalty_wall_push: float = -10.0
    penalty_bad_shot: float = -6.0
    penalty_wasted_shot: float = -10.0
    penalty_use_spam: float = -3.0

    # Success thresholds.
    success_distance: float = 160.0


TASKS: Dict[str, TaskConfig] = {
    "stage_01_navigation": TaskConfig(
        name="stage_01_navigation",
        description="Reach the early corridor / move away from spawn reliably.",
        allowed_actions=[
            "move_forward",
            "move_backward",
            "turn_left",
            "turn_right",
            "strafe_left",
            "strafe_right",
        ],
        enable_navigation=True,
        enable_combat=False,
        enable_items=False,
        enable_use=False,
        enable_exit=False,
        target_name="corridor_entry",
        target_x=-410.0,
        target_y=360.0,
        target_radius=120.0,
        reward_task_success=200.0,
        reward_new_best_distance=10.0,
        reward_distance_progress_cap=3.0,
        penalty_timeout=-150.0,
        success_distance=120.0,
    ),

    "stage_02_corridor_navigation": TaskConfig(
        name="stage_02_corridor_navigation",
        description="Move through the corridor and learn wall/rail rerouting.",
        allowed_actions=[
            "move_forward",
            "move_backward",
            "turn_left",
            "turn_right",
            "strafe_left",
            "strafe_right",
        ],
        enable_navigation=True,
        enable_combat=False,
        enable_items=False,
        enable_use=False,
        enable_exit=False,
        enable_reroute=True,
        target_name="post_corridor",
        target_x=-350.0,
        target_y=650.0,
        target_radius=150.0,
        reward_task_success=300.0,
        reward_new_best_distance=15.0,
        reward_distance_progress_cap=5.0,
        penalty_wall_push=-12.0,
        penalty_timeout=-250.0,
        success_distance=150.0,
    ),

    "stage_03_basic_combat": TaskConfig(
        name="stage_03_basic_combat",
        description="Learn to shoot and kill 2 enemies.",
        allowed_actions=[
            "move_forward",
            "move_backward",
            "turn_left",
            "turn_right",
            "strafe_left",
            "strafe_right",
            "shoot",
        ],
        enable_navigation=False,
        enable_combat=True,
        enable_items=False,
        enable_use=False,
        enable_exit=False,
        required_kills=2,
        reward_task_success=350.0,
        reward_kill=60.0,
        reward_good_shot=25.0,
        penalty_bad_shot=-6.0,
        penalty_wasted_shot=-10.0,
        penalty_timeout=-150.0,
    ),

    "stage_04_nav_combat": TaskConfig(
        name="stage_04_nav_combat",
        description="Reach corridor/post-corridor area while killing 2 enemies.",
        allowed_actions=[
            "move_forward",
            "move_backward",
            "turn_left",
            "turn_right",
            "strafe_left",
            "strafe_right",
            "shoot",
        ],
        enable_navigation=True,
        enable_combat=True,
        enable_items=False,
        enable_use=False,
        enable_exit=False,
        enable_reroute=True,
        target_name="post_corridor_combat",
        target_x=-350.0,
        target_y=650.0,
        target_radius=160.0,
        required_kills=2,
        reward_task_success=500.0,
        reward_kill=40.0,
        reward_good_shot=20.0,
        reward_new_best_distance=15.0,
        reward_distance_progress_cap=5.0,
        penalty_timeout=-300.0,
    ),

    "stage_05_items": TaskConfig(
        name="stage_05_items",
        description="Pick up useful items such as ammo/health/armor.",
        allowed_actions=[
            "move_forward",
            "move_backward",
            "turn_left",
            "turn_right",
            "strafe_left",
            "strafe_right",
        ],
        enable_navigation=True,
        enable_combat=False,
        enable_items=True,
        enable_use=False,
        enable_exit=False,
        required_items=2,
        reward_task_success=350.0,
        reward_item=30.0,
        reward_new_best_distance=10.0,
        penalty_timeout=-200.0,
    ),

    "stage_06_use_doors": TaskConfig(
        name="stage_06_use_doors",
        description="Learn use action near doors/switches.",
        allowed_actions=[
            "move_forward",
            "move_backward",
            "turn_left",
            "turn_right",
            "strafe_left",
            "strafe_right",
            "use",
        ],
        enable_navigation=True,
        enable_combat=False,
        enable_items=False,
        enable_use=True,
        enable_exit=False,
        required_uses=1,
        reward_task_success=350.0,
        reward_good_use=30.0,
        penalty_use_spam=-4.0,
        penalty_timeout=-200.0,
    ),

    "stage_07_exit_completion": TaskConfig(
        name="stage_07_exit_completion",
        description="Complete the level. Combat/items/use may assist but exit is the main objective.",
        allowed_actions=list(ALL_ACTIONS),
        enable_navigation=True,
        enable_combat=True,
        enable_items=True,
        enable_use=True,
        enable_exit=True,
        enable_secrets=False,
        enable_reroute=True,
        target_name="level_exit",
        target_x=-400.0,
        target_y=1296.0,
        target_radius=160.0,
        reward_task_success=1000.0,
        reward_new_best_distance=20.0,
        reward_distance_progress_cap=8.0,
        reward_kill=30.0,
        reward_item=10.0,
        reward_good_shot=20.0,
        reward_good_use=25.0,
        penalty_timeout=-500.0,
    ),

    "stage_08_secrets": TaskConfig(
        name="stage_08_secrets",
        description="Secret completion after level completion behavior is stable.",
        allowed_actions=list(ALL_ACTIONS),
        enable_navigation=True,
        enable_combat=True,
        enable_items=True,
        enable_use=True,
        enable_exit=True,
        enable_secrets=True,
        enable_reroute=True,
        reward_task_success=1200.0,
        reward_new_best_distance=15.0,
        reward_kill=30.0,
        reward_item=15.0,
        reward_good_shot=20.0,
        reward_good_use=25.0,
        penalty_timeout=-500.0,
    ),
}


def get_task_config(name: Optional[str]) -> TaskConfig:
    if not name:
        name = "stage_01_navigation"

    if name not in TASKS:
        valid = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown curriculum task: {name}. Valid tasks: {valid}")

    return TASKS[name]


def list_task_names() -> List[str]:
    return list(TASKS.keys())
