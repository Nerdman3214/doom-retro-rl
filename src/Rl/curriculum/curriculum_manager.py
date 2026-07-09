class CurriculumManager:
    """
    Single source of truth for curriculum stage names.
    Stage indices match doom_env.py's self.curriculum_stage (0-7).
    doom_env.py controls advancement via self.curriculum_stage;
    this class just maps that integer to a name for reward routing.
    """

    def __init__(self):
        self.stages = [
            {"name": "movement"},        # 0
            {"name": "game_basics"},     # 1
            {"name": "avoid_stuck"},     # 2
            {"name": "combat_basic"},    # 3
            {"name": "combat_full"},     # 4
            {"name": "key_doors"},       # 5
            {"name": "full_game"},       # 6
            {"name": "complete_level"},  # 7
        ]

    def get_stage_name(self, curriculum_stage: int) -> str:
        """Return the name for the given stage index."""
        if 0 <= curriculum_stage < len(self.stages):
            return self.stages[curriculum_stage]["name"]
        return self.stages[-1]["name"]