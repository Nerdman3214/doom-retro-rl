class CurriculumManager:

    def __init__(self):

        self.current_stage = 0

        self.stages = [

            {
                "name": "movement",
                "required_score": 50,
            },

            {
                "name": "track_enemy",
                "required_score": 100,
            },

            {
                "name": "dodge_enemies",
                "required_score": 150,
            },

            {
                "name": "combat",
                "required_score": 300,
            },
        ]

    def get_stage_name(self):

        return self.stages[self.current_stage]["name"]

    def advance(self):

        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1