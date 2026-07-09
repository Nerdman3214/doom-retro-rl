from pathlib import Path

path = Path("env/doom_env.py")
text = path.read_text()

# ------------------------------------------------------------
# Add HudPredictor import block after ObjectPredictor import block
# ------------------------------------------------------------
if "from vision.hud_predictor import HudPredictor" not in text:
    marker = """try:
    from vision.object_predictor import ObjectPredictor
except Exception:
    ObjectPredictor = None
"""
    replacement = marker + """
try:
    from vision.hud_predictor import HudPredictor
except Exception as e:
    print(f"[hud_vision] HudPredictor disabled: {e}")
    HudPredictor = None
"""
    text = text.replace(marker, replacement)

# ------------------------------------------------------------
# Initialize hud_predictor after object_predictor setup
# ------------------------------------------------------------
if "self.hud_predictor = HudPredictor()" not in text:
    marker = """        if ObjectPredictor is not None:
            try:
                self.object_predictor = ObjectPredictor()
                print("[object_vision] Loaded object_multilabel_classifier.pt")
            except Exception as e:
                print(f"[object_vision] Could not load object predictor: {e}")
                self.object_predictor = None
"""
    replacement = marker + """
        if HudPredictor is not None:
            try:
                self.hud_predictor = HudPredictor()
                print("[hud_vision] Loaded lightweight HUD predictor")
            except Exception as e:
                print(f"[hud_vision] Could not initialize HUD predictor: {e}")
                self.hud_predictor = None
        else:
            self.hud_predictor = None
"""
    text = text.replace(marker, replacement)

# ------------------------------------------------------------
# Add pre-action HUD vision after pre_game_state is created
# ------------------------------------------------------------
if "[pre_hud_vision]" not in text:
    marker = """        pre_frame = self.observer.get_frame()
        pre_game_state = self.observer.get_game_state()
        pre_vision = self.detect_vision(pre_frame)
"""
    replacement = marker + """
        # -----------------------------------------------------
        # Pre-action HUD vision
        # -----------------------------------------------------
        # This gives the agent human-visible status information such as
        # health/ammo/armor buckets without mixing HUD pixels into scene/object vision.
        pre_hud_result = None

        if getattr(self, "hud_predictor", None) is not None and pre_frame is not None:
            try:
                pre_hud_result = self.hud_predictor.predict(
                    pre_frame,
                    game_state=pre_game_state,
                )
                pre_game_state["hud_vision"] = pre_hud_result

                if self._step_count % 25 == 0:
                    print(
                        "[pre_hud_vision] "
                        f"health={pre_hud_result.get('health')} "
                        f"health_bucket={pre_hud_result.get('health_bucket')} "
                        f"ammo={pre_hud_result.get('ammo')} "
                        f"ammo_bucket={pre_hud_result.get('ammo_bucket')} "
                        f"armor={pre_hud_result.get('armor')} "
                        f"damage_flash={pre_hud_result.get('damage_flash')}"
                    )

            except Exception as e:
                print(f"[pre_hud_vision] prediction failed: {e}")
                pre_hud_result = None
"""
    text = text.replace(marker, replacement)

# ------------------------------------------------------------
# Add post-action HUD vision after game_state = self.observer.get_game_state()
# ------------------------------------------------------------
if "[hud_vision]" not in text:
    marker = """        game_state = self.observer.get_game_state()

        reward += self.route_progress_reward(game_state)
"""
    replacement = """        game_state = self.observer.get_game_state()

        # -----------------------------------------------------
        # Post-action HUD vision
        # -----------------------------------------------------
        hud_result = None

        if getattr(self, "hud_predictor", None) is not None and raw_frame is not None:
            try:
                hud_result = self.hud_predictor.predict(
                    raw_frame,
                    game_state=game_state,
                )
                game_state["hud_vision"] = hud_result

                if self._step_count % 25 == 0:
                    print(
                        "[hud_vision] "
                        f"health={hud_result.get('health')} "
                        f"health_bucket={hud_result.get('health_bucket')} "
                        f"ammo={hud_result.get('ammo')} "
                        f"ammo_bucket={hud_result.get('ammo_bucket')} "
                        f"armor={hud_result.get('armor')} "
                        f"damage_flash={hud_result.get('damage_flash')}"
                    )

            except Exception as e:
                print(f"[hud_vision] prediction failed: {e}")
                hud_result = None

        reward += self.route_progress_reward(game_state)
"""
    text = text.replace(marker, replacement)

path.write_text(text)
print("Patched env/doom_env.py with HudPredictor")
