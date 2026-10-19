"""Hot-reloadable external prompt configuration loader for config/prompts.json."""
from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class PromptConfigLoader:
    """
    Loads `config/prompts.json` and automatically hot-reloads whenever the file's
    modification timestamp (`st_mtime`) increases on disk.
    If a user edit introduces a JSON syntax error, logs a warning and falls back
    gracefully to the last valid in-memory configuration without crashing (SRS 7.1).
    """

    def __init__(self, json_path: Optional[Path | str] = None) -> None:
        if json_path is None:
            env_path = os.environ.get("PROMPTS_JSON_PATH", "config/prompts.json")
            json_path = Path(env_path)
        self.json_path = Path(json_path).resolve()
        self._last_mtime: float = -1.0
        self._cached_config: Dict[str, Any] = {}
        self.reload_count: int = 0
        self.get_config()

    def get_config(self) -> Dict[str, Any]:
        """Return the latest valid prompt configuration, hot-reloading if modified."""
        try:
            stat = os.stat(self.json_path)
            mtime = stat.st_mtime
        except OSError as exc:
            logger.warning("Cannot stat prompts file %s: %s", self.json_path, exc)
            return copy.deepcopy(self._cached_config)

        if mtime != self._last_mtime or not self._cached_config:
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if not isinstance(loaded, dict) or "vertex_ai_nano_banana" not in loaded:
                    raise ValueError("Missing required top-level key 'vertex_ai_nano_banana'")
                self._cached_config = loaded
                self._last_mtime = mtime
                self.reload_count += 1
            except (json.JSONDecodeError, ValueError, OSError) as exc:
                logger.warning(
                    "Malformed or unreadable config/prompts.json (%s); falling back to last valid config.",
                    exc,
                )
        return copy.deepcopy(self._cached_config)

    def update_prompts_json(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and persist updated prompt configuration to disk."""
        if not isinstance(new_config, dict) or "vertex_ai_nano_banana" not in new_config:
            raise ValueError("Invalid prompt configuration schema: missing 'vertex_ai_nano_banana'")
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(new_config, f, indent=2, ensure_ascii=False)
        self._last_mtime = -1.0  # Force reload on next access
        return self.get_config()

    def build_nano_banana_prompt(self, recipe_context: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """
        Assemble the system instruction, formatted user prompt, and model config
        for Google Cloud Vertex AI Nano Banana dual-image refinement.
        """
        cfg = self.get_config()["vertex_ai_nano_banana"]
        prompts = cfg["dual_image_refinement_prompt"]
        finish_mods = prompts.get("finish_modifiers", {})
        special_mods = prompts.get("special_coating_modifiers", {})
        light_mods = prompts.get("lighting_environment_modifiers", {})

        finish_key = recipe_context.get("finish_key", "intrinsic_paint_or_surfacer")
        finish_text = finish_mods.get(finish_key, finish_mods.get("intrinsic_paint_or_surfacer", ""))
        exposed_details = recipe_context.get("exposed_layer_details", "Exposed base paint / surfacer")
        finish_text = finish_text.replace("{{exposed_layer_details}}", exposed_details)

        special_key = recipe_context.get("special_coating_key", "none")
        special_text = special_mods.get(special_key, special_mods.get("none", ""))

        light_key = recipe_context.get("lighting_env_key", "standard_daylight")
        light_text = light_mods.get(light_key, light_mods.get("standard_daylight", ""))

        neg_directives = prompts.get("negative_directives", "")
        if isinstance(neg_directives, list):
            neg_directives = " ".join(neg_directives)

        user_prompt = prompts["user_prompt_template"]
        replacements = {
            "{{workflow_summary}}": recipe_context.get("workflow_summary", "Multi-layer Gunze Mr. Color airbrush simulation"),
            "{{finish_modifier}}": finish_text,
            "{{special_coating_modifier}}": special_text,
            "{{lighting_environment_modifier}}": light_text,
            "{{negative_directives}}": neg_directives,
        }
        for placeholder, val in replacements.items():
            user_prompt = user_prompt.replace(placeholder, str(val))

        return prompts["system_instruction"], user_prompt, cfg
