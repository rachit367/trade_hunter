"""
Trade Store — JSON-file-based persistence for risk management state.

Survives process restarts. Stores:
  - Daily P&L
  - Consecutive loss count
  - Open position details
  - Cooldown timestamp
  - Balance snapshot
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("trade_store")

DEFAULT_PATH = "trade_state.json"


class TradeStore:
    """
    JSON-file-based persistence for trade state.

    Loaded on startup, saved after each trade close.
    Survives process restarts on Render/Docker/etc.
    """

    def __init__(self, filepath: str = DEFAULT_PATH):
        self.filepath = filepath

    def save(self, state: dict):
        """Save risk manager state to JSON file."""
        try:
            payload = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "state": state,
            }
            # Write atomically: write to temp file, then rename
            tmp_path = self.filepath + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp_path, self.filepath)
            logger.debug("Trade state saved to %s", self.filepath)
        except Exception as e:
            logger.error("Failed to save trade state: %s", e)

    def load(self) -> Optional[dict]:
        """Load risk manager state from JSON file."""
        if not os.path.exists(self.filepath):
            logger.info("No trade state file found at %s", self.filepath)
            return None

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)
            state = payload.get("state", {})
            last_updated = payload.get("last_updated", "unknown")
            logger.info("Trade state loaded from %s (last updated: %s)",
                        self.filepath, last_updated)
            return state
        except Exception as e:
            logger.error("Failed to load trade state: %s", e)
            return None

    def clear(self):
        """Delete the state file."""
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
            logger.info("Trade state cleared: %s", self.filepath)
