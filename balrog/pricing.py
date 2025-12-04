"""
Pricing information for different LLM models.

Prices are fetched from the LiteLLM pricing database:
https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

The pricing data is cached locally for performance.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional
import urllib.request

logger = logging.getLogger(__name__)

# URL to the LiteLLM pricing JSON
LITELLM_PRICING_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

# Cache settings
CACHE_DIR = Path.home() / ".cache" / "balrog"
CACHE_FILE = CACHE_DIR / "model_pricing.json"
CACHE_DURATION = 24 * 60 * 60  # 24 hours in seconds

# In-memory cache
_pricing_cache: Optional[Dict] = None


def fetch_pricing_data(force_refresh: bool = False) -> Dict:
    """Fetch pricing data from LiteLLM, with local caching.

    Args:
        force_refresh (bool): If True, ignore cache and fetch fresh data.

    Returns:
        Dict: The pricing data from LiteLLM.
    """
    global _pricing_cache

    # Check in-memory cache first
    if _pricing_cache is not None and not force_refresh:
        return _pricing_cache

    # Check if we have a valid cached file
    if not force_refresh and CACHE_FILE.exists():
        cache_age = time.time() - CACHE_FILE.stat().st_mtime
        if cache_age < CACHE_DURATION:
            try:
                with open(CACHE_FILE, "r") as f:
                    data = json.load(f)
                    _pricing_cache = data
                    logger.debug(f"Loaded pricing data from cache (age: {cache_age/3600:.1f} hours)")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load cached pricing data: {e}")

    # Fetch fresh data
    try:
        logger.info(f"Fetching pricing data from {LITELLM_PRICING_URL}")
        with urllib.request.urlopen(LITELLM_PRICING_URL, timeout=10) as response:
            data = json.loads(response.read().decode())

            # Cache the data
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w") as f:
                json.dump(data, f)

            _pricing_cache = data
            logger.info("Successfully fetched and cached pricing data")
            return data
    except Exception as e:
        logger.error(f"Failed to fetch pricing data: {e}")

        # Try to use stale cache if available
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r") as f:
                    data = json.load(f)
                    _pricing_cache = data
                    logger.warning("Using stale cached pricing data")
                    return data
            except Exception as cache_error:
                logger.error(f"Failed to load stale cache: {cache_error}")

        # Return empty dict as fallback
        logger.warning("No pricing data available, using empty fallback")
        return {}


def get_model_pricing(model_id: str) -> Dict[str, float]:
    """Get pricing for a specific model.

    Args:
        model_id (str): The model identifier.

    Returns:
        Dict[str, float]: Dictionary with 'input' and 'output' prices per token.
    """
    pricing_data = fetch_pricing_data()

    # Default fallback
    default_pricing = {"input": 0.0, "output": 0.0}

    if not pricing_data:
        return default_pricing

    # Try exact match first
    if model_id in pricing_data:
        model_info = pricing_data[model_id]
        return {
            "input": model_info.get("input_cost_per_token", 0.0),
            "output": model_info.get("output_cost_per_token", 0.0),
        }

    # Try case-insensitive partial match
    model_id_lower = model_id.lower()

    # First pass: look for exact substring matches
    for key, value in pricing_data.items():
        if key.lower() == model_id_lower:
            return {
                "input": value.get("input_cost_per_token", 0.0),
                "output": value.get("output_cost_per_token", 0.0),
            }

    # Second pass: look for partial matches
    # Prefer longer matches (more specific)
    best_match = None
    best_match_length = 0

    for key, value in pricing_data.items():
        key_lower = key.lower()

        # Skip non-chat models and special entries
        if value.get("mode") not in ["chat", None]:
            continue

        # Check if there's a match
        if key_lower in model_id_lower or model_id_lower in key_lower:
            match_length = min(len(key_lower), len(model_id_lower))
            if match_length > best_match_length:
                best_match = value
                best_match_length = match_length

    if best_match:
        return {
            "input": best_match.get("input_cost_per_token", 0.0),
            "output": best_match.get("output_cost_per_token", 0.0),
        }

    # No match found
    logger.warning(f"No pricing found for model '{model_id}', using default (0.0)")
    return default_pricing


def calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost for a given number of input and output tokens.

    Args:
        model_id (str): The model identifier.
        input_tokens (int): Number of input tokens.
        output_tokens (int): Number of output tokens.

    Returns:
        float: Total cost in USD.
    """
    pricing = get_model_pricing(model_id)

    # Prices are per token, so just multiply
    input_cost = input_tokens * pricing["input"]
    output_cost = output_tokens * pricing["output"]

    return input_cost + output_cost


def format_cost(cost: float) -> str:
    """Format cost for display.

    Args:
        cost (float): Cost in USD.

    Returns:
        str: Formatted cost string.
    """
    if cost < 0.01:
        return f"${cost:.6f}"
    elif cost < 1.00:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def refresh_pricing_cache() -> bool:
    """Manually refresh the pricing cache.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        fetch_pricing_data(force_refresh=True)
        return True
    except Exception as e:
        logger.error(f"Failed to refresh pricing cache: {e}")
        return False


def get_cache_info() -> Dict:
    """Get information about the pricing cache.

    Returns:
        Dict: Information about cache status, age, etc.
    """
    info = {
        "cache_file": str(CACHE_FILE),
        "cache_exists": CACHE_FILE.exists(),
        "in_memory_loaded": _pricing_cache is not None,
    }

    if CACHE_FILE.exists():
        cache_age = time.time() - CACHE_FILE.stat().st_mtime
        info["cache_age_hours"] = cache_age / 3600
        info["cache_valid"] = cache_age < CACHE_DURATION

    return info
