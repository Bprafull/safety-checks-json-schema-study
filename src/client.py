"""OpenRouter API Client with Rate Limiting, Spend/Call Guards, and Retries.

Low-RAM rule: Uses only Python standard library and requests.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DELAY_SECONDS,
    MAX_CALLS,
    MAX_SPEND_USD,
    MODE,
    MODELS,
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
)


class OpenRouterClient:
    """Robust OpenRouter API client enforcing rate limits, spend guards, and retries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        mode: str = MODE,
        models: Optional[List[str]] = None,
        delay_seconds: float = DELAY_SECONDS,
        max_calls: Optional[int] = MAX_CALLS,
        max_spend_usd: Optional[float] = MAX_SPEND_USD,
    ):
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in .env or environment. "
                "Please configure OPENROUTER_API_KEY in your local .env file."
            )

        self.mode = mode
        self.configured_models = list(models or MODELS)
        self.delay_seconds = delay_seconds
        self.max_calls = max_calls
        self.max_spend_usd = max_spend_usd
        self.default_max_tokens = 2000 if self.mode == "free" else 1200

        # State tracking
        self.total_calls = 0
        self.total_cost_usd = 0.0
        self.last_call_time = 0.0
        self.model_map: Dict[str, str] = {}

        # Default headers required/recommended by OpenRouter
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/safety-checks-json-schema-study",
            "X-Title": "Safety Checks for JSON Schema Study",
        }

        # Verify and resolve models
        self._verify_and_resolve_models()

    def _fetch_available_models(self) -> List[str]:
        """Fetch list of active model IDs from OpenRouter."""
        try:
            resp = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=self.headers,
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                return [m.get("id") for m in data if m.get("id")]
        except Exception as e:
            print(f"[OpenRouterClient] Warning: Could not fetch models list: {e}")
        return []

    def _verify_and_resolve_models(self) -> None:
        """Verify configured models exist; fall back cleanly if any slug returns 404/missing."""
        available_ids = set(self._fetch_available_models())
        if not available_ids:
            # If endpoint query failed, preserve configured models as-is
            for m in self.configured_models:
                self.model_map[m] = m
            return

        free_candidates = [m for m in available_ids if m.endswith(":free")]

        for model in self.configured_models:
            if model in available_ids:
                self.model_map[model] = model
            else:
                # In free mode, find a fallback free model
                if self.mode == "free":
                    # Attempt fallback from candidate list
                    fallback = None
                    for cand in free_candidates:
                        if cand not in self.model_map.values():
                            fallback = cand
                            break
                    if fallback:
                        print(
                            f"[OpenRouterClient] Configured model '{model}' not found online. "
                            f"Auto-falling back to active free model: '{fallback}'"
                        )
                        self.model_map[model] = fallback
                    else:
                        print(f"[OpenRouterClient] Warning: Model '{model}' not in available list.")
                        self.model_map[model] = model
                else:
                    print(f"[OpenRouterClient] Warning: Paid model '{model}' not found in models list.")
                    self.model_map[model] = model

    def get_resolved_models(self) -> List[str]:
        """Return list of active resolved model IDs."""
        return [self.model_map.get(m, m) for m in self.configured_models]

    def _enforce_guards(self) -> None:
        """Enforce call limits, spend caps, and minimum inter-call delay."""
        # 1. max_calls safety cap
        if self.max_calls is not None and self.total_calls >= self.max_calls:
            raise RuntimeError(
                f"[Safety Guard] Hard limit of {self.max_calls} total calls reached for this run. "
                "Aborting to protect call quota."
            )

        # 2. Paid mode max_spend_usd guard
        if self.mode == "paid" and self.max_spend_usd is not None:
            if self.total_cost_usd >= self.max_spend_usd:
                raise RuntimeError(
                    f"[Budget Guard] Hard limit of ${self.max_spend_usd:.2f} cumulative spend reached "
                    f"(spent: ${self.total_cost_usd:.4f}). Aborting."
                )

        # 3. Rate limit inter-call delay
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.delay_seconds:
            sleep_duration = self.delay_seconds - elapsed
            time.sleep(sleep_duration)

    def generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Send chat completion request with retries and return standardized response dict.

        Returns:
            {
                "ok": bool,
                "model": str,
                "content": str,
                "raw_usage": dict,
                "cost_usd": float,
                "latency_ms": float,
                "error": str | None
            }
        """
        # Resolve model name if mapped
        target_model = self.model_map.get(model, model)

        # Enforce rate and quota guards
        self._enforce_guards()

        effective_max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        result: Dict[str, Any] = {
            "ok": False,
            "model": target_model,
            "content": "",
            "reasoning": "",
            "raw_usage": {},
            "cost_usd": 0.0,
            "latency_ms": 0.0,
            "error": None,
        }

        attempt = 0
        backoff_sec = 2.0

        while attempt <= max_retries:
            attempt += 1
            start_time = time.time()
            try:
                resp = requests.post(
                    OPENROUTER_API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=45,
                )
                latency_ms = (time.time() - start_time) * 1000
                result["latency_ms"] = round(latency_ms, 2)
                self.last_call_time = time.time()

                # Handle Rate Limit (429) or Server Errors (5xx) with backoff
                if resp.status_code == 429 or resp.status_code >= 500:
                    error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    if attempt <= max_retries:
                        retry_after = resp.headers.get("Retry-After")
                        wait_time = float(retry_after) if retry_after else (backoff_sec * (2 ** (attempt - 1)))
                        print(f"[{target_model}] {error_msg}. Retrying in {wait_time:.1f}s (attempt {attempt}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        if self.mode == "free" and target_model != "particle-ai/particle-1:free":
                            fallback = "particle-ai/particle-1:free"
                            print(f"[{target_model}] Exhausted 429 retries upstream. Auto-falling back to: '{fallback}'...")
                            self.model_map[target_model] = fallback
                            target_model = fallback
                            payload["model"] = fallback
                            attempt = 0
                            time.sleep(2.0)
                            continue
                        result["error"] = f"Exhausted retries: {error_msg}"
                        return result

                # Parse JSON body
                try:
                    res_json = resp.json()
                except Exception as json_err:
                    result["error"] = f"JSON decode error: {json_err} (Status: {resp.status_code}, Body: {resp.text[:200]})"
                    return result

                # Check for API-level errors
                if resp.status_code != 200 or "error" in res_json:
                    err_info = res_json.get("error", resp.text[:200])
                    result["error"] = f"API Error (HTTP {resp.status_code}): {err_info}"
                    return result

                # Extract content
                choices = res_json.get("choices", [])
                if not choices:
                    result["error"] = "Empty choices returned by API"
                    return result

                first_choice = choices[0]
                message = first_choice.get("message", {})
                content = message.get("content") or ""
                reasoning = message.get("reasoning") or ""

                # Extract usage and cost
                usage = res_json.get("usage", {})
                cost = float(usage.get("total_cost") or usage.get("cost") or 0.0)

                # Update client state tracking
                self.total_calls += 1
                self.total_cost_usd += cost

                result["ok"] = True
                result["content"] = content
                result["reasoning"] = reasoning
                result["raw_usage"] = usage
                result["cost_usd"] = cost
                return result

            except requests.exceptions.RequestException as req_err:
                latency_ms = (time.time() - start_time) * 1000
                result["latency_ms"] = round(latency_ms, 2)
                self.last_call_time = time.time()
                error_msg = f"Network exception: {req_err}"
                if attempt <= max_retries:
                    wait_time = backoff_sec * (2 ** (attempt - 1))
                    print(f"[{target_model}] {error_msg}. Retrying in {wait_time:.1f}s (attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    result["error"] = f"Exhausted retries: {error_msg}"
                    return result

        return result


def run_smoke_test() -> None:
    """Execute smoke test: 1 call per model (3 calls total) to verify API key and model connectivity."""
    print("=" * 65)
    print("OPENROUTER CLIENT SMOKE TEST (3 Free Calls)")
    print("=" * 65)

    client = OpenRouterClient()
    resolved_models = client.get_resolved_models()

    print(f"Active Mode:     {client.mode}")
    print(f"Configured:      {client.configured_models}")
    print(f"Resolved Models: {resolved_models}")
    print(f"Delay Between:   {client.delay_seconds}s")
    print(f"Max Calls Cap:   {client.max_calls}")
    print("-" * 65)

    all_passed = True
    smoke_messages = [{"role": "user", "content": "Reply with the single word READY."}]

    for idx, model_id in enumerate(resolved_models, start=1):
        print(f"\n[Call {idx}/3] Testing model: {model_id} ...")
        res = client.generate(model=model_id, messages=smoke_messages, max_tokens=16)

        if res["ok"]:
            reply = res["content"].strip().replace("\n", " ")
            print(f"  Status:    SUCCESS")
            print(f"  Response:  '{reply}'")
            print(f"  Latency:   {res['latency_ms']} ms")
            print(f"  Usage:     {res['raw_usage']}")
            print(f"  Calls Run: {client.total_calls}/{client.max_calls}")
        else:
            print(f"  Status:    FAILED")
            print(f"  Error:     {res['error']}")
            print(f"  Latency:   {res['latency_ms']} ms")
            all_passed = False

    print("\n" + "=" * 65)
    print(f"SMOKE TEST SUMMARY: Total Calls Made = {client.total_calls}, Total Cost = ${client.total_cost_usd:.4f}")
    if all_passed:
        print("ALL 3 MODELS PASSED SUCCESSFULLY! Exit Code 0.")
        print("=" * 65 + "\n")
        sys.exit(0)
    else:
        print("ONE OR MORE SMOKE TEST CALLS FAILED! Exit Code 1.")
        print("=" * 65 + "\n")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenRouter API Client for Safety Study")
    parser.add_argument("--smoke", action="store_true", help="Run 3-call smoke test across configured models")
    args = parser.parse_args()

    if args.smoke:
        run_smoke_test()
    else:
        print("OpenRouterClient loaded successfully. Run with --smoke to execute the 3-model smoke test.")


if __name__ == "__main__":
    main()
