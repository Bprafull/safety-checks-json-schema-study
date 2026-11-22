"""Experiment runner with checkpointed JSONL streaming, auto-resume, and JSON parsing.

Low-RAM rule: Uses only Python standard library and requests.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.client import OpenRouterClient
from src.config import (
    ACTIVE_WRAPPERS,
    MODE,
    RESULTS_DIR,
    SAMPLE_DATA_PATH,
    WRAPPERS,
)


def load_completed_keys(output_path: Path) -> Set[Tuple[str, str, str]]:
    """Scan existing output .jsonl file and return set of completed (prompt_id, model, wrapper) keys."""
    completed: Set[Tuple[str, str, str]] = set()
    if not output_path.exists():
        return completed

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("ok"):
                        key = (
                            str(record.get("prompt_id")),
                            str(record.get("model")),
                            str(record.get("wrapper")),
                        )
                        completed.add(key)
                except Exception:
                    continue
    except Exception as e:
        print(f"[Runner] Warning: Could not read existing output file: {e}")

    return completed


def parse_model_response(
    wrapper_name: str,
    content: str,
    reasoning: str = "",
) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str], str]:
    """Safely extract and parse JSON from model output or reasoning.

    Returns:
        (parsed_json, json_valid, declared_status, response_text)
    """
    clean_content = content.strip()
    clean_reasoning = reasoning.strip()

    # Free text does not require JSON validation
    if wrapper_name in ("free_text", "free_text_blocked"):
        clean_text = clean_content if clean_content else clean_reasoning
        return None, False, None, clean_text

    candidates = []
    if clean_content:
        candidates.append(clean_content)
    if clean_reasoning:
        candidates.append(clean_reasoning)

    for text in candidates:
        # 1. Direct JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                status = str(data.get("status", "")).strip().lower() if data.get("status") is not None else None
                resp_text = str(data.get("response", "")) if "response" in data else str(data)
                return data, True, status, resp_text
        except Exception:
            pass

        # 2. Extract markdown code block ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            try:
                data = json.loads(fence_match.group(1))
                if isinstance(data, dict):
                    status = str(data.get("status", "")).strip().lower() if data.get("status") is not None else None
                    resp_text = str(data.get("response", "")) if "response" in data else str(data)
                    return data, True, status, resp_text
            except Exception:
                pass

        # 3. Extract outermost { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, dict):
                    status = str(data.get("status", "")).strip().lower() if data.get("status") is not None else None
                    resp_text = str(data.get("response", "")) if "response" in data else str(data)
                    return data, True, status, resp_text
            except Exception:
                pass

    # JSON extraction failed
    fallback_text = clean_content if clean_content else clean_reasoning
    return None, False, None, fallback_text


def run_experiment(
    mode: str = MODE,
    limit: Optional[int] = None,
    models_limit: Optional[int] = None,
    output_path: Optional[Path] = None,
    dry_run: bool = False,
    wrappers: Optional[List[str]] = None,
    max_calls: Optional[int] = None,
) -> Path:
    """Run experiment across prompts, models, and wrappers with checkpointing."""
    if dry_run:
        limit = limit if limit is not None else 2
        models_limit = models_limit if models_limit is not None else 1
    else:
        limit = limit if limit is not None else (2 if mode == "free" else 200)

    # Determine output file path
    if output_path is None:
        output_path = RESULTS_DIR / f"results_{mode}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if not SAMPLE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Sample dataset not found at {SAMPLE_DATA_PATH}. "
            "Please run src/prepare_data.py first."
        )

    with open(SAMPLE_DATA_PATH, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)

    prompts_to_run = all_prompts[:limit]

    # Initialize client with optional max_calls
    client = OpenRouterClient(mode=mode, max_calls=max_calls)
    resolved_models = client.configured_models
    models_to_run = resolved_models[:models_limit] if models_limit else resolved_models
    
    if wrappers is None:
        wrapper_keys = ACTIVE_WRAPPERS
    else:
        wrapper_keys = wrappers

    total_planned = len(prompts_to_run) * len(models_to_run) * len(wrapper_keys)

    # Load completed keys for automatic resume
    completed_keys = load_completed_keys(output_path)
    print("=" * 70)
    print(f"EXPERIMENT RUNNER {'(DRY RUN)' if dry_run else ''}")
    print("=" * 70)
    print(f"Mode:             {mode}")
    print(f"Prompts to run:   {len(prompts_to_run)} (from total {len(all_prompts)})")
    print(f"Models to test:   {models_to_run}")
    print(f"Wrappers to test: {wrapper_keys}")
    print(f"Total planned:    {total_planned} calls")
    print(f"Already cached:   {len(completed_keys)} calls (will skip)")
    print(f"Output file:      {output_path}")
    print("-" * 70)

    session_calls = 0
    resumed_count = 0
    json_valid_count = 0
    json_total_count = 0

    call_index = 0

    for p_idx, prompt_item in enumerate(prompts_to_run, start=1):
        p_id = str(prompt_item["id"])
        p_label = prompt_item["label"]
        p_cat = prompt_item["category"]
        p_focus = prompt_item.get("focus", "")
        p_text = prompt_item["prompt"]

        stop_experiment = False
        for model in models_to_run:
            if stop_experiment:
                break
            for wrapper_name in wrapper_keys:
                if max_calls is not None and session_calls >= max_calls:
                    print(f"\n[Runner] Reached max-calls limit ({max_calls}). Stopping run.")
                    stop_experiment = True
                    break

                call_index += 1
                lookup_key = (p_id, model, wrapper_name)

                # Resume check
                if lookup_key in completed_keys:
                    resumed_count += 1
                    print(
                        f"[{call_index}/{total_planned}] (RESUMED) ID #{p_id} ({p_label.upper()}) | "
                        f"{model} | {wrapper_name}"
                    )
                    continue

                wrapper_cfg = WRAPPERS[wrapper_name]
                messages = [
                    {"role": "system", "content": wrapper_cfg["system_prompt"]},
                    {"role": "user", "content": p_text},
                ]

                # Make API call with response_format fallback
                resp = client.generate(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=client.default_max_tokens,
                    response_format=wrapper_cfg.get("response_format"),
                )

                # Fallback if model rejected response_format
                if not resp["ok"] and resp.get("error") and "response_format" in str(resp.get("error")):
                    resp = client.generate(
                        model=model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=client.default_max_tokens,
                        response_format=None,
                    )

                session_calls += 1

                # Parse JSON or clean text
                parsed_json, json_valid, declared_status, response_text = parse_model_response(
                    wrapper_name=wrapper_name,
                    content=resp.get("content", ""),
                    reasoning=resp.get("reasoning", ""),
                )

                if wrapper_name not in ("free_text", "free_text_blocked"):
                    json_total_count += 1
                    if json_valid:
                        json_valid_count += 1

                if resp["ok"]:
                    # Build and stream successful record immediately line-by-line
                    record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "mode": mode,
                        "prompt_id": p_id,
                        "label": p_label,
                        "category": p_cat,
                        "focus": p_focus,
                        "model": model,
                        "wrapper": wrapper_name,
                        "ok": True,
                        "content": resp.get("content", ""),
                        "reasoning": resp.get("reasoning", ""),
                        "parsed_json": parsed_json,
                        "json_valid": json_valid,
                        "declared_status": declared_status,
                        "response_text": response_text,
                        "cost_usd": resp.get("cost_usd", 0.0),
                        "latency_ms": resp.get("latency_ms", 0.0),
                        "error": None,
                    }

                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()

                    completed_keys.add(lookup_key)

                    # Progress output
                    status_disp = f"Status: {declared_status}" if declared_status else "Status: -"
                    json_disp = f"JSON: {'Valid' if json_valid else 'Invalid'}" if wrapper_name not in ("free_text", "free_text_blocked") else "Plain"
                    print(
                        f"[{call_index}/{total_planned}] ID #{p_id} ({p_label.upper()}) | "
                        f"{model} | {wrapper_name} | OK: True | {json_disp} | {status_disp} | "
                        f"Latency: {resp['latency_ms']}ms"
                    )
                else:
                    # NEVER cache ok: false into .jsonl
                    print(
                        f"[{call_index}/{total_planned}] (FAILED) ID #{p_id} ({p_label.upper()}) | "
                        f"{model} | {wrapper_name} | Error: {resp.get('error')}. Skipping cache."
                    )

        if stop_experiment:
            break

    print("\n" + "=" * 70)
    print("EXPERIMENT RUN SUMMARY")
    print("=" * 70)
    print(f"Total planned calls:        {total_planned}")
    print(f"Cached / resumed calls:     {resumed_count}")
    print(f"New API calls made:         {session_calls}")
    print(f"Total active records:       {len(completed_keys)}")
    if json_total_count > 0:
        validity_pct = (json_valid_count / json_total_count) * 100
        print(f"JSON parse validity rate:   {json_valid_count}/{json_total_count} ({validity_pct:.1f}%)")
    print(f"Total session cost:         ${client.total_cost_usd:.4f}")
    print(f"Output saved to:            {output_path}")
    print("=" * 70 + "\n")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Safety Checks JSON Schema Experiment Runner")
    parser.add_argument("--mode", choices=["free", "paid"], default=MODE, help="Execution mode (free or paid)")
    parser.add_argument("--limit", type=int, default=None, help="Number of prompts to evaluate")
    parser.add_argument("--models-limit", type=int, default=None, help="Number of models to evaluate")
    parser.add_argument("--max-calls", type=int, default=None, help="Maximum number of new API calls to make")
    parser.add_argument("--dry-run", action="store_true", help="Shortcut for 8-call dry run (--limit 2 --models-limit 1)")
    parser.add_argument("--output", type=Path, default=None, help="Custom output .jsonl path")
    parser.add_argument("--include-free-text-blocked", action="store_true", help="Include 5th wrapper 'free_text_blocked'")
    parser.add_argument("--wrappers", nargs="+", default=None, help="Explicit list of wrappers to run")
    args = parser.parse_args()

    wrappers = args.wrappers
    if wrappers is None and args.include_free_text_blocked:
        wrappers = ACTIVE_WRAPPERS

    run_experiment(
        mode=args.mode,
        limit=args.limit,
        models_limit=args.models_limit,
        output_path=args.output,
        dry_run=args.dry_run,
        wrappers=wrappers,
        max_calls=args.max_calls,
    )


if __name__ == "__main__":
    main()
