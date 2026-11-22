# Safety Checks for JSON Schema Study

## 1. Project Overview & Research Goal
Investigate whether forcing LLMs into strict JSON output schemas induces a **"safety tax"** compared to standard free-form chat. Specifically, does JSON schema enforcement:
- Reduce necessary refusals on unsafe prompts (jailbreak/bypass vector)?
- Increase false over-refusals on benign borderline prompts (utility degradation)?

Evaluations use paired safe and unsafe prompts from the public **walledai/XSTest** dataset across 4 distinct output formats.

---

## 2. Experimental Design: 4 Output Wrappers

1. **`free_text`**: Standard free-form conversational response without forced structuring.
2. **`strict_json`**: Strict JSON schema:
   ```json
   {
     "status": "ok" | "blocked",
     "response": "string"
   }
   ```
3. **`json_with_reasoning`**: Strict JSON schema requiring a prior reasoning check before answering:
   ```json
   {
     "safety_check": "string (rationale evaluating risk)",
     "status": "ok" | "blocked",
     "response": "string"
   }
   ```
4. **`json_with_escalate`**: Strict JSON schema providing a tri-state status for uncertain or borderline cases:
   ```json
   {
     "status": "ok" | "blocked" | "escalate_to_human",
     "response": "string"
   }
   ```

---

## 3. Strict Technical & Architectural Constraints

- **Platform / OS**: Windows PowerShell using the existing `.venv` Python virtual environment.
- **Low-RAM Footprint**:
  - Strictly use Python standard library plus `requests` and `matplotlib`.
  - **Forbidden**: `pandas`, `numpy`, `torch`, `transformers`, or agent orchestration frameworks.
- **API Architecture**:
  - Gateway: OpenRouter API (`https://openrouter.ai/api/v1/chat/completions`).
  - Auth: `OPENROUTER_API_KEY` parsed directly from local `.env` via standard library string parsing (no `python-dotenv` dependency).
- **Free-First Execution Modes**:
  - **`MODE = "free"` (Default)**:
    - Target Models:
      - `nvidia/nemotron-3.5-lightning:free`
      - `google/gemma-4-26b-a4b-it:free`
      - `baidu/ernie-5.5-flash-preview:free`
    - Rate Control: Minimum 3.5s sleep between calls.
    - Safety Cap: Hard limit of **24 total calls per run** (safely below OpenRouter's 50 requests/day limit).
  - **`MODE = "paid"` (Optional)**:
    - Target Models:
      - `anthropic/claude-haiku-5.5`
      - `google/gemini-3.7-flash`
      - `deepseek/deepseek-v4-flash-0731`
    - Hard Budget Cap: **$1.80** maximum spend guard (strictly beneath the $2 account cap).
- **Fault Tolerance & Reliability**:
  - All experiment runs stream results line-by-line to `.jsonl` files.
  - Automatic resume: Queries already recorded in `.jsonl` are detected and skipped on subsequent runs to prevent duplicate API calls and wasted quota.
- **Source Dataset**: `walledai/XSTest` (curated safe vs unsafe prompts).
