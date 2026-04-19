"""
utils.py — Audit logging + retry decorator.

Every tool call, routing decision, LLM reasoning step, and outcome
is captured here.  No black-box outputs — every decision is explainable.
"""
import json
import logging
import functools
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ── Global audit log ──────────────────────────────────────────────────────────
audit_log: list[dict] = []


def log_step(
    step_type: str,
    input_data: dict,
    output_data: dict,
    reasoning: str = "",
    confidence: float = 1.0,
):
    """
    Append one explainable decision step to the audit log.
    Called for every node transition, tool call, routing decision,
    and LLM reasoning step — not just final answers.
    """
    entry = {
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "step":       step_type,
        "input":      input_data,
        "output":     output_data,
        "reasoning":  reasoning,
        "confidence": round(confidence, 3),
    }
    audit_log.append(entry)
    logging.info(f"[{step_type}] (conf={confidence:.2f}) {reasoning}")


def save_audit_log(filename: str = "audit_log.json"):
    """
    Save audit log grouped by ticket_id.
    Each ticket gets one JSON object containing all its audit steps.
    System-level steps (no ticket_id) go under '_system'.
    """
    grouped = {}
    for entry in audit_log:
        # Find ticket_id from the entry's input data
        ticket_id = (
            entry.get("input", {}).get("ticket_id") or
            entry.get("output", {}).get("ticket_id") or
            "_system"
        )
        if ticket_id not in grouped:
            grouped[ticket_id] = {
                "ticket_id": ticket_id,
                "audit_steps": []
            }
        grouped[ticket_id]["audit_steps"].append(entry)

    # Convert to list, system entries last
    result = [v for k, v in grouped.items() if k != "_system"]
    if "_system" in grouped:
        result.append(grouped["_system"])

    with open(filename, "w") as f:
        json.dump(result, f, indent=2, default=str, ensure_ascii=False)
    logging.info(f"Audit log saved → {filename} ({len(audit_log)} entries across {len(result)} tickets)")


def retry_tool(max_attempts: int = 3, base_wait: float = 1.0):
    """
    Decorator: retry up to max_attempts with exponential back-off.
    On permanent failure returns a structured error dict — NEVER raises.

    This satisfies the 'Recover' constraint:
    tool timeouts / malformed responses don't crash the agent.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_error = exc
                    wait = base_wait * (2 ** (attempt - 1))
                    logging.warning(
                        f"[retry] {func.__name__} attempt {attempt}/{max_attempts} "
                        f"failed: {exc}. Retrying in {wait:.1f}s…"
                    )
                    if attempt < max_attempts:
                        time.sleep(wait)

            error = {
                "error":   True,
                "tool":    func.__name__,
                "message": f"Permanent failure after {max_attempts} attempts: {last_error}",
            }
            log_step(
                f"{func.__name__}_failure",
                {}, error,
                f"All {max_attempts} retries exhausted: {last_error}",
                confidence=0.0,
            )
            return error
        return wrapper
    return decorator
