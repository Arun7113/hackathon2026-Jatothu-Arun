"""
main.py — ShopWave Support System Entry Point
=============================================
Ingests all 20 tickets, processes them in concurrent batches,
saves audit_log.json after EVERY batch (never lose progress).

Single command:  python main.py

.env variables:
  GEMINI_API_KEY      required
  GEMINI_FAST_MODEL   default: gemini-2.0-flash-lite  (classifier)
  GEMINI_SMART_MODEL  default: gemini-2.0-flash        (triage/act/reply)
  MAX_TICKETS         default: 20
  BATCH_SIZE          default: 3   (keep low on free tier)
  BATCH_DELAY         default: 15  (seconds between batches)
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from tools import load_data
from agent import run_agent
from utils import save_audit_log, audit_log, log_step

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "3"))
BATCH_DELAY = float(os.getenv("BATCH_DELAY", "15"))


async def process_ticket_async(ticket: dict) -> dict:
    """Run one ticket on the thread-pool executor (non-blocking)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_agent, ticket)


async def main():
    # ── 1. Load data ──────────────────────────────────────────────────────
    load_data()
    logging.info("Data loaded")

    # ── 2. Ingest tickets ─────────────────────────────────────────────────
    with open(Path("data/tickets.json")) as f:
        tickets: list[dict] = json.load(f)

    max_t = int(os.getenv("MAX_TICKETS", len(tickets)))
    tickets = tickets[:max_t]

    log_step(
        "system_start",
        {"total_tickets": len(tickets), "batch_size": BATCH_SIZE,
         "batch_delay": BATCH_DELAY},
        {},
        f"Starting: {len(tickets)} tickets, batch_size={BATCH_SIZE}, delay={BATCH_DELAY}s",
        confidence=1.0,
    )
    logging.info(
        f"Processing {len(tickets)} tickets — "
        f"batch_size={BATCH_SIZE}, delay={BATCH_DELAY}s"
    )

    # ── 3. Concurrent batches ─────────────────────────────────────────────
    all_results: list[dict] = []
    total_batches = (len(tickets) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(tickets), BATCH_SIZE):
        batch = tickets[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        ids = [t["ticket_id"] for t in batch]
        logging.info(f"Batch {batch_num}/{total_batches}: {ids}")

        # All tickets in this batch run concurrently ✅
        tasks = [process_ticket_async(t) for t in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for ticket, result in zip(batch, results):
            if isinstance(result, Exception):
                logging.error(f"[{ticket['ticket_id']}] EXCEPTION: {result}")
                # Dead-letter: failed tickets are logged, not silently dropped
                all_results.append({
                    "ticket_id":       ticket["ticket_id"],
                    "classification":  "error",
                    "resolution":      f"SYSTEM ERROR: {result}",
                    "escalated":       False,
                    "refund_issued":   False,
                    "tool_call_count": 0,
                    "confidence":      0.0,
                    "urgency":         "unknown",
                    "action_taken":    "error",
                    "audit_trail":     [f"Unhandled exception: {result}"],
                    "reply_sent":      False,
                    "error":           True,
                })
                log_step(
                    "ticket_error",
                    {"ticket_id": ticket["ticket_id"]},
                    {"error": str(result)},
                    f"Ticket {ticket['ticket_id']} failed — logged to dead-letter",
                    confidence=0.0,
                )
            else:
                all_results.append(result)

        # ── Save after every batch — progress never lost ──────────────────
        save_audit_log("audit_log.json")
        logging.info(
            f"Batch {batch_num}/{total_batches} done — "
            f"audit_log.json updated ({len(audit_log)} entries)"
        )

        if i + BATCH_SIZE < len(tickets):
            logging.info(f"Waiting {BATCH_DELAY}s before next batch…")
            await asyncio.sleep(BATCH_DELAY)
        

    # ── 4. Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'SHOPWAVE — AUTONOMOUS SUPPORT AGENT — RESULTS':^78}")
    print("=" * 78)

    resolved = escalated = refunded = errors = 0

    for r in all_results:
        if r.get("error"):
            errors += 1
            tag = "💥 ERROR"
        elif r.get("escalated"):
            escalated += 1
            tag = "⬆ ESCALATED"
        else:
            resolved += 1
            tag = "✅ RESOLVED"

        if r.get("refund_issued"):
            refunded += 1

        cls  = r.get("classification", "?").upper()
        tc   = r.get("tool_call_count", 0)
        conf = r.get("confidence", 0.0)
        act  = r.get("action_taken", "?")

        print(
            f"\n[{r['ticket_id']}] {tag:<14} "
            f"class={cls:<14} action={act:<22} "
            f"tools={tc}  conf={conf:.2f}"
        )
        preview = r.get("resolution", "")[:180]
        if preview:
            print(f"  → {preview}{'…' if len(r.get('resolution',''))>180 else ''}")
        for step in r.get("audit_trail", []):
            print(f"     {step}")

    # Summary stats
    print("\n" + "─" * 78)
    print(f"  Total       : {len(all_results)}")
    print(f"  Resolved    : {resolved}")
    print(f"  Escalated   : {escalated}")
    print(f"  Refunds     : {refunded}")
    print(f"  Errors      : {errors}")
    print(f"  Audit entries: {len(audit_log)}")
    print(f"  Audit file  : audit_log.json")
    print("─" * 78 + "\n")

    # ── 5. Final audit save ───────────────────────────────────────────────
    save_ticket_summaries(all_results)
    log_step(
        "system_complete",
        {},
        {"resolved": resolved, "escalated": escalated,
         "refunded": refunded, "errors": errors,
         "total_audit_entries": len(audit_log)},
        "All tickets processed",
        confidence=1.0,
    )
    save_audit_log("audit_log.json")
    logging.info("Done.")

import json
from pathlib import Path

def save_ticket_summaries(results: list[dict], filename: str = "ticket_summaries.json"):
    """
    Human-readable per-ticket summaries showing the complete flow path,
    every tool called, why decisions were made, and final outcome.
    
    Purpose: judges/reviewers can read ONE entry and understand exactly
    what happened — without parsing hundreds of raw audit log entries.
    """
    summaries = []

    for r in results:
        trail = r.get("audit_trail", [])

        # ── Extract tools actually called ─────────────────────────────────
        tools_called = []
        for line in trail:
            tool_map = {
                "[lookup] customer=":       ("get_customer",            "READ"),
                "[lookup] order=":          ("get_order",               "READ"),
                "[lookup] no order_id":     ("get_orders_by_customer",  "READ"),
                "[lookup] no product_id":   ("search_knowledge_base",   "READ"),
                "[lookup] product=":        ("get_product",             "READ"),
                "[kb_search]":              ("search_knowledge_base",   "READ"),
                "[act/refund] eligibility": ("check_refund_eligibility","READ"),
                "[act/refund] issued":      ("issue_refund",            "WRITE"),
                "[act/refund] escalated":   ("escalate",               "WRITE"),
                "[act/cancel]":             ("cancel_order",            "WRITE"),
                "[act/warranty] escalated": ("escalate",               "WRITE"),
                "[act/escalate]":           ("escalate",               "WRITE"),
                "[act] escalated":          ("escalate",               "WRITE"),
                "[reply] send_reply":       ("send_reply",             "WRITE"),
            }
            for pattern, (tool_name, tool_type) in tool_map.items():
                if pattern in line:
                    # Avoid duplicate consecutive tool entries
                    if not tools_called or tools_called[-1]["tool"] != tool_name:
                        tools_called.append({
                            "tool":   tool_name,
                            "type":   tool_type,
                            "detail": line.strip().replace("[act/refund] ", "")
                                                   .replace("[lookup] ", "")
                                                   .replace("[act/cancel] ", "")
                                                   .replace("[reply] ", "")
                        })
                    break

        # ── Extract routing path ──────────────────────────────────────────
        flow_path = ["ingest", "lookup", "classify", "kb_search", "triage"]
        action    = r.get("action_taken", "")
        if "escalat" in action:
            flow_path += ["triage → escalate", "human_agent"]
        elif action == "refund_issued":
            flow_path += ["act → check_eligibility → issue_refund", "reply"]
        elif action == "refund_denied":
            flow_path += ["act → check_eligibility → deny", "reply"]
        elif action == "cancel_attempted":
            flow_path += ["act → cancel_order", "reply"]
        elif action == "order_status_provided":
            flow_path += ["act → status_lookup", "reply"]
        elif action == "general_inquiry_answered":
            flow_path += ["act → kb_answer", "reply"]
        else:
            flow_path += ["act", "reply"]

        # ── Extract key decisions from trail ──────────────────────────────
        decisions = []
        for line in trail:
            if "[classifier]" in line:
                decisions.append({"node": "classifier", "decision": line.replace("[classifier] ", "").strip()})
            elif "[triage]" in line:
                decisions.append({"node": "triage",     "decision": line.replace("[triage] ",     "").strip()})
            elif "[act/refund]" in line and "decision" in line.lower():
                decisions.append({"node": "act",        "decision": line.replace("[act/refund] ", "").strip()})
            elif "[act/" in line and "escalated" in line:
                decisions.append({"node": "act",        "decision": line.strip()})

        # ── Outcome label ─────────────────────────────────────────────────
        if r.get("error"):
            outcome = "ERROR"
            outcome_detail = r.get("resolution", "")[:120]
        elif r.get("escalated"):
            outcome = "ESCALATED_TO_HUMAN"
            outcome_detail = f"Priority: {r.get('urgency','?')} — routed to Support Specialist"
        elif action == "refund_issued":
            outcome = "RESOLVED_REFUND_ISSUED"
            outcome_detail = f"Refund processed for ticket"
        elif action == "refund_denied":
            outcome = "RESOLVED_REFUND_DENIED"
            outcome_detail = "Customer informed of ineligibility with policy explanation"
        elif action == "cancel_attempted":
            outcome = "RESOLVED_ORDER_CANCELLED"
            outcome_detail = "Cancellation processed"
        elif action == "order_status_provided":
            outcome = "RESOLVED_STATUS_PROVIDED"
            outcome_detail = "Shipping/delivery info sent to customer"
        else:
            outcome = "RESOLVED_GENERAL"
            outcome_detail = "Policy/FAQ answer sent"

        # ── Confidence breakdown (human-readable) ─────────────────────────
        conf_signals = {}
        for line in trail:
            if "[triage]" in line and "Confidence=" in line:
                # Parse "customer_found=✓, order_found=✗ ..."
                import re
                matches = re.findall(r'(\w+)=([✓✗])', line)
                conf_signals = {k: (v == "✓") for k, v in matches}
                break

        summary = {
            # ── Identity ──────────────────────────────────────────────────
            "ticket_id":      r.get("ticket_id"),
            "classification": r.get("classification"),
            "urgency":        r.get("urgency"),

            # ── Flow taken ────────────────────────────────────────────────
            "flow_path":      " → ".join(flow_path),
            "tool_call_count": r.get("tool_call_count", 0),
            "tools_called":   tools_called,      # ordered list with type + detail

            # ── Key decisions at each node ────────────────────────────────
            "decisions":      decisions,

            # ── Confidence ────────────────────────────────────────────────
            "confidence":         round(r.get("confidence", 0.0), 3),
            "confidence_signals": conf_signals,  # which data was available

            # ── Outcome ───────────────────────────────────────────────────
            "outcome":          outcome,
            "outcome_detail":   outcome_detail,
            "escalated":        r.get("escalated", False),
            "refund_issued":    r.get("refund_issued", False),
            "action_taken":     action,
            "reply_sent":       r.get("reply_sent", False),

            # ── Customer-facing reply ─────────────────────────────────────
            "customer_reply": r.get("resolution", ""),
        }
        summaries.append(summary)

    with open(filename, "w") as f:
        json.dump(summaries, f, indent=2,ensure_ascii=False)
    logging.info(f"Ticket summaries saved → {filename} ({len(summaries)} tickets)")


if __name__ == "__main__":
    asyncio.run(main())
