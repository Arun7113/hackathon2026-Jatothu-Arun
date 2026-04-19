"""
agent.py — Public entry point for processing a single ticket.
Builds the initial AgentState and invokes the compiled LangGraph graph.
"""
from langchain_core.messages import HumanMessage, AIMessage

from state import AgentState
from graph import compiled_graph
from utils import log_step


def run_agent(ticket: dict) -> dict:
    """
    Process one support ticket end-to-end through the LangGraph pipeline.

    Args:
        ticket: dict with ticket_id, customer_email, subject, body, source, tier

    Returns:
        dict with ticket_id, classification, resolution, escalated,
              refund_issued, tool_call_count, audit_trail, confidence
    """
    ticket_id      = ticket["ticket_id"]
    customer_email = ticket["customer_email"]

    log_step(
        "ticket_ingested",
        {"ticket_id": ticket_id, "source": ticket.get("source")},
        {"subject": ticket["subject"]},
        f"Ticket {ticket_id} ingested: '{ticket['subject']}'",
        confidence=1.0,
    )

    initial_state: AgentState = {
        # Raw ticket
        "ticket_id":      ticket_id,
        "customer_email": customer_email,
        "subject":        ticket["subject"],
        "body":           ticket["body"],
        "source":         ticket.get("source", "unknown"),

        # Ingest
        "extracted_order_id":  None,
        "extracted_order_ids": [],

        # Lookup
        "customer":    None,
        "order":       None,
        "product":     None,
        "all_orders":  [],
        "lookup_tool_calls": 0,

        # Classifier
        "classification": "",

        # KB search
        "kb_result": None,

        # Triage
        "urgency":           "medium",
        "resolvable":        True,
        "confidence":        1.0,
        "confidence_signals": {},
        "triage_reasoning":  "",

        # Act
        "refund_eligible":           None,
        "refund_eligibility_reason": None,
        "refund_issued":             None,
        "cancellation_result":       None,
        "action_taken":              "",
        "action_reasoning":          "",
        "customer_reply":            "",

        # Reply
        "reply_sent": False,
        "escalated":  False,

        # Cross-cutting
        "tool_calls_made": 0,
        "audit_trail":     [],
        "messages": [
            HumanMessage(content=(
                f"Support ticket {ticket_id}\n"
                f"From: {customer_email}\n"
                f"Subject: {ticket['subject']}\n"
                f"Body: {ticket['body']}"
            ))
        ],
    }

    # Run the graph
    final_state = compiled_graph.invoke(initial_state)

    # Extract final resolution text
    resolution = final_state.get("customer_reply", "")
    if not resolution:
        for msg in reversed(final_state.get("messages") or []):
            if isinstance(msg, AIMessage) and msg.content:
                resolution = msg.content
                break
    if not resolution:
        resolution = "Ticket processed."

    result = {
        "ticket_id":       ticket_id,
        "classification":  final_state.get("classification", ""),
        "resolution":      resolution,
        "escalated":       final_state.get("escalated", False),
        "refund_issued":   final_state.get("refund_issued", False),
        "tool_call_count": final_state.get("tool_calls_made", 0),
        "confidence":      final_state.get("confidence", 0.0),
        "urgency":         final_state.get("urgency", "medium"),
        "action_taken":    final_state.get("action_taken", ""),
        "audit_trail":     final_state.get("audit_trail", []),
        "reply_sent":      final_state.get("reply_sent", False),
    }

    log_step(
        "ticket_complete",
        {"ticket_id": ticket_id},
        {k: v for k, v in result.items() if k not in ("resolution", "audit_trail")},
        f"Ticket {ticket_id} complete — "
        f"class={result['classification']} "
        f"action={result['action_taken']} "
        f"tool_calls={result['tool_call_count']} "
        f"escalated={result['escalated']} "
        f"conf={result['confidence']:.2f}",
        confidence=result["confidence"],
    )

    return result
