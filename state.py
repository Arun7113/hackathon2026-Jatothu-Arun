"""
state.py — LangGraph AgentState for ShopWave.

The state is the agent's working memory.  Every node reads from it
and writes back enriched information.  By the time we reach [act],
the state contains full customer/order/product/policy context.

New architecture order:
  ingest → lookup → classifier → kb_search → triage → act → reply
"""
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):

    # ── Raw ticket ────────────────────────────────────────────────────────
    ticket_id:      str
    customer_email: str
    subject:        str
    body:           str
    source:         str

    # ── Ingest outputs ────────────────────────────────────────────────────
    extracted_order_id:   Optional[str]   # regex-extracted from body
    extracted_order_ids:  list[str]       # all ORD-XXXX found in body

    # ── Lookup outputs (guaranteed ≥3 tool calls) ─────────────────────────
    customer:   Optional[dict]   # get_customer result
    order:      Optional[dict]   # get_order / latest order result
    product:    Optional[dict]   # get_product result
    all_orders: list[dict]       # all customer orders (when no order_id)
    lookup_tool_calls: int       # how many tool calls lookup made (must be ≥3)

    # ── Classifier outputs (runs AFTER lookup for full context) ───────────
    # refund | cancellation | order_status | warranty | escalate | general
    classification: str

    # ── KB search output ──────────────────────────────────────────────────
    kb_result: Optional[str]

    # ── Triage outputs ────────────────────────────────────────────────────
    urgency:         str          # low | medium | high | urgent
    resolvable:      bool         # can the agent resolve without human?
    confidence:      float        # 0.0–1.0 confidence score
    confidence_signals: dict      # breakdown of how confidence was computed
    triage_reasoning: str         # human-readable explanation

    # ── Act outputs ───────────────────────────────────────────────────────
    refund_eligible:          Optional[bool]
    refund_eligibility_reason: Optional[str]
    refund_issued:            Optional[bool]
    cancellation_result:      Optional[dict]
    action_taken:             str           # what the act node did
    action_reasoning:         str           # why

    # ── Reply output ──────────────────────────────────────────────────────
    customer_reply: str
    reply_sent:     bool
    escalated:      bool

    # ── Cross-cutting ─────────────────────────────────────────────────────
    tool_calls_made: int
    audit_trail:     list[str]    # human-readable per-step decisions
    messages: Annotated[list[BaseMessage], add_messages]
