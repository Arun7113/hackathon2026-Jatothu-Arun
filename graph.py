"""
graph.py — LangGraph StateGraph for ShopWave (new architecture).

Flow:
  START → ingest → lookup → classifier → kb_search → triage → act → reply → END

All nodes are connected linearly.
The triage node's 'resolvable' flag drives the act node's internal routing
(no extra conditional edges needed — act handles all branching internally).
This is simpler and more debuggable than many conditional edges.

The compiled_graph is a module-level singleton — built once, reused for all tickets.
"""
from langgraph.graph import StateGraph, END
from langgraph.constants import START

from state import AgentState
from nodes import (
    ingest_node,
    lookup_node,
    classifier_node,
    kb_search_node,
    triage_node,
    act_node,
    reply_node,
)
from utils import log_step


def build_graph() -> StateGraph:
    """Build and compile the ShopWave support agent graph."""
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("ingest",     ingest_node)
    builder.add_node("lookup",     lookup_node)
    builder.add_node("classifier", classifier_node)
    builder.add_node("kb_search",  kb_search_node)
    builder.add_node("triage",     triage_node)
    builder.add_node("act",        act_node)
    builder.add_node("reply",      reply_node)

    # ── Linear pipeline edges ─────────────────────────────────────────────
    builder.add_edge(START,        "ingest")
    builder.add_edge("ingest",     "lookup")
    builder.add_edge("lookup",     "classifier")
    builder.add_edge("classifier", "kb_search")
    builder.add_edge("kb_search",  "triage")
    builder.add_edge("triage",     "act")
    builder.add_edge("act",        "reply")
    builder.add_edge("reply",      END)

    log_step(
        "graph_compiled",
        {"nodes": ["ingest","lookup","classifier","kb_search","triage","act","reply"]},
        {},
        "LangGraph StateGraph compiled successfully",
        confidence=1.0,
    )
    return builder.compile()


# Compiled once at import time — reused for all tickets
compiled_graph = build_graph()
