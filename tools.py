"""
tools.py — ShopWave Mock Tool Implementations
=============================================
All 8 tools from the PDF spec, plus 2 internal helpers.

READ / LOOKUP
  get_order(order_id)
  get_customer(email)
  get_product(product_id)
  search_knowledge_base(query)

WRITE / ACT
  check_refund_eligibility(order_id)   ← may throw errors (simulated)
  issue_refund(order_id, amount)       ← IRREVERSIBLE
  send_reply(ticket_id, message)
  escalate(ticket_id, summary, priority)

Internal helpers
  get_orders_by_customer(customer_id)
  cancel_order(order_id)

Failure simulation: tools randomly timeout or return malformed data
to satisfy the 'Recover' constraint.  The @retry_tool decorator handles
graceful recovery — always returns a result dict, never raises to caller.
"""
import json
import random
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from models import Customer, Order, Product
from utils import log_step, retry_tool

# ── In-memory stores ──────────────────────────────────────────────────────────
customers_db: dict[str, Customer] = {}   # key = email
orders_db:    dict[str, Order]    = {}   # key = order_id
products_db:  dict[str, Product]  = {}   # key = product_id
_kb_text: str = ""                        # full knowledge_base.md text


def load_data(base_dir: str = "data"):
    global customers_db, orders_db, products_db, _kb_text

    with open(Path(base_dir) / "customers.json") as f:
        for c in json.load(f):
            obj = Customer(**c)
            customers_db[obj.email] = obj

    with open(Path(base_dir) / "orders.json") as f:
        for o in json.load(f):
            obj = Order(**o)
            orders_db[obj.order_id] = obj

    with open(Path(base_dir) / "products.json") as f:
        for p in json.load(f):
            obj = Product(**p)
            products_db[obj.product_id] = obj

    kb = Path(base_dir) / "knowledge_base.md"
    if kb.exists():
        _kb_text = kb.read_text(encoding="utf-8")

    print(
        f"[load_data] {len(customers_db)} customers | "
        f"{len(orders_db)} orders | "
        f"{len(products_db)} products | "
        f"KB {len(_kb_text)} chars"
    )


# ── READ / LOOKUP ─────────────────────────────────────────────────────────────

@retry_tool(max_attempts=3, base_wait=1.0)
def get_order(order_id: str) -> dict:
    """Order details, status, timestamps."""
    if random.random() < 0.08:
        raise TimeoutError("Order service timeout")
    order = orders_db.get(order_id)
    result = (
        order.model_dump(mode="json") if order
        else {"error": True, "message": f"Order {order_id} not found"}
    )
    log_step("get_order", {"order_id": order_id}, result,
             f"status={order.status if order else 'not_found'}")
    return result


@retry_tool(max_attempts=3, base_wait=1.0)
def get_customer(email: str) -> dict:
    """Customer profile, tier, history."""
    if random.random() < 0.08:
        raise TimeoutError("Customer service timeout")
    customer = customers_db.get(email)
    result = (
        customer.model_dump(mode="json") if customer
        else {"error": True, "message": f"Customer {email} not found"}
    )
    log_step("get_customer", {"email": email}, result,
             f"name={customer.name if customer else 'not_found'} "
             f"tier={customer.tier if customer else 'n/a'}")
    return result


@retry_tool(max_attempts=3, base_wait=1.0)
def get_product(product_id: str) -> dict:
    """Product metadata, category, warranty."""
    if random.random() < 0.08:
        raise TimeoutError("Product catalog unavailable")
    product = products_db.get(product_id)
    result = (
        product.model_dump(mode="json") if product
        else {"error": True, "message": f"Product {product_id} not found"}
    )
    log_step("get_product", {"product_id": product_id}, result,
             f"name={product.name if product else 'not_found'}")
    return result


@retry_tool(max_attempts=3, base_wait=1.0)
def search_knowledge_base(query: str) -> str:
    """Policy & FAQ semantic search over knowledge_base.md."""
    if not _kb_text:
        return "KB unavailable. Default: 30-day returns, 5-7 day refunds."

    # Parse H2 sections
    sections: list[tuple[str, str]] = []
    current_title: Optional[str] = None
    current_lines: list[str] = []
    for line in _kb_text.splitlines():
        if line.startswith("## "):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)
    if current_title and current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    # Keyword scoring
    query_lower = query.lower()
    query_words = set(query_lower.split())
    scored = []
    for title, body in sections:
        combined = (title + " " + body).lower()
        score = sum(1 for w in query_words if w in combined)
        if query_lower in combined:
            score += 5
        scored.append({"score": score, "title": title, "body": body})
    scored.sort(key=lambda x: x["score"], reverse=True)

    top = [f"### {s['title']}\n{s['body']}" for s in scored[:2] if s["score"] > 0]
    result = "\n\n".join(top) if top else _kb_text[:2000]

    log_step("search_knowledge_base", {"query": query},
             {"sections": [s["title"] for s in scored[:2] if s["score"] > 0]},
             f"Matched {len(top)} sections for: '{query[:60]}'")
    return result


# ── WRITE / ACT ───────────────────────────────────────────────────────────────

@retry_tool(max_attempts=3, base_wait=1.0)
def check_refund_eligibility(order_id: str) -> dict:
    """
    Returns eligibility + reason.
    Simulates 10% malformed response — tests Recover constraint.
    """
    if random.random() < 0.10:
        raise ValueError("Eligibility service returned malformed JSON")

    order = orders_db.get(order_id)
    if not order:
        result = {"eligible": False, "reason": "Order not found", "order_id": order_id}
        log_step("check_refund_eligibility", {"order_id": order_id}, result,
                 "Order not found", confidence=0.0)
        return result

    customer = next(
        (c for c in customers_db.values() if c.customer_id == order.customer_id), None
    )
    today = date.today()

    if order.refund_status == "refunded":
        eligible, reason = False, "Order already refunded"
    elif order.return_deadline and today > order.return_deadline:
        is_vip = customer and customer.tier == "vip"
        has_exception = (
            "extended return" in (customer.notes if customer else "").lower() or
            "pre-approved" in order.notes.lower()
        )
        if is_vip and has_exception:
            eligible, reason = True, "VIP pre-approved extended return exception"
        else:
            eligible, reason = False, f"Return window expired on {order.return_deadline}"
    elif order.amount > 200:
        eligible, reason = False, (
            f"Amount ${order.amount:.2f} exceeds $200 auto-refund limit — escalation required"
        )
    else:
        eligible, reason = True, "Within return window, no restrictions"

    result = {
        "eligible":        eligible,
        "reason":          reason,
        "order_id":        order_id,
        "order_amount":    order.amount,
        "order_status":    order.status,
        "return_deadline": str(order.return_deadline) if order.return_deadline else None,
    }
    log_step("check_refund_eligibility", {"order_id": order_id}, result,
             f"eligible={eligible}: {reason}", confidence=0.9)
    return result


@retry_tool(max_attempts=3, base_wait=1.0)
def issue_refund(order_id: str, amount: float) -> dict:
    """IRREVERSIBLE — issue a refund. Must check eligibility first."""
    # Schema validation before acting
    if not isinstance(amount, (int, float)) or amount <= 0:
        return {"success": False, "message": f"Invalid refund amount: {amount}"}

    order = orders_db.get(order_id)
    if not order:
        result = {"success": False, "message": f"Order {order_id} not found"}
    elif order.refund_status == "refunded":
        result = {"success": False, "message": "Already refunded — cannot double-refund"}
    else:
        order.refund_status = "refunded"
        refund_id = f"REF-{order_id}-{int(datetime.utcnow().timestamp())}"
        result = {
            "success":         True,
            "refund_id":       refund_id,
            "message":         f"Refund of ${amount:.2f} initiated for {order_id}",
            "processing_time": "5-7 business days",
        }
    log_step("issue_refund", {"order_id": order_id, "amount": amount}, result,
             result.get("message", ""), confidence=1.0)
    return result


@retry_tool(max_attempts=3, base_wait=1.0)
def send_reply(ticket_id: str, message: str) -> dict:
    """Sends response to the customer."""
    result = {
        "sent":      True,
        "ticket_id": ticket_id,
        "preview":   message[:120] + ("…" if len(message) > 120 else ""),
    }
    log_step("send_reply", {"ticket_id": ticket_id},
             result, f"Reply sent for {ticket_id}")
    return result


@retry_tool(max_attempts=3, base_wait=1.0)
def escalate(ticket_id: str, summary: str, priority: str) -> dict:
    """Routes to human with full context."""
    if priority not in {"low", "medium", "high", "urgent"}:
        priority = "medium"
    result = {
        "escalated":   True,
        "ticket_id":   ticket_id,
        "priority":    priority,
        "summary":     summary,
        "assigned_to": "Support Specialist",
        "timestamp":   datetime.utcnow().isoformat() + "Z",
    }
    log_step("escalate", {"ticket_id": ticket_id, "priority": priority},
             result, f"Escalated ({priority}): {summary[:80]}")
    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

@retry_tool(max_attempts=3, base_wait=1.0)
def get_orders_by_customer(customer_id: str) -> list:
    """All orders for a customer (fallback when no order_id in ticket)."""
    if random.random() < 0.08:
        raise TimeoutError("Order service timeout")
    orders = [
        o.model_dump(mode="json")
        for o in orders_db.values()
        if o.customer_id == customer_id
    ]
    log_step("get_orders_by_customer", {"customer_id": customer_id},
             {"count": len(orders)}, f"Found {len(orders)} orders")
    return orders


@retry_tool(max_attempts=3, base_wait=1.0)
def cancel_order(order_id: str) -> dict:
    """Cancel an order in 'processing' status."""
    order = orders_db.get(order_id)
    if not order:
        result = {"success": False, "message": f"Order {order_id} not found"}
    elif order.status == "processing":
        order.status = "cancelled"
        result = {"success": True, "message": f"Order {order_id} cancelled successfully"}
    else:
        result = {
            "success": False,
            "message": (
                f"Cannot cancel — status is '{order.status}'. "
                "Only 'processing' orders can be cancelled."
            ),
        }
    log_step("cancel_order", {"order_id": order_id}, result, result["message"])
    return result
