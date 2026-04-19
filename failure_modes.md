# Failure Mode Analysis — ShopWave Support Agent

## Failure 1: Tool Timeout (get_order / get_customer / get_product)
**Probability**: 8% per call (simulated)
**Response**: `@retry_tool(max_attempts=3)` retries with 1s→2s→4s back-off.
On permanent failure returns `{"error": True, "message": "..."}`.
`lookup_node` checks `.get("error")` and continues with partial data.
If both customer and order fail → confidence drops → triage flags `resolvable=False` → escalate.

## Failure 2: Malformed Response from check_refund_eligibility
**Probability**: 10% per call (simulated ValueError)
**Response**: Same retry chain. On permanent failure: `eligible=False`, ticket escalated.
`issue_refund` is NEVER called unless `eligible=True` — irreversible action guard enforced in code.

## Failure 3: Gemini 429 Rate Limit
**Response**: `BATCH_SIZE=3` + `BATCH_DELAY=15s` limits concurrent calls.
`asyncio.gather(return_exceptions=True)` catches failures per-ticket without killing the batch.
`audit_log.json` saved after EVERY batch — progress never lost on crash.
Failed tickets stored as dead-letter entries (not silently dropped).

## Failure 4: Double Refund Attempt
**Response**: `issue_refund()` checks `order.refund_status == "refunded"` and returns
`{"success": False, "message": "Already refunded"}` — never double-refunds.
Schema validation (amount > 0, isinstance float) runs before the call.

## Failure 5: Missing Order ID in Ticket
**Response**: `ingest_node` uses regex to extract ORD-XXXX.
If none found, `lookup_node` calls `get_orders_by_customer()` to fetch all orders.
Most recent order used as primary. Still counts as tool call 2 → ≥3 tool calls guaranteed.
