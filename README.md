

# ShopWave — Autonomous Support Resolution Agent

**Agentic AI Hackathon 2026**  
*"Don't write code. Engineer Reasoning."*

An autonomous support agent built with LangGraph that ingests 20 simulated tickets, classifies them, retrieves policies, triages confidence, executes actions (refund, cancel, etc.), and escalates when uncertain — all with full auditability.

---

## 🚀 Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/your-username/hackathon2026-yourname.git
cd hackathon2026-yourname
pip install -r requirements.txt


### 2. Set Up API Key
```bash
cp .env.example .env
# Open .env and add your Gemini API key
# Get a free key at: https://aistudio.google.com/apikey
```

### 3. Run the Agent
```bash
python main.py
```

The script will process all tickets in concurrent batches, generate audit logs, and print a summary to the console.

---

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point — loads data, runs batches, saves logs. |
| `agent.py` | Wraps a single ticket through the LangGraph pipeline. |
| `graph.py` | Defines the LangGraph state graph (nodes + edges). |
| `nodes.py` | Node implementations: ingest, lookup, classifier, kb_search, triage, act, reply. |
| `state.py` | TypedDict `AgentState` defining the agent's working memory. |
| `tools.py` | All mock tools (read/write) with retry & failure simulation. |
| `llm_client.py` | Gemini LLM client via OpenAI-compatible endpoint. |
| `utils.py` | Audit logging (`log_step`), retry decorator, and `save_audit_log`. |
| `models.py` | Pydantic models for Customer, Order, Product. |
| `data/` | Mock data: `tickets.json`, `customers.json`, `orders.json`, `products.json`, `knowledge_base.md`. |

---

## 🧠 Architecture

The agent is a **LangGraph StateGraph** with a linear pipeline:

```
START → ingest → lookup → classifier → kb_search → triage → act → reply → END
```

### Node Responsibilities

| Node | What It Does |
|------|--------------|
| **ingest** | Deterministically extracts order IDs from the ticket body (regex). |
| **lookup** | Makes **≥3 tool calls**: `get_customer` → `get_order` (or `get_orders_by_customer`) → `get_product`. Handles failures gracefully. |
| **classifier** | Classifies the ticket (`refund`, `cancellation`, `order_status`, `warranty`, `escalate`, `general`) **with full context** (customer tier, order status, product details). |
| **kb_search** | Fetches relevant policy sections from `knowledge_base.md` based on classification. |
| **triage** | Computes a **confidence score** from 8 objective data signals. Decides urgency (`low`/`medium`/`high`/`urgent`) and whether the agent can resolve autonomously (`resolvable`). |
| **act** | Executes the appropriate action: issue refund, cancel order, provide order status, escalate warranty, answer general inquiry, or escalate. All write operations happen here. |
| **reply** | Sends the final response to the customer via `send_reply`. **Always runs**—every ticket gets a reply. |

---

## 🛠️ Tools

### Read / Lookup
- `get_customer(email)`
- `get_order(order_id)`
- `get_product(product_id)`
- `search_knowledge_base(query)`

### Write / Act
- `check_refund_eligibility(order_id)`
- `issue_refund(order_id, amount)`
- `send_reply(ticket_id, message)`
- `escalate(ticket_id, summary, priority)`

### Internal Helpers
- `get_orders_by_customer(customer_id)`
- `cancel_order(order_id)`

All tools are wrapped with `@retry_tool` (exponential backoff, max 3 attempts) to satisfy the **Recover** constraint. They never raise exceptions; they return structured error dicts.

---

## ✅ Hackathon Constraints Met

| Requirement | Implementation |
|-------------|----------------|
| **≥3 tool calls per chain** | `lookup` always performs 3+ calls. |
| **Concurrent processing** | `asyncio.gather` processes tickets in configurable batch sizes. |
| **Graceful tool failure** | `@retry_tool` + `_safe` wrapper; failures logged, agent continues. |
| **Explainable decisions** | Every step logged with `log_step`; `audit_trail` list + `confidence_signals` dict. |
| **Escalate intelligently** | `triage` marks non‑resolvable cases; `act` escalates with structured summary. |
| **Dead‑letter queue** | Failed tickets are logged as `error` and included in results (never silently dropped). |

---

## 📄 Output Files (Deliverables)

### 1. `audit_log.json` — Full Decision Trail

**Required by hackathon.** Contains **every** tool call, node transition, LLM inference, and confidence calculation across all tickets.

**Structure:**
```json
[
  {
    "timestamp": "2026-04-19T14:05:28.441Z",
    "step": "get_customer",
    "input": {"email": "alice.turner@email.com"},
    "output": {"name": "Alice Turner", "tier": "vip"},
    "reasoning": "name=Alice Turner tier=vip",
    "confidence": 1.0
  },
  ...
]
```

### 2. `ticket_summaries.json` — Per‑Ticket Readable Summary

**Bonus for human readability.** One clean object per ticket summarizing:
- Classification & action taken
- Escalation status & confidence
- **Every tool invoked** (name, success, summary)
- Resolution preview

**Structure:**
```json
[
  {
    "ticket_id": "TKT-001",
    "classification": "refund",
    "action_taken": "escalated",
    "escalated": true,
    "confidence": 1.0,
    "urgency": "urgent",
    "tool_calls": [
      {"tool": "get_customer", "success": true, "details": "name=Alice Turner tier=vip"},
      {"tool": "get_order", "success": true, "details": "status=delivered"},
      ...
    ],
    "resolution_preview": "Hi Alice, your case has been escalated..."
  }
]
```

---

## 🔧 Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *required* | Your Gemini API key. |
| `GEMINI_FAST_MODEL` | `gemini-2.0-flash-lite` | For classifier (saves quota). |
| `GEMINI_SMART_MODEL` | `gemini-2.0-flash` | For triage, act, reply. |
| `MAX_TICKETS` | `20` | Number of tickets to process. |
| `BATCH_SIZE` | `3` | Concurrent tickets per batch (keep low on free tier). |
| `BATCH_DELAY` | `15` | Seconds between batches (increase if hitting 429 errors). |

---

## 📹 Demo & Deliverables

- ✅ `README.md` — This file.
- ✅ `architecture.png` (or `.pdf`) — Diagram of the agent loop and tool design.
- ✅ `failure_modes.md` — Documented failure scenarios and recovery.
- ✅ `audit_log.json` — Generated after running `python main.py`.
- ✅ `demo.mp4` — Screen recording (max 5 min) showing the agent processing all 20 tickets.

---

## 🧪 Testing & Development

- **Python 3.11+**
- Install dependencies: `pip install -r requirements.txt`
- To process only a few tickets, set `MAX_TICKETS=3` in `.env`.
- Logs are written to the console and saved incrementally to `audit_log.json`.

---

## 👤 Author

**Your Name**  
Agentic AI Hackathon 2026  
*"Stop talking to AI. Start building with AI."*
```
