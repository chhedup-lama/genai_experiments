"""
README (single file FastAPI + OpenAI demo)

Install:
  python -m venv .venv
  # Mac/Linux: source .venv/bin/activate
  # Windows: .venv\Scripts\activate
  pip install -r requirements.txt

Run:
  uvicorn main:app --reload

Browser test:
  Visit http://localhost:8000/health to confirm service health.

Postman test:
  Send POST http://localhost:8000/chat with JSON body:
  {
    "session_id": "demo-user-1",
    "message": "I need running shoes under $120"
  }
Optional:
  Set OPENAI_MOCK_MODE=1 for offline testing (skips live OpenAI calls).
"""

from __future__ import annotations

import os
import sys
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import typing
from dotenv import load_dotenv

load_dotenv()

if sys.version_info >= (3, 13):
    _forward_ref_evaluate = typing.ForwardRef._evaluate

    def _compat_forward_ref_evaluate(self, globalns=None, localns=None, recursive_guard=None, /, *args, **kwargs):
        if args:
            if recursive_guard is None:
                recursive_guard = args[0]
                args = args[1:]
        guard = recursive_guard if recursive_guard is not None else set()
        if not isinstance(guard, set):
            guard = set(guard)
        kwargs["recursive_guard"] = guard
        return _forward_ref_evaluate(self, globalns, localns, *args, **kwargs)

    typing.ForwardRef._evaluate = _compat_forward_ref_evaluate

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI, APIError, AuthenticationError, OpenAIError

# -----------------------------
# Config
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required. Set it in .env or your shell before starting the server.")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change if you want
MOCK_OPENAI = os.getenv("OPENAI_MOCK_MODE", "0").lower() in ("1", "true", "yes")
if OPENAI_API_KEY in {"sk-your-key-here", "mock", "test-key"}:
    MOCK_OPENAI = True

client: Optional[OpenAI] = None
if not MOCK_OPENAI:
    client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Digital Commerce Multi-Agent (FastAPI + OpenAI)")


def raise_openai_http_error(context: str, exc: OpenAIError) -> None:
    """Convert OpenAI SDK errors into FastAPI HTTP responses."""
    if isinstance(exc, AuthenticationError):
        detail = f"OpenAI authentication failed while {context}. Verify OPENAI_API_KEY."
        raise HTTPException(status_code=401, detail=detail) from exc
    if isinstance(exc, APIError):
        detail = f"OpenAI API error while {context}: {exc}"
        raise HTTPException(status_code=502, detail=detail) from exc
    detail = f"Unexpected OpenAI client error while {context}: {exc}"
    raise HTTPException(status_code=500, detail=detail) from exc


def mock_master_route(session_id: str, user_message: str) -> Dict[str, Any]:
    text = user_message.lower()
    if any(k in text for k in ["checkout", "buy", "shipping", "ship", "cart", "total"]):
        agent = "CartCheckoutAgent"
    elif any(k in text for k in ["order", "refund", "return", "status"]):
        agent = "OrderSupportAgent"
    elif any(k in text for k in ["promo", "marketing", "campaign", "discount", "offer"]):
        agent = "MarketingAgent"
    else:
        agent = "CatalogAgent"
    return {"agent": agent, "reason": "Mock routing heuristic", "must_ask": []}


def mock_agent_reply(agent: str, req: ChatRequest, session: Dict[str, Any]) -> str:
    user_text = req.message.strip()
    if agent == "CatalogAgent":
        results = tool_search_products(query=user_text)
        if not results["results"]:
            return "(mock) I couldn't find matching products, but I can widen the search."
        top = results["results"][:3]
        summary = ", ".join(f"{item['name']} ${item['price']}" for item in top)
        return f"(mock CatalogAgent) Here are some options: {summary}."
    if agent == "CartCheckoutAgent":
        cart = tool_view_cart(req.session_id)
        if not cart["items"]:
            return "(mock CartCheckoutAgent) Your cart is empty. Ask me to add something."
        subtotal = money(cart["subtotal"])
        lines = "; ".join(f"{item['name']} x{item['qty']}" for item in cart["items"])
        return f"(mock CartCheckoutAgent) Cart contains {lines} for {subtotal}."
    if agent == "OrderSupportAgent":
        return "(mock OrderSupportAgent) I can help with order status or returns once you place an order."
    if agent == "MarketingAgent":
        return "(mock MarketingAgent) Here's a friendly promo: Save 10% on accessories this week!"
    return f"(mock {agent}) I'm ready to help with: {user_text}"

# -----------------------------
# In-memory "database"
# -----------------------------
PRODUCTS = [
    {
        "id": "sku_1001",
        "name": "AeroRun Lite Running Shoes",
        "category": "shoes",
        "price": 89.99,
        "tags": ["running", "lightweight", "men", "women"],
        "rating": 4.5,
        "inventory": 24,
    },
    {
        "id": "sku_1002",
        "name": "TrailBlaze Grip Runner",
        "category": "shoes",
        "price": 119.00,
        "tags": ["running", "trail", "grip"],
        "rating": 4.6,
        "inventory": 10,
    },
    {
        "id": "sku_1003",
        "name": "CityWalk Comfort Sneakers",
        "category": "shoes",
        "price": 74.50,
        "tags": ["casual", "comfort"],
        "rating": 4.2,
        "inventory": 30,
    },
    {
        "id": "sku_2001",
        "name": "Everyday Cotton T-Shirt",
        "category": "apparel",
        "price": 19.99,
        "tags": ["tshirt", "cotton", "basics"],
        "rating": 4.3,
        "inventory": 120,
    },
    {
        "id": "sku_3001",
        "name": "HydraSteel Water Bottle 1L",
        "category": "accessories",
        "price": 24.95,
        "tags": ["bottle", "steel", "fitness"],
        "rating": 4.7,
        "inventory": 55,
    },
]

# session state: carts, orders, user prefs
SESSIONS: Dict[str, Dict[str, Any]] = {}
ORDERS: Dict[str, Dict[str, Any]] = {}

# -----------------------------
# Helpers
# -----------------------------
def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "history": [],  # chat history for master + child interactions
            "cart": [],  # list[{sku, qty}]
            "user_profile": {
                "locale": "en-IN",
                "currency": "INR",
                "preferences": {},
            },
        }
    return SESSIONS[session_id]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def money(amount: float, currency: str = "USD") -> str:
    # simple formatting; replace with real currency rules
    symbol = "$" if currency == "USD" else "₹" if currency == "INR" else f"{currency} "
    return f"{symbol}{amount:,.2f}"


# -----------------------------
# Business functions (tools)
# -----------------------------
def tool_search_products(query: str = "", category: Optional[str] = None, max_price: Optional[float] = None) -> Dict[str, Any]:
    q = (query or "").lower().strip()
    results = []
    for p in PRODUCTS:
        if category and p["category"] != category:
            continue
        if max_price is not None and p["price"] > max_price:
            continue
        hay = " ".join([p["name"], " ".join(p["tags"]), p["category"]]).lower()
        if q and q not in hay:
            continue
        results.append(p)
    # sort by rating desc then price asc
    results.sort(key=lambda x: (-x["rating"], x["price"]))
    return {"results": results[:10], "count": len(results)}


def tool_get_product(product_id: str) -> Dict[str, Any]:
    for p in PRODUCTS:
        if p["id"] == product_id:
            return {"product": p}
    return {"error": "NOT_FOUND", "message": f"Unknown product_id: {product_id}"}


def tool_add_to_cart(session_id: str, product_id: str, quantity: int = 1) -> Dict[str, Any]:
    sess = get_session(session_id)
    if quantity <= 0:
        return {"error": "INVALID_QUANTITY"}
    prod = next((p for p in PRODUCTS if p["id"] == product_id), None)
    if not prod:
        return {"error": "NOT_FOUND"}
    if prod["inventory"] < quantity:
        return {"error": "OUT_OF_STOCK", "available": prod["inventory"]}
    # add/merge
    cart = sess["cart"]
    for item in cart:
        if item["sku"] == product_id:
            item["qty"] += quantity
            return {"cart": cart}
    cart.append({"sku": product_id, "qty": quantity})
    return {"cart": cart}


def tool_view_cart(session_id: str) -> Dict[str, Any]:
    sess = get_session(session_id)
    cart = []
    total = 0.0
    for item in sess["cart"]:
        prod = next((p for p in PRODUCTS if p["id"] == item["sku"]), None)
        if not prod:
            continue
        line = prod["price"] * item["qty"]
        total += line
        cart.append(
            {
                "sku": prod["id"],
                "name": prod["name"],
                "price": prod["price"],
                "qty": item["qty"],
                "line_total": line,
            }
        )
    return {"items": cart, "subtotal": round(total, 2)}


def tool_calculate_shipping(country: str, postal_code: str, subtotal: float) -> Dict[str, Any]:
    # toy logic: free shipping over $100, otherwise flat
    if subtotal >= 100:
        return {"shipping": 0.0, "eta_days": 3, "method": "Standard"}
    return {"shipping": 7.99, "eta_days": 5, "method": "Standard"}


def tool_place_order(session_id: str, country: str, postal_code: str, payment_method: str = "cod") -> Dict[str, Any]:
    sess = get_session(session_id)
    cart_view = tool_view_cart(session_id)
    if not cart_view["items"]:
        return {"error": "EMPTY_CART"}
    shipping = tool_calculate_shipping(country, postal_code, cart_view["subtotal"])
    order_total = round(cart_view["subtotal"] + shipping["shipping"], 2)

    # decrement inventory
    for item in sess["cart"]:
        prod = next((p for p in PRODUCTS if p["id"] == item["sku"]), None)
        if prod and prod["inventory"] >= item["qty"]:
            prod["inventory"] -= item["qty"]
        else:
            return {"error": "INVENTORY_CHANGED", "sku": item["sku"]}

    order_id = f"ord_{uuid.uuid4().hex[:10]}"
    order = {
        "order_id": order_id,
        "created_at": now_iso(),
        "items": cart_view["items"],
        "subtotal": cart_view["subtotal"],
        "shipping": shipping,
        "total": order_total,
        "ship_to": {"country": country, "postal_code": postal_code},
        "payment_method": payment_method,
        "status": "CONFIRMED",
    }
    ORDERS[order_id] = order
    sess["cart"] = []  # clear cart
    return {"order": order}


def tool_get_order_status(order_id: str) -> Dict[str, Any]:
    order = ORDERS.get(order_id)
    if not order:
        return {"error": "NOT_FOUND"}
    return {"order_id": order_id, "status": order["status"], "created_at": order["created_at"], "total": order["total"]}


def tool_initiate_return(order_id: str, reason: str) -> Dict[str, Any]:
    order = ORDERS.get(order_id)
    if not order:
        return {"error": "NOT_FOUND"}
    if order["status"] in ("RETURN_REQUESTED", "RETURNED", "REFUNDED"):
        return {"error": "ALREADY_IN_PROGRESS", "status": order["status"]}
    order["status"] = "RETURN_REQUESTED"
    order["return_reason"] = reason
    order["return_requested_at"] = now_iso()
    return {"ok": True, "order_id": order_id, "status": order["status"]}


def tool_set_user_preference(session_id: str, key: str, value: str) -> Dict[str, Any]:
    sess = get_session(session_id)
    sess["user_profile"]["preferences"][key] = value
    return {"ok": True, "preferences": sess["user_profile"]["preferences"]}


def tool_get_user_profile(session_id: str) -> Dict[str, Any]:
    sess = get_session(session_id)
    return {"user_profile": sess["user_profile"]}


# -----------------------------
# Tool schema for OpenAI function calling
# -----------------------------
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search products by keyword and optional filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string", "nullable": True},
                    "max_price": {"type": "number", "nullable": True},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Get a single product by product_id (sku).",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "Add an item to the user's cart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "product_id": {"type": "string"},
                    "quantity": {"type": "integer", "minimum": 1},
                },
                "required": ["session_id", "product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_cart",
            "description": "View the user's cart with totals.",
            "parameters": {
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_shipping",
            "description": "Calculate shipping cost and ETA.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string"},
                    "postal_code": {"type": "string"},
                    "subtotal": {"type": "number"},
                },
                "required": ["country", "postal_code", "subtotal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Place an order using the user's cart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "country": {"type": "string"},
                    "postal_code": {"type": "string"},
                    "payment_method": {"type": "string"},
                },
                "required": ["session_id", "country", "postal_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get order status by order_id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "initiate_return",
            "description": "Start a return for an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["order_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_user_preference",
            "description": "Set a user preference (e.g., 'shoe_size'='9', 'favorite_color'='black').",
            "parameters": {
                "type": "object",
                "properties": {"session_id": {"type": "string"}, "key": {"type": "string"}, "value": {"type": "string"}},
                "required": ["session_id", "key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": "Fetch user profile and saved preferences.",
            "parameters": {
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "search_products": lambda args: tool_search_products(**args),
    "get_product": lambda args: tool_get_product(**args),
    "add_to_cart": lambda args: tool_add_to_cart(**args),
    "view_cart": lambda args: tool_view_cart(**args),
    "calculate_shipping": lambda args: tool_calculate_shipping(**args),
    "place_order": lambda args: tool_place_order(**args),
    "get_order_status": lambda args: tool_get_order_status(**args),
    "initiate_return": lambda args: tool_initiate_return(**args),
    "set_user_preference": lambda args: tool_set_user_preference(**args),
    "get_user_profile": lambda args: tool_get_user_profile(**args),
}


# -----------------------------
# Prompts (5 agents)
# -----------------------------
MASTER_PROMPT = """You are the Master Orchestrator for a general digital commerce app.
Your job:
1) Decide which child agent should handle the user's request.
2) Enforce safety + correctness: never invent order IDs, prices, or inventory.
3) If a tool call is needed, let the chosen child agent do it (do not call tools yourself).

Choose exactly one agent:
- CatalogAgent: product search, comparisons, recommendations, FAQs about products.
- CartCheckoutAgent: add/view cart, shipping, checkout, place order, payment guidance.
- OrderSupportAgent: order status, returns, refunds, delivery issues.
- MarketingAgent: promotions, personalized offers, re-engagement messaging, ad/email copy.

Return ONLY valid JSON like:
{"agent":"CatalogAgent","reason":"...","must_ask": ["optional question 1", "..."]}

Rules:
- If the user wants to buy / checkout / shipping cost -> CartCheckoutAgent.
- If user asks where is my order / return / refund -> OrderSupportAgent.
- If user asks discounts, campaign copy, retention, personalization -> MarketingAgent.
- Otherwise -> CatalogAgent.
"""

CATALOG_PROMPT = """You are CatalogAgent for a commerce app.
You help users discover products and decide what to buy.

You can:
- search_products(query, category, max_price)
- get_product(product_id)
- set_user_preference(session_id, key, value)
- get_user_profile(session_id)

Guidelines:
- Ask 1-2 clarifying questions only if necessary (size, budget, use-case).
- Prefer 3 options with short pros/cons and a clear recommendation.
- Never claim inventory is available unless tool data shows it.
"""

CART_CHECKOUT_PROMPT = """You are CartCheckoutAgent.
You help with cart, shipping, totals, and placing orders.

You can:
- add_to_cart(session_id, product_id, quantity)
- view_cart(session_id)
- calculate_shipping(country, postal_code, subtotal)
- place_order(session_id, country, postal_code, payment_method)

Guidelines:
- If user wants shipping cost, request country + postal code.
- Before placing order, summarize items + subtotal + shipping + total and confirm required inputs exist.
- Never fabricate totals; always compute via tools.
"""

ORDER_SUPPORT_PROMPT = """You are OrderSupportAgent.
You help with order status, delivery issues, returns, refunds.

You can:
- get_order_status(order_id)
- initiate_return(order_id, reason)

Guidelines:
- Ask for order_id if missing.
- Be concise and action-oriented.
- Never invent order details.
"""

MARKETING_PROMPT = """You are MarketingAgent.
You create promotions, personalized recommendations messaging, and copy.

You can:
- get_user_profile(session_id)
- set_user_preference(session_id, key, value)
- search_products(query, category, max_price)

Guidelines:
- Tailor offers to user's stated intent and preferences.
- Provide 2-3 variants of copy if asked (short, friendly, premium).
- Avoid spammy language; be clear about terms (no fake discounts).
"""

AGENT_PROMPTS = {
    "CatalogAgent": CATALOG_PROMPT,
    "CartCheckoutAgent": CART_CHECKOUT_PROMPT,
    "OrderSupportAgent": ORDER_SUPPORT_PROMPT,
    "MarketingAgent": MARKETING_PROMPT,
}

# -----------------------------
# OpenAI wrappers (Chat Completions + tool loop)
# -----------------------------
def chat_with_tools(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    max_tool_rounds: int = 6,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Runs a tool-calling loop using Chat Completions.
    Returns (final_text, updated_messages).
    """
    if client is None:
        raise RuntimeError("OpenAI client is not initialized.")
    tools = tools or []
    for _ in range(max_tool_rounds):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools if tools else None,
            tool_choice=tool_choice if tool_choice else None,
        )
        msg = resp.choices[0].message
        messages.append(
            {
                "role": msg.role,
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])],
            }
        )

        # If no tool calls, we're done
        if not msg.tool_calls:
            return (msg.content or "").strip(), messages

        # Execute tool calls
        for tc in msg.tool_calls:
            fn = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            if fn not in TOOL_DISPATCH:
                tool_result = {"error": "UNKNOWN_TOOL", "tool": fn}
            else:
                try:
                    tool_result = TOOL_DISPATCH[fn](args)
                except Exception as e:
                    tool_result = {"error": "TOOL_EXEC_FAILED", "tool": fn, "message": str(e)}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn,
                    "content": json.dumps(tool_result),
                }
            )

    # If we hit max rounds, force a graceful exit
    messages.append(
        {
            "role": "assistant",
            "content": "I’m having trouble completing that in one go. Please retry or narrow the request.",
        }
    )
    return "I’m having trouble completing that in one go. Please retry or narrow the request.", messages


def master_route(session_id: str, user_message: str) -> Dict[str, Any]:
    """
    Master picks the agent. Uses JSON-only output (no tools).
    """
    if MOCK_OPENAI:
        return mock_master_route(session_id, user_message)
    if client is None:
        raise RuntimeError("OpenAI client is not initialized.")
    messages = [
        {"role": "system", "content": MASTER_PROMPT},
        {"role": "user", "content": user_message},
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"agent": "CatalogAgent", "reason": "Fallback routing due to invalid JSON from master.", "must_ask": []}

    agent = data.get("agent", "CatalogAgent")
    if agent not in AGENT_PROMPTS:
        agent = "CatalogAgent"
    data["agent"] = agent
    return data


# -----------------------------
# API models
# -----------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-provided session id")
    message: str = Field(..., description="User message")


class ChatResponse(BaseModel):
    session_id: str
    agent: str
    text: str
    route_reason: str
    must_ask: List[str] = []


# -----------------------------
# FastAPI endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "time": now_iso()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sess = get_session(req.session_id)

    # 1) Master routes
    if MOCK_OPENAI:
        route = mock_master_route(req.session_id, req.message)
    else:
        try:
            route = master_route(req.session_id, req.message)
        except OpenAIError as exc:
            raise_openai_http_error("routing conversation", exc)
    agent = route["agent"]
    route_reason = route.get("reason", "")
    must_ask = route.get("must_ask", []) or []

    # 2) Child agent handles with tool loop
    child_prompt = AGENT_PROMPTS[agent]

    # Keep a lightweight per-session conversation (optional)
    # Store only the last ~12 turns to avoid runaway context.
    history = sess["history"][-12:]

    messages: List[Dict[str, Any]] = []
    messages.append({"role": "system", "content": child_prompt})
    messages.append(
        {
            "role": "system",
            "content": f"Session context:\n- session_id: {req.session_id}\n- cart: {json.dumps(sess['cart'])}\n- time_utc: {now_iso()}",
        }
    )
    # include some recent history
    messages.extend(history)
    messages.append({"role": "user", "content": req.message})

    if MOCK_OPENAI:
        final_text = mock_agent_reply(agent, req, sess)
        updated_messages = history + [
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": final_text},
        ]
    else:
        try:
            final_text, updated_messages = chat_with_tools(messages, tools=TOOLS)
        except OpenAIError as exc:
            raise_openai_http_error(f"running {agent}", exc)

    # Update stored history (strip tool blobs to keep it simple)
    # Keep: user + assistant messages only (ignore tool role entries)
    compact: List[Dict[str, Any]] = []
    for m in updated_messages:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            compact.append({"role": m["role"], "content": m["content"]})
    sess["history"] = compact[-12:]

    return ChatResponse(
        session_id=req.session_id,
        agent=agent,
        text=final_text,
        route_reason=route_reason,
        must_ask=must_ask,
    )


# Optional: quick endpoints for debugging state
@app.get("/debug/session/{session_id}")
def debug_session(session_id: str):
    sess = get_session(session_id)
    return {"session_id": session_id, "cart": sess["cart"], "user_profile": sess["user_profile"], "history": sess["history"]}


@app.get("/debug/orders")
def debug_orders():
    return {"orders": ORDERS}
