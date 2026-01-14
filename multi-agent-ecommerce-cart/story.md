# Multi‑Agent E‑Commerce Story (Human‑Friendly Spec)

This document explains, in plain language, how the `multi-agent-ecommerce-cart/main.py` app works.  
Treat this file as the **source of truth for behavior and prompts**.  
In future, you can update this file and then ask the AI assistant to **sync the story into `main.py`**.

---

## 1. Big Picture

- This is a **digital commerce chat API** built with **FastAPI** and **OpenAI**.
- A user talks to a single chat endpoint: `POST /chat`.
- Behind the scenes there is:
  - **One master agent** that decides *which child agent* should respond.
  - **Four child (specialist) agents** that each handle a specific job:
    - Catalog / product discovery
    - Cart + checkout
    - Order support
    - Marketing / promotions
- The system can also call **“tools”** (Python functions) for:
  - Searching products
  - Managing a cart
  - Calculating shipping
  - Placing orders
  - Checking order status and returns
  - Managing user preferences

The actual app is in `main.py`, and this `story.md` describes what it does in non‑technical terms.

---

## 2. Environment & Configuration (for humans)

- The app uses OpenAI via the official `openai` Python SDK.
- Required environment variables (usually set in `.env` at the project root):
  - `OPENAI_API_KEY` – your real OpenAI key (required unless using mock mode).
  - `OPENAI_MODEL` – which model to use, defaults to `gpt-4o-mini`.
  - `OPENAI_MOCK_MODE` – if set to `1`, `true`, or `yes`, the app runs in **mock mode** and does not call OpenAI at all.

If `OPENAI_MOCK_MODE` is on (or if the API key is clearly a dummy like `sk-your-key-here`), the app:
- Pretends to be the agents but uses hard‑coded Python logic instead of real LLM calls.
- This is useful for local demos or offline testing.

---

## 3. In‑Memory “Database”

For simplicity, **everything is stored in memory** (nothing is persisted to a real database):

- `PRODUCTS` – a small list of sample products:
  - Running shoes, sneakers, t‑shirt, water bottle, etc.
  - Each product has: `id`, `name`, `category`, `price`, `tags`, `rating`, `inventory`.

- `SESSIONS` – a dictionary keyed by `session_id`:
  - For each session, it stores:
    - `history`: a short chat history.
    - `cart`: items the user wants to buy (product ID + quantity).
    - `user_profile`: basic preferences, such as locale, currency, and custom keys.

- `ORDERS` – a dictionary of fake orders placed during this run of the app.

If the server restarts, all of this data is lost (since it’s in memory only).

---

## 4. Business Tools (what the agents can “do”)

These are Python functions that represent real business actions.  
The LLM can “call” them via the OpenAI tools / function‑calling mechanism.

**Main tools:**

- `search_products(query, category, max_price)`
  - Looks through `PRODUCTS` and returns up to 10 matching items.
  - Filters by category and maximum price if provided.
  - Sorts by rating (high to low), then by price (low to high).

- `get_product(product_id)`
  - Returns details of a single product (or an error if not found).

- `add_to_cart(session_id, product_id, quantity)`
  - Finds the user session.
  - Ensures the quantity is positive and inventory is available.
  - Adds or increments items in the user’s cart.

- `view_cart(session_id)`
  - Returns the list of cart items and the subtotal.

- `calculate_shipping(country, postal_code, subtotal)`
  - Simple shipping rule:
    - If subtotal ≥ 100 → shipping is free.
    - Else → a flat shipping cost and estimated days.

- `place_order(session_id, country, postal_code, payment_method)`
  - Uses the cart + shipping calculation to create a fake order.
  - Decreases product inventory.
  - Clears the cart.
  - Stores the order in `ORDERS`.

- `get_order_status(order_id)`
  - Returns the status of a previously created order.

- `initiate_return(order_id, reason)`
  - Marks an order as `RETURN_REQUESTED` with a reason and timestamp.

- `set_user_preference(session_id, key, value)`
  - Stores arbitrary preferences (e.g., `shoe_size = 9`, `favorite_color = black`).

- `get_user_profile(session_id)`
  - Returns the user’s profile and preferences.

These functions are exposed to the LLM as “tools” with JSON schemas, so the model can ask to call them.

---

## 5. Agents Overview

There are **5 logical agents** in the story:

1. **Master Orchestrator** (the “boss” agent)
2. **CatalogAgent** (products and discovery)
3. **CartCheckoutAgent** (cart + checkout)
4. **OrderSupportAgent** (orders and returns)
5. **MarketingAgent** (promotions and personalization)

Only the **master** talks directly to the user at routing time.  
Then it decides which child agent should handle the actual reply.

### 5.1 Master Orchestrator

**Role (plain English):**

- Reads the user’s message and decides:
  - Which child agent should respond.
  - Why that agent makes sense.
  - Whether there are any follow‑up questions that must be asked.

**Key behaviors:**

- It must **never invent**:
  - Order IDs
  - Prices
  - Inventory levels
- It **does not call tools itself**.
- It only returns a small JSON object describing routing.

**Possible agents it chooses:**

- `CatalogAgent` – browsing products, comparisons, recommendations.
- `CartCheckoutAgent` – cart actions, shipping, checkout, payment guidance.
- `OrderSupportAgent` – order status, returns, refunds, delivery issues.
- `MarketingAgent` – promotions and marketing copy.

**Routing rules (simplified):**

- If user asks about **buying / checkout / shipping / cart / total** → `CartCheckoutAgent`.
- If user asks about **orders / returns / refunds / status** → `OrderSupportAgent`.
- If user asks about **discounts / promotions / campaigns / offers** → `MarketingAgent`.
- Otherwise (the default) → `CatalogAgent`.

**Return format:**

```json
{"agent":"CatalogAgent","reason":"...","must_ask":["optional question 1","..."]}
```

In real execution, this master agent is implemented by the `MASTER_PROMPT` in `main.py` and called via `master_route(...)`.  
In mock mode, its behavior is replaced by `mock_master_route(...)`.

---

### 5.2 CatalogAgent (product discovery)

**Human‑friendly description:**

- Helps the user **find products** and **decide what to buy**.
- Handles questions like:
  - “I need running shoes under $120”
  - “Show me comfy sneakers for daily wear”

**Tools it can use:**

- `search_products(query, category, max_price)`
- `get_product(product_id)`
- `set_user_preference(session_id, key, value)`
- `get_user_profile(session_id)`

**Behavior guidelines (from prompt):**

- Ask **1–2 clarifying questions only when needed**:
  - e.g., size, budget, intended use.
- Prefer giving **around 3 options** with:
  - Short pros/cons.
  - A clear recommendation at the end.
- Never claim inventory is available unless tool data shows it.

This agent’s exact wording is defined in the `CATALOG_PROMPT` constant in `main.py`.

---

### 5.3 CartCheckoutAgent (cart + checkout)

**Human‑friendly description:**

- Helps with:
  - Adding items to cart.
  - Viewing cart and totals.
  - Calculating shipping.
  - Placing an order.

**Tools it can use:**

- `add_to_cart(session_id, product_id, quantity)`
- `view_cart(session_id)`
- `calculate_shipping(country, postal_code, subtotal)`
- `place_order(session_id, country, postal_code, payment_method)`

**Behavior guidelines (from prompt):**

- If user wants **shipping cost**, it should ask for:
  - Country
  - Postal code
- Before placing an order, it should:
  - Summarize items, subtotal, shipping, and total.
  - Confirm that required inputs exist.
- It must **never fabricate totals**; it has to compute via tools.

This agent’s behavior is defined by `CART_CHECKOUT_PROMPT` in `main.py`.

---

### 5.4 OrderSupportAgent (order status & returns)

**Human‑friendly description:**

- Handles questions about:
  - “Where is my order?”
  - “I want a refund”
  - “I need to return my shoes”

**Tools it can use:**

- `get_order_status(order_id)`
- `initiate_return(order_id, reason)`

**Behavior guidelines (from prompt):**

- Ask for `order_id` when missing.
- Be concise and action‑oriented.
- Never invent order details.

This is implemented by the `ORDER_SUPPORT_PROMPT` string in `main.py`.

---

### 5.5 MarketingAgent (promotions & personalization)

**Human‑friendly description:**

- Focuses on **marketing / promo / copywriting** tasks:
  - Generating promotional messages.
  - Personalizing based on user preferences.
  - Creating ad or email copy variations.

**Tools it can use:**

- `get_user_profile(session_id)`
- `set_user_preference(session_id, key, value)`
- `search_products(query, category, max_price)`

**Behavior guidelines (from prompt):**

- Tailor offers to user’s stated intent and saved preferences.
- Provide **2–3 variants** of copy when asked:
  - e.g., short, friendly, premium tone.
- Avoid spammy language and fake discounts; be clear about terms.

This is defined by the `MARKETING_PROMPT` constant in `main.py`.

---

## 6. How a Typical Request Flows

1. A client sends a `POST /chat` request with:
   - `session_id`
   - `message` (user text)

2. The server:
   - Loads or creates a session (`get_session`).

3. **Master routing:**
   - If mock mode is on → use `mock_master_route`.
   - Otherwise → call OpenAI with `MASTER_PROMPT` via `master_route`.
   - Gets back `{"agent": "...", "reason": "...", "must_ask": [...]}`.

4. **Child agent handling:**
   - Picks the correct prompt from `AGENT_PROMPTS[agent]`.
   - Builds a conversation history:
     - System: child prompt.
     - System: session context (cart, time, etc.).
     - Recent history (last ~12 turns).
     - User: current message.

5. **Tool loop:**
   - In real mode: uses `chat_with_tools(...)` and OpenAI tools (function calling).
   - In mock mode: uses `mock_agent_reply(...)`, which returns a canned (fake) response.

6. **Update history:**
   - Keeps only user + assistant messages (no raw tool payloads).
   - Stores the last ~12 turns to keep context manageable.

7. **Response to client:**
   - Returns a `ChatResponse`:
     - `session_id`
     - `agent` (which child handled it)
     - `text` (final message)
     - `route_reason` (why the master chose that agent)
     - `must_ask` (any required follow‑up questions)

---

## 7. Endpoints Summary

- `GET /health`
  - Simple health check with current UTC time.

- `POST /chat`
  - Main entry point; runs the full routing + agent logic described above.

- `GET /debug/session/{session_id}`
  - Shows current cart, user profile, and history for a session (for debugging).

- `GET /debug/orders`
  - Shows the in‑memory `ORDERS` dictionary.

---

## 8. How to Use This Story for Changes

This `story.md` is written so that future you (or an AI assistant) can:

- Read the **high‑level behavior** without diving into code.
- See what each agent is allowed to do and how it should respond.
- Understand which tools exist and what they mean in business terms.

**Workflow suggestion:**

1. Edit `story.md` in plain language first:
   - For example:
     - Add a new agent (e.g., `LoyaltyAgent`).
     - Change what tools an agent can call.
     - Adjust guidelines (e.g., number of recommendations, tone, etc.).
2. Then tell the AI assistant something like:
   - “Please update `main.py` so that it matches the current `story.md` (especially section X).”
3. The assistant can:
   - Update prompts in `main.py`.
   - Add or remove tools in the `TOOLS` list and `TOOL_DISPATCH`.
   - Add new routes or agents as needed.

**Important note:**  
Right now, the code does **not** automatically read `story.md` at runtime.  
The sync between `story.md` and `main.py` happens when you ask the assistant to apply your story changes to the code.

