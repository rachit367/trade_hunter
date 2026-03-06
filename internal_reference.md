# Delta Exchange API Documentation

## Overview

Welcome to the Delta Exchange API Reference. This documentation provides exhaustive guides and reference materials for building stable, production-ready trading bots and clients using our REST API and Websocket feeds.

Delta Exchange provides distinct environments for testing and production:
- **Production URL:** `https://api.india.delta.exchange`
- **Testnet/Demo URL:** `https://cdn-ind.testnet.deltaex.org`

---

## 1. Core Architecture & Bot Design Patterns

When building a high-frequency trading bot or a long-running client, adhering to specific architectural patterns is essential to avoid rate-limiting, missing price movements, and getting banned.

### Recommended System Architecture

1.  **Transport Layer (Websockets for Data, REST for Action):**
    *   *Public Data:* Connect to Websockets (`wss://socket.india.delta.exchange`) for L2 Orderbook, Mark Price, and Trades. *Never* poll the REST API (`GET /v2/l2orderbook`) continuously.
    *   *Private Data:* Subscribe to user-specific websocket channels (orders, positions, fills, margins) to keep local state updated.
    *   *Actions:* Use the REST API (`POST /v2/orders`) to execute trades and manage positions.
2.  **Deadman Switch (Heartbeat):**
    *   Implement `/v2/heartbeat` in a dedicated background thread. Set up `cancel_orders` as the action if your client disconnects (unhealthy count).
    *   *Why?* If your bot crashes or network fails, this prevents you from holding active limit orders in a volatile market.
3.  **Local State Management:**
    *   Maintain a local copy of your balances and positions updated via WS. Polling `GET /v2/wallet/balances` over the REST API consumes heavy rate limits and introduces latency.
4.  **Rate Limit Handler:**
    *   REST limits: 10,000 weight per 5 minutes.
    *   Build a dispatcher that catches HTTP 429 (`Too Many Requests`) and specifically reads the `X-RATE-LIMIT-RESET` response header to sleep dynamically rather than hardcoding.
5.  **Market Maker Protection (MMP):**
    *   If providing liquidity, configure MMP (`PUT /v2/users/update_mmp`) to auto-freeze your trading if your fills exceed a specified size/delta/vega within a time window. It acts as an emergency brake.

---

## 3. Current Status: IP Whitelist Required
Your recent verification run confirms that the **Signature Mismatch is FIXED**. However, a new security error was triggered:
`401 Unauthorized: ip_not_whitelisted_for_api_key`

### How to Fix:
1.  **Identify IP**: Your current IP is `157.48.86.71`.
2.  **Update Delta Settings**:
    - Go to [india.delta.exchange](https://india.delta.exchange) -> **API Keys**.
    - Find your key `A2mCz...` and click **Edit**.
    - Add `157.48.86.71` to the **Restricted IP** list or set it to **No Restriction** (if you are on a dynamic IP).

## 4. Verification Script
I created a standalone verification script [verify_auth.py](file:///c:/Users/rachi/OneDrive/Desktop/Projects/trade_hunter/verify_auth.py) to test the fix independently.

## 2. Authentication & Security

All private endpoints require API keys generated from the Delta Exchange dashboard. Note that keys generated on the main global site (`api.delta.exchange`) will **not** work on the India API domain.

### Signature Generation Algorithm

Requests are authenticated using an HMAC SHA256 signature generated from the request payload.
Signatures expire after 5 seconds, necessitating rigorous NTP clock synchronization.

**Signature Structure String:**
`METHOD + TIMESTAMP + PATH + QUERY_STRING + PAYLOAD`

### ✅ Python Implementation

```python
import time
import hmac
import hashlib
import requests
import json
from urllib.parse import urlencode

API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
BASE_URL = "https://api.india.delta.exchange"

def generate_signature(method, timestamp, path, query_string, payload):
    # Construct the data string exactly as required
    signature_data = method.upper() + timestamp + path + query_string + payload
    
    # Generate the HMAC SHA256 signature
    signature = hmac.new(
        API_SECRET.encode("utf-8"),
        signature_data.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    
    return signature

def delta_request(method, endpoint, params=None, payload=None):
    timestamp = str(int(time.time()))
    path = f"/v2{endpoint}"
    
    # Properly format query string
    query_string = ""
    if params:
        query_string = "?" + urlencode(params)
        
    # Payload string (empty string for GET, literal empty string if no body)
    payload_str = json.dumps(payload, separators=(',', ':')) if payload else ""
    
    signature = generate_signature(method, timestamp, path, query_string, payload_str)
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "api-key": API_KEY,
        "signature": signature,
        "timestamp": timestamp,
        "User-Agent": "python-delta-client/1.0" # REQUIRED to avoid CDN 403 blocks
    }

    url = BASE_URL + path + query_string
    if method.upper() == "GET":
        response = requests.get(url, headers=headers)
    elif method.upper() == "POST":
        response = requests.post(url, headers=headers, data=payload_str)
    elif method.upper() == "DELETE":
        response = requests.delete(url, headers=headers, data=payload_str)
        
    return response.json()

# Example Call:
# print(delta_request("GET", "/wallet/balances"))
```

---

## 3. Exhaustive Endpoints Reference

All API responses follow a strict envelope format:
```json
// Success
{
  "success": true,
  "result": { ... },
  "meta": { "after": "...", "before": null }
}
// Error
{
  "success": false,
  "error": { "code": "insufficient_margin", "context": { "additional_margin_required": "0.121" } }
}
```

### 3.1 Public Endpoints

#### Get Products
Retrieves trading products.
*   **Method:** `GET /v2/products`
*   **Parameters (Query):**
    *   `contract_types`: (e.g., `perpetual_futures,call_options`)
    *   `states`: (e.g., `live,expired`)
*   **Common Response Fields:** `id` (product_id), `symbol`, `underlying_asset`, `impact_size`.

#### Get Tickers
Live prices and 24h stats.
*   **Method:** `GET /v2/tickers`
*   **Parameters (Query):** `contract_types`, `underlying_asset_symbols`
*   **Notable Metric:** Returns `greeks` (delta, vega, gamma, theta) for options.

#### Get L2 Orderbook
*   **Method:** `GET /v2/l2orderbook/{symbol}`
*   **Parameters (Path):** `symbol` (e.g., `BTCUSD`)
*   **Parameters (Query):** `depth`
*   *Stripe-level Tip:* Do not poll this. Use websockets.

### 3.2 Trading & Execution (Private)

#### Place Order
*   **Method:** `POST /v2/orders`
*   **Weight:** 5
*   **Payload Schema:**
```json
{
  "product_id": 27,                 // Integer Required
  "size": 10,                       // Integer Required (Contracts)
  "side": "buy",                    // "buy" | "sell" Required
  "order_type": "limit_order",      // "limit_order" | "market_order"
  "limit_price": "59000",           // String Required for limit_order
  "time_in_force": "gtc",           // "gtc" | "ioc"
  "post_only": "false",             // "true" | "false" String
  "reduce_only": "false",           // "true" | "false" String
  "client_order_id": "sig_001"      // Max 32 chars
}
```

#### Place Bracket Order (TP/SL)
*   **Method:** `POST /v2/orders/bracket`
*   **Details:** Attach Take Profit / Stop Loss to an entirely specific product.
*   **Payload Schema:**
```json
{
  "product_id": 27,
  "stop_loss_order": {
    "order_type": "limit_order",
    "stop_price": "56000",
    "limit_price": "55000"
  },
  "take_profit_order": {
    "order_type": "limit_order",
    "stop_price": "65000",
    "limit_price": "64000"
  },
  "bracket_stop_trigger_method": "last_traded_price" // mark_price | last_traded_price
}
```

#### Batch Create Orders
*   **Method:** `POST /v2/orders/batch`
*   **Weight:** 25
*   **Payload Schema:**
```json
{
  "product_id": 27,
  "orders": [
    { "size": 10, "side": "buy", "order_type": "limit_order", "limit_price": "58000" },
    { "size": 10, "side": "buy", "order_type": "limit_order", "limit_price": "57500" }
  ]
}
```

#### Cancel Order
*   **Method:** `DELETE /v2/orders`
*   **Weight:** 5
*   **Payload:** `{ "id": 13452112, "product_id": 27 }`

### 3.3 Account & Positional Data (Private)

#### Get Wallet Balances
*   **Method:** `GET /v2/wallet/balances`
*   **Weight:** 3

#### Get Margined Positions
*   **Method:** `GET /v2/positions/margined`
*   **Weight:** 3
*   **Performance Note:** Has a 10s delay. For real-time state, use `GET /v2/positions` or websockets.

#### Close All Positions
*   **Method:** `POST /v2/positions/close_all`
*   **Payload:** `{ "close_all_portfolio": true, "close_all_isolated": true, "user_id": 12345 }`

---

## 4. Websocket Feeds

Websockets are strictly required for building a scalable bot.
- **Production URL:** `wss://socket.india.delta.exchange`
- **Connection Limit:** 150 connections per 5 minutes per IP. Unused connections disconnect after 60 seconds.

### Public Channels
- **`l2_orderbook`**: Real-time depth.
- **`v2/ticker`**: Live price, mark price, 24h volume.
- **`mark_price`**: Crucial for funding calculations and liquidation monitoring.

### Private Channels (Requires Authentication Message)
- **`orders`**: Stream of open/filled/cancelled statuses.
- **`positions`**: Real-time entry prices and liquidation levels.

---

## 5. Comprehensive Error Troubleshooting

Handling errors gracefully distinguishes a script from a production bot.

### ❌ `SignatureExpired` (401)
*   **Payload:** `{ "error": "SignatureExpired", "message": "your signature has expired" }`
*   **Root Cause:** The timestamp in your headers is > 5 seconds older than the Delta server time.
*   **Resolution:**
    *   Sync your OS clock via NTP: `sudo ntpdate pool.ntp.org` (Linux) or run Windows Time Sync.
    *   Do not cache timestamps. Regenerate `str(int(time.time()))` precisely before the request.

### ❌ `Forbidden` (403)
*   **Payload:** `{ "error": "Forbidden", "message": "Request blocked by CDN" }`
*   **Root Cause:** Cloudflare/CDN rejected your request before it hit Delta.
*   **Resolution:** You **MUST** include a `User-Agent` header (e.g., `"User-Agent": "TradeBot/1.0"`). Standard `requests` library without a custom user agent is frequently blocked.

### ❌ `ip_not_whitelisted_for_api_key` (401)
*   **Payload:** `{ "error": { "code": "ip_not_whitelisted_for_api_key", "context": { "client_ip": "..." } } }`
*   **Root Cause:** The API key has an IP Whitelist enabled, but your current server/local IP is not on that list.
*   **Resolution:**
    1.  Log in to [india.delta.exchange](https://india.delta.exchange).
    2.  Go to **API Keys**.
    3.  Either **Disable IP Whitelisting** for that key (not recommended for production) OR **Add the Client IP** shown in the error message (`157.48.86.71`) to the whitelist.

### ❌ `Signature Mismatch` (401)
*   **Payload:** `{ "success": false, "error": { "code": "Signature Mismatch" } }`
*   **Root Cause:** The payload you hashed locally does not byte-match what Delta received.
*   **Resolution:**
    *   Ensure NO spaces in JSON compilation: In Python, strictly use `json.dumps(payload, separators=(',', ':'))`.
    *   Check appending `''` (empty string) for `query_string` when no parameters exist.

### ❌ `InvalidApiKey` / `UnauthorizedApiAccess` (401)
*   **Root Cause:** Key doesn't exist, lacks specific permissions (e.g. Trading vs Read Data), or cross-environment usage.
*   **Resolution:** Ensure testnet credentials aren't going to the `api.india.delta.exchange` domain, and that Trading checkboxes are ticked.

### ❌ Execution Errors (Code: 400)
*   `insufficient_margin`: Wallet lacks funds for initial margin of the order.
*   `order_size_exceed_available`: Market/IOC order lacks liquidity in the orderbook to fill completely.
*   `risk_limits_breached`: Exceeds account global risk parameters.
*   `immediate_execution_post_only`: Your limit order had `post_only=true`, but crossed the book and would execute as a taker. Instead, the match engine rejects it.

---

## 6. Checklist for Production Deployment

1. [ ] **Decimal Strings:** Are you casting all prices (limit, stop) and sizes to strings natively before hashing? No floats should pass through.
2. [ ] **NTP Sync Daemon:** Is the server running your bot running a cronjob to sync Network Time?
3. [ ] **User-Agent Header:** Is `User-Agent` explicitly set on every request?
4. [ ] **429 Backoff:** Does your network wrapper catch HTTP 429 and sleep according to `X-RATE-LIMIT-RESET`?
5. [ ] **Heartbeats:** Did you implement `/v2/heartbeat` in a daemon thread so open orders auto-cancel if your bot crashes?
