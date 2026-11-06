# MEXC Spot Scalper: Technical Plan, Reference Architecture, and Audit Checklist

Status: reference plan, written 2026-11-06 against the MEXC Spot V3 API documentation
(GitHub mirror of the official docs, last changelog entry 2025-02-24, plus the 2025-2026
rate-limit changes published on the live docs site). Everything marked **VERIFY** must be
re-checked against the live docs or a live `exchangeInfo` call before going live, because
MEXC changes limits without notice.

Target: a maker-only scalper on **SOLUSDT** spot (configurable symbol), single account,
single process. Reference snippets are Python 3.11+ `asyncio`; the structure maps 1:1 to
Node/TypeScript.

Use Section 10 to diff an existing bot against this plan. Sections 1-9 are the spec.
Section 11 is the reference implementation skeleton. Appendix A lists every verified fact
with its source and what still needs re-verification.

---

## 0. What the logic can and cannot guarantee

These are the constraints that shape every design decision below. They are facts about the
venue and about market microstructure, not opinions.

1. **Order throughput is capped by MEXC, not by your code.** Order placement, single
   cancel, and cancel-all share one per-UID bucket of **12 requests/second** (live
   changelog, 2026). Batch orders are limited to **2 requests/second**, 20 orders each
   (30 in the newest docs, **VERIFY**). A full two-sided requote costs 4 requests
   (2 cancels + 2 placements, there is no amend endpoint), so the bot can do at most
   about **3 full requotes per second**. "Very fast trades in short succession" means
   "at most a handful of order actions per second". Market-maker tiers with hundreds of
   requests/second exist but require an application and volume.

2. **Taker trading loses by construction.** Regular-tier spot fees are **0% maker /
   0.05% taker** (0.04% taker with MX deduction; **VERIFY** for your account via
   `GET /api/v3/tradeFee`). A taker round trip costs 10 bp before slippage. The SOLUSDT
   top-of-book spread is normally 1 tick, 0.01 USDT, which is about 0.5 bp at a 200 USDT
   reference price. Submitting orders "regardless of opportunity" with taker fills has an
   expected value of roughly **-10 bp per round trip**. That is a deterministic loss, not
   a strategy, and no order sizing or frequency changes its sign.

3. **Maker trading pays no fee but pays adverse selection.** A resting bid is filled
   disproportionately in the moments before the price falls. For a participant 30-80 ms
   from the matching engine on a liquid pair, the measured adverse-selection cost is
   typically about equal to the half-spread captured, so the net edge is small and can be
   negative. It must be **measured** (Section 2.4), never assumed.

4. **No program logic makes every session profitable.** Designs that appear to (never
   closing at a loss, grids or martingale without an inventory cap, "wait for the price
   to come back") convert a stream of small wins into one large loss; the per-session hit
   rate looks like 95%+ until the session that erases it. What a correctly built bot *can*
   guarantee: it never pays taker fees unintentionally, never exceeds its inventory and
   loss limits, never trades on stale data, and stops itself. Profit is conditional on
   measured edge exceeding measured cost.

Consequently this plan specifies: post-only two-sided quoting around a microprice with
inventory skew; explicit edge measurement with a go/no-go gate; hard risk limits enforced
in code; and a rollout path (replay -> shadow -> canary -> live) that produces the
evidence before real size is at risk.

---

## 1. Scope and assumptions

| Item | Decision |
| --- | --- |
| Venue / product | MEXC **Spot** V3 REST + WebSocket (protobuf). Futures excluded: the contract API's order endpoints are marked "Under maintenance" and permissioned. |
| Symbol | `SOLUSDT` (base `SOL`, quote `USDT`), configurable. One symbol per process. |
| Account | One API key, trade + read permissions only, IP-whitelisted. Sub-account recommended so the key cannot touch other funds. |
| Runtime | One process, one `asyncio` event loop. Python 3.11+ reference. |
| Location | AWS Tokyo (`ap-northeast-1`) or Singapore (`ap-southeast-1`); MEXC's own FAQ recommends Japan/Singapore AWS for stable access. Measure RTT to `api.mexc.com` from both and pick the lower. |
| Numerics | `Decimal` for every price, quantity, fee, and balance. Floats only for statistics (volatility, markouts). |
| Time | Milliseconds UTC everywhere. Server-clock offset maintained continuously. |
| Persistence | SQLite (WAL mode) for orders, fills, markouts, and session PnL. |
| Not in scope | Multi-venue arbitrage, on-chain Solana execution, leverage, margin. |

---

## 2. Economics and the go/no-go gate

### 2.1 Cost model per round trip (one buy + one sell of notional N)

| Component | Maker/maker | Taker/taker | Notes |
| --- | --- | --- | --- |
| Exchange fee | 0 bp (0% maker) | 10 bp (2 x 0.05%) | Read live values from `exchangeInfo.makerCommission/takerCommission` and `GET /api/v3/tradeFee`. |
| Spread | **captured**: +S (quoted width) | **paid**: -S | S in bp of mid. 1 tick on SOLUSDT at 200 USDT is about 0.5 bp. |
| Adverse selection (AS) | paid, measured as markout | paid, usually smaller for takers | Section 2.4. |
| Slippage | 0 (price is yours) | > 0 when the top level is thin | |
| Rate-limit opportunity cost | 4 requests per requote | 1-2 requests per trade | Budget 12/s shared. |

Expected PnL per round trip, maker/maker:

```
E[PnL] = N * (S_captured - AS - 2*F_maker) / 1e4      (all terms in bp)
```

With `F_maker = 0`, the whole question is whether `S_captured > AS`. On SOLUSDT the
quoted width available to a post-only participant is 1-3 ticks (0.5-1.5 bp). The go/no-go
gate in 2.5 exists because `AS` on a liquid pair is of the same order.

### 2.2 Instrument constants loaded at startup (never hardcoded)

From `GET /api/v3/exchangeInfo?symbol=SOLUSDT`:

| Field | Meaning | How the bot uses it |
| --- | --- | --- |
| `quotePrecision` | price decimals | `tick = 10^-quotePrecision`; every price is quantized to it (bids round down, asks round up). |
| `baseAssetPrecision` | quantity decimals | `qty_step = 10^-baseAssetPrecision`; quantities round down. |
| `baseSizePrecision` | minimum order quantity (string) | reject any order below it. |
| `quoteAmountPrecision` | minimum order notional in quote (string) | reject any order with `price*qty` below it (error 30002 otherwise). |
| `maxQuoteAmount` | maximum notional per order | upper clamp. |
| `orderTypes` | allowed order types | must contain `LIMIT_MAKER`, otherwise refuse to start. |
| `isSpotTradingAllowed` | API spot trading allowed | must be `true`. |
| `status` / `tradeSideType` | 1 online; 1 = both sides allowed | refuse to start otherwise. |
| `makerCommission`, `takerCommission` | fee rates as strings | fee-mode guard (Section 7). |

Also: `GET /api/v3/defaultSymbols` (API-enabled symbols) and the signed
`GET /api/v3/selfSymbols` (symbols enabled for *your* account) must both include the
symbol; `GET /api/v3/kyc/status` should be checked once and logged.

### 2.3 Edge definition per fill

For a fill at price `p_f` on side `s` (+1 buy, -1 sell) with mid at fill time `m_0` and mid
`Δ` seconds later `m_Δ`:

```
captured_half_spread_bp = s * (m_0 - p_f) / m_0 * 1e4         # > 0 for a passive fill
markout_bp(Δ)           = s * (m_Δ - m_0) / m_0 * 1e4         # < 0 means adverse selection
edge_bp(Δ)              = captured_half_spread_bp + markout_bp(Δ) - fee_bp
```

`AS(Δ) = -mean(markout_bp(Δ))` over fills. Report `Δ ∈ {1 s, 5 s, 30 s, 120 s}`. The 30 s
and 120 s horizons are the ones that predict whether the inventory you are left holding
makes or loses money; 1 s flatters every market maker.

### 2.4 Markout measurement (mandatory instrumentation)

On every private deal (`spot@private.deals.v3.api.pb`), record `fill_price, side, qty,
fee, mid_0` and schedule captures of mid at each Δ from the 10 ms bookTicker stream.
Persist to SQLite table `markouts`. Compute daily: mean, standard error, and count per Δ,
split by side and by hour of day.

### 2.5 Go/no-go gate (numbers are defaults; make them config)

| Stage | Requirement to proceed |
| --- | --- |
| Replay backtest (Section 9.2) | `edge_bp(30 s) > 0` with conservative fill model over >= 5 trading days of recorded data, and max intraday drawdown < 1.5% of allocated capital. |
| Shadow mode (9.3) | >= 500 simulated fills; mean `edge_bp(30 s)` > 0 and `mean / stderr > 2`. |
| Canary live (9.4) | >= 200 real fills at minimum notional; realized `edge_bp(30 s)` within 1 bp of shadow; fill rate within 30% of shadow. |
| Scale-up | Only by doubling notional per step; re-run the canary check at each step. |

If any stage fails, the strategy parameters change (wider spread floor, stricter guards,
different hours) and the stage is re-run. The bot never goes to size on faith.

---

## 3. Strategy specification (the "logic")

### 3.1 Inputs and signals (recomputed on every bookTicker update)

| Signal | Definition | Source |
| --- | --- | --- |
| `bid, ask, bq, aq` | best bid/ask price and size | `spot@public.aggre.bookTicker.v3.api.pb@100ms@SOLUSDT` (10 ms stream only for markouts) |
| `mid` | `(bid + ask) / 2` | derived |
| `micro` | `(bid*aq + ask*bq) / (bq + aq)` size-weighted microprice; optionally 5-level weighted using `spot@public.limit.depth.v3.api.pb@SOLUSDT@5` | derived |
| `imb` | `(bq - aq) / (bq + aq)` in [-1, 1] | derived |
| `flow` | signed traded volume over the last `W_flow` seconds (buy - sell), normalized by total | `spot@public.aggre.deals.v3.api.pb@100ms@SOLUSDT` (`tradeType` 1 buy, 2 sell) |
| `sigma_bp` | realized volatility of mid over `W_vol` seconds (std of 1 s log returns), in bp | derived, EWMA |
| `spread_ticks` | `(ask - bid) / tick` | derived |
| `inv` | `base_balance_free+locked - base_target`, in base units | private account stream + reconciliation |

### 3.2 Quote computation

Practitioner form of the Avellaneda-Stoikov quote (reservation price shifted against
inventory, half-spread scaled by volatility), all in price units:

```
q_norm      = clamp(inv / max_inventory_base, -1, 1)
ref         = micro - q_norm * skew_ticks * tick                 # lean away from inventory
half_spread = max(min_half_spread_ticks * tick,                  # floor: the edge you insist on
                  vol_mult * sigma_bp / 1e4 * mid)                # widen when volatile
raw_bid     = ref - half_spread
raw_ask     = ref + half_spread
bid_px      = round_down_to_tick(raw_bid)
ask_px      = round_up_to_tick(raw_ask)
# post-only clamp: LIMIT_MAKER must not cross; keep at least 1 tick inside the touch
bid_px      = min(bid_px, ask - tick)
ask_px      = max(ask_px, bid + tick)
```

Optional directional lean (default off; enable only if shadow data shows it adds edge):
`ref += lean_ticks * tick * sign(flow)` when `|flow| > flow_threshold`.

Sizes:

```
notional     = order_notional_quote                           # e.g. 20 USDT
bid_qty      = round_down_to_step(notional / bid_px) * (1 - max(0, q_norm))   # shrink the side that adds inventory
ask_qty      = round_down_to_step(notional / ask_px) * (1 - max(0, -q_norm))
```

A side is dropped entirely (no order) when its size falls below `baseSizePrecision` or its
notional below `quoteAmountPrecision`, when `|q_norm| >= 1` on that side, or when a guard
in 3.5 is active.

### 3.3 Inventory control

| Rule | Behavior |
| --- | --- |
| Target | `base_target` SOL held at start (fund both sides: SOL for asks, USDT for bids). |
| Soft cap | `max_inventory_base`: at the cap, the inventory-adding side is not quoted. |
| Hard cap | `hard_inventory_base` (> soft cap, reached only through partial fills during gaps): cancel all quotes, then flatten the excess with one `IMMEDIATE_OR_CANCEL` LIMIT at the touch (or `MARKET` if IOC fails twice). This is the **only** path allowed to pay taker fees, and it is logged as a risk event. |
| Time cap | If `|inv| > 0.5 * max_inventory_base` for longer than `max_inventory_age_s`, lean harder (`skew_ticks *= 2`) rather than crossing. |
| Self-trade | Quotes never cross each other by construction. Before any flatten order, cancel both quotes and wait for the cancel acks. Also send `stpMode=cancel_maker` on every order if the account supports it (**VERIFY**: present in current SDKs, absent in the Feb-2025 doc mirror). |

### 3.4 Requote policy and the action budget

Every order action consumes from the shared 12/s UID bucket. The bot runs a local token
bucket at **8/s sustained, burst 10**, and keeps 2 tokens in reserve for risk-reducing actions
(cancels, cancel-all, flatten). Priority order when tokens are scarce: cancel-all >
flatten > cancel > place.

Requote a side only when at least one holds:

- desired price differs from the live order's price by `>= requote_tol_ticks` (default 1);
- desired side is now "no quote" or the live side is absent;
- the live order is older than `max_quote_age_s` (default 3 s) and the queue ahead of it
  (estimated from the limit-depth size at that level when placed) has not drained;
- inventory crossed a skew threshold.

Sequence per side: **cancel first, wait for the cancel ack or a terminal status on the
private orders stream, then place.** Never place the replacement before the cancel is
acknowledged; with 2 orders per side you would exceed your intended exposure during the
window. Coalesce: if several bookTicker updates arrive while a requote is in flight, act
only on the latest state after the in-flight action completes.

### 3.5 Guards (pull quotes; do not trade through these)

| Guard | Trigger | Action | Reset |
| --- | --- | --- | --- |
| Stale data | no bookTicker for `> stale_ms` (default 1000 ms) or `sendTime` lag `> lag_ms` (500 ms) | cancel-all, pause | fresh data for 2 s |
| Crossed / locked book | `ask <= bid` | pause | book uncrosses |
| Wide spread | `spread_ticks > max_spread_ticks` (e.g. 10) | quote nothing (usually a news event) | spread normal for 5 s |
| Volatility | `sigma_bp > vol_halt_bp` (e.g. 25 bp per minute) | cancel-all, pause | below `vol_resume_bp` for 30 s |
| Momentum | `|flow| > flow_halt` and price moved `> move_halt_bp` in 2 s | pull the side being run over | 10 s |
| Rate limit | any 429/418, or local bucket empty for `> 2 s` | cancel-all with the reserved tokens, back off `Retry-After` | manual or timer |
| Connectivity | private WS down `> 5 s`, or REST error rate `> 20%` over 30 s | cancel-all via REST, pause | both streams healthy 10 s |

### 3.6 Session accounting

Positions and PnL are derived from **private deals only** (never from your own order
intents). Average-cost method:

```
on BUY fill:  cash -= price*qty + fee_in_quote;  inv += qty
on SELL fill: cash += price*qty - fee_in_quote;  inv -= qty
MTM(t)        = cash(t) + inv(t) * mid(t)
session_pnl   = MTM(t) - MTM(t_0)
realized_pnl  = session_pnl - inv(t) * (mid(t) - avg_cost)
```

If MX fee deduction is enabled the fee currency is `MX`; either disable MX deduction for the
trading account or convert fees at the `MXUSDT` mid captured at fill time. Daily loss and
drawdown limits (Section 7) operate on `session_pnl`, not realized PnL, so an open losing
inventory cannot hide.

### 3.7 Explicitly excluded

- Market or IOC orders anywhere except the hard-cap flatten path.
- Submitting orders on a timer "regardless of opportunity".
- Averaging down, grids, or martingale sizing without the inventory caps above.
- Any rule of the form "do not sell at a loss". The loss limit is the rule.

---

## 4. MEXC integration spec (verified against the official docs)

### 4.1 REST basics

| Item | Value |
| --- | --- |
| Base URL | `https://api.mexc.com` |
| Auth header | `X-MEXC-APIKEY: <key>`; `Content-Type: application/json` or `application/x-www-form-urlencoded` |
| Signature | `signature = lowercase hex( HMAC-SHA256( key = secretKey, msg = totalParams ) )`, where `totalParams` is the query string concatenated with the request body. Send **everything in the query string** (as the official `POST /api/v3/order` example does); then `totalParams == query string` and there is no ambiguity. Values with special characters (the `batchOrders` JSON) must be URL-encoded with **uppercase** hex before signing. |
| Timing | `timestamp` (ms) is mandatory on signed calls. Server accepts if `timestamp < serverTime + 1000` and `serverTime - timestamp <= recvWindow`. Default `recvWindow` 5000, max 60000; use 5000 or less. |
| Clock sync | `GET /api/v3/time` at startup and every 60 s; `offset = serverTime - localMidpoint`. On error `700003` resync immediately and retry once. |
| Return codes | 4xx malformed request; **403 = WAF violation**; **429 = rate limit** (`Retry-After` header in seconds); **418 = IP banned** (`Retry-After` = seconds until unban); **5xx = outcome UNKNOWN, the order may exist** (Section 6.4). |

### 4.2 Rate limits

| Scope | Limit | Source / status |
| --- | --- | --- |
| Order placement + cancel + cancel-all (per UID) | **12 requests/second total, one shared bucket** | live changelog entries dated 2026-03-11 / 2026-05-03 / 2026-05-22 (**VERIFY** on the live page; the GitHub mirror predates it) |
| `POST /api/v3/batchOrders` | 2 requests/second, max 20 orders per request (newest docs: 30, **VERIFY**) | docs |
| `GET /api/v3/account` | 2 requests/second | docs |
| IP-weighted endpoints | mirror: 500 weight per 10 s per endpoint; live intro page: **300 weight / 10 s shared** across IP-weighted endpoints | **VERIFY** |
| UID-weighted endpoints | 500 weight per 10 s | docs |
| Open orders per account | 500 not-fully-filled orders | FAQ |
| WebSocket | 100 messages/second per connection; **30 subscriptions per connection**; disconnect after 30 s with no subscription or 60 s with no traffic; **must answer server PING with PONG**; user-data connections are cut at 24 h | docs + FAQ |
| listenKey | valid 60 min; `PUT` extends by 60 min (send every 30 min); max 60 keys per UID; 5 connections per key | docs |
| API keys | max 30 per account; a key without an IP whitelist expires after 90 days; up to 10 IPs per key | FAQ |
| Penalties | 429 -> back off; repeated violations -> automated IP ban scaling from 2 minutes to 3 days; FAQ also describes a 10-minute lockout after exceeding the frequency limit | docs |

### 4.3 Endpoints used

| Purpose | Method / path | Weight | Notes |
| --- | --- | --- | --- |
| Server time | `GET /api/v3/time` | 1 (IP) | clock sync |
| Instrument rules | `GET /api/v3/exchangeInfo?symbol=` | 10 (IP) | once at start, then hourly |
| API-enabled symbols | `GET /api/v3/defaultSymbols`; signed `GET /api/v3/selfSymbols` | 1 | startup check |
| Fee rate | signed `GET /api/v3/tradeFee?symbol=` | 20 (IP) | startup; fee-mode guard |
| Depth snapshot | `GET /api/v3/depth?symbol=&limit=` | 1 (IP) | only if maintaining a full local book (4.6) |
| Place order | `POST /api/v3/order` | 1 IP / 1 UID | `type=LIMIT_MAKER`, `newClientOrderId` |
| Validate order | `POST /api/v3/order/test` | 1 | CI smoke test of signing and filters, not in the hot path |
| Batch place | `POST /api/v3/batchOrders` | 1 / 1 | initial ladder only; 2 req/s |
| Cancel | `DELETE /api/v3/order` (`orderId` or `origClientOrderId`) | 1 | shares the 12/s bucket |
| Cancel all | `DELETE /api/v3/openOrders?symbol=` | 1 | up to 5 symbols comma-separated; the "nuke" used by every guard |
| Query order | `GET /api/v3/order` | 2 | resolve UNKNOWN outcomes by `origClientOrderId` |
| Open orders | `GET /api/v3/openOrders?symbol=` | 3 | reconciliation every 5-10 s |
| Balances | `GET /api/v3/account` | 10, 2 req/s | reconciliation every 10 s |
| Fills | `GET /api/v3/myTrades?symbol=` | 10 | backfill after reconnect; last 1 month only, max 100 per call |
| User stream key | `POST / PUT / DELETE /api/v3/userDataStream` | | 4.7 |

### 4.4 Order semantics

- Order types (enum): `LIMIT`, `MARKET`, `LIMIT_MAKER`, `IMMEDIATE_OR_CANCEL`,
  `FILL_OR_KILL`. There is **no `timeInForce` parameter** (responses echo `GTC`); the
  type carries the semantics. Quotes use `LIMIT_MAKER` (post-only; shown as
  `orderType = 2 POST_ONLY` on the private stream). A `LIMIT_MAKER` that would match is
  rejected by the engine rather than filled as a taker; treat the rejection as a no-op and
  requote on the next update (**VERIFY** the exact error code with a minimum-size test).
- `MARKET` needs `quantity` (base) or, for buys, `quoteOrderQty` (quote) when
  `quoteOrderQtyMarketAllowed` is true. Used only in the flatten path.
- Order statuses: `NEW`, `PARTIALLY_FILLED`, `FILLED`, `CANCELED`,
  `PARTIALLY_CANCELED`. Private stream integer codes: 1 not traded, 2 fully traded,
  3 partially traded, 4 canceled, 5 partially canceled.
- `newClientOrderId` is free-form; keep it to <= 32 alphanumeric characters (**VERIFY**
  the limit; it is undocumented). Cancel by `origClientOrderId` so the bot never depends
  on having received the placement response.
- **No amend/replace endpoint.** A requote is cancel + place.
- Self-trade prevention: `stpMode` in `{"" , cancel_maker, cancel_taker, cancel_both}`
  (**VERIFY**). Private deals carry `isSelfTrade`.
- `POST /api/v3/order` response: `symbol, orderId, orderListId, price, origQty, type,
  side, transactTime`. `orderId` is a string.
- Batch example in the docs uses `"type": "LIMIT_ORDER"` inside the JSON list; the
  per-order fields are `type, price, quantity, symbol, side, newClientOrderId`. Errors are
  returned per order (`code`, `msg`) inside the array, so parse each element.

### 4.5 Symbol filters

See 2.2. Rounding rules: price bids `ROUND_FLOOR`, asks `ROUND_CEILING`, to `tick`;
quantities `ROUND_FLOOR` to `qty_step`; check `qty >= baseSizePrecision`,
`price*qty >= quoteAmountPrecision`, `price*qty <= maxQuoteAmount` before every placement.
Error `30002` ("minimum transaction volume cannot be less than") is the signature of a
missing notional check.

### 4.6 Public WebSocket (market data)

| Item | Value |
| --- | --- |
| URL | `wss://wbs-api.mexc.com/ws` |
| Framing | binary frames are protobuf `PushDataV3ApiWrapper` (`channel`, `oneof body`, `symbol`, `symbolId`, `createTime`, `sendTime`); text frames are JSON control replies (`{"id":0,"code":0,"msg":"<channel>"}`, `{"msg":"PONG"}`) |
| Subscribe | `{"method":"SUBSCRIPTION","params":["<channel>", ...]}`; unsubscribe with `"UNSUBSCRIPTION"` |
| Keepalive | send `{"method":"PING"}` every 20-30 s; reply to server pings; expect disconnect at 60 s of silence |
| Proto files | `github.com/mexcdevelop/websocket-proto` (compile with `protoc --python_out`) |

Channels used:

| Channel | Message | Fields | Use |
| --- | --- | --- | --- |
| `spot@public.aggre.bookTicker.v3.api.pb@100ms@SOLUSDT` | `PublicAggreBookTickerV3Api` | `bidPrice, bidQuantity, askPrice, askQuantity, version, lastOrderCreateTime` | quoting loop |
| `spot@public.aggre.bookTicker.v3.api.pb@10ms@SOLUSDT` | same | | markout capture and latency measurement only |
| `spot@public.limit.depth.v3.api.pb@SOLUSDT@5` | `PublicLimitDepthsV3Api` | `asks[], bids[] {price, quantity}, eventType, version, lastOrderCreateTime` | 5-level microprice, queue-ahead estimate; full snapshot each push, no sync logic needed |
| `spot@public.aggre.deals.v3.api.pb@100ms@SOLUSDT` | `PublicAggreDealsV3Api` | `deals[] {price, quantity, tradeType (1 buy, 2 sell), time, tradeId}, eventType` | flow signal, shadow fill model |
| `spot@public.aggre.depth.v3.api.pb@(100ms|10ms)@SOLUSDT` | `PublicAggreDepthsV3Api` | `asks[], bids[], eventType, fromVersion, toVersion` | optional full local book |

Full local order book (only if you need queue position beyond 5 levels), per the official
procedure: subscribe to the diff stream; fetch `GET /api/v3/depth?symbol=&limit=1000`
(`lastUpdateId` is the snapshot version); drop pushes with `toVersion < snapshot.version`;
if `fromVersion > snapshot.version` re-snapshot; apply the first push whose
`[fromVersion, toVersion]` contains the snapshot version; thereafter require
`fromVersion == previous.toVersion + 1` or re-snapshot. Quantities are absolute; `0`
removes the level.

### 4.7 Private WebSocket (user data)

| Item | Value |
| --- | --- |
| Key | `POST /api/v3/userDataStream` -> `{"listenKey": ...}` (needs account-read permission); `PUT` every 30 min; `DELETE` on shutdown |
| URL | `wss://wbs-api.mexc.com/ws?listenKey=<key>` |
| Subscribe | same `SUBSCRIPTION` message with the three private channels; same PING rule; connection is cut at 24 h, so schedule a proactive reconnect at 23 h |

| Channel | Message | Fields used |
| --- | --- | --- |
| `spot@private.orders.v3.api.pb` | `PrivateOrdersV3Api` | `id, clientId, price, quantity, amount, avgPrice, orderType (1 LIMIT, 2 POST_ONLY, 3 IOC, 4 FOK, 5 MARKET, 100 stop), tradeType (1 buy, 2 sell), isMaker, remainQuantity, lastDealQuantity, cumulativeQuantity, cumulativeAmount, status (1..5), createTime` |
| `spot@private.deals.v3.api.pb` | `PrivateDealsV3Api` | `price, quantity, amount, tradeType, isMaker, isSelfTrade, tradeId, clientOrderId, orderId, feeAmount, feeCurrency, time` |
| `spot@private.account.v3.api.pb` | `PrivateAccountV3Api` | `vcoinName, balanceAmount, balanceAmountChange, frozenAmount, frozenAmountChange, type (ENTRUST, ENTRUST_PLACE, ENTRUST_CANCEL, TRADE_FEE, ENTRUST_UNFROZEN, ...), time` |

Deals are the source of truth for fills and fees; orders are the source of truth for
status; account updates are a cross-check only.

### 4.8 Error handling

| Code / condition | Meaning | Bot action |
| --- | --- | --- |
| HTTP 429 | rate limit | stop placing; cancel-all with reserved tokens if quotes are live; sleep `Retry-After`; count toward the rate-limit guard |
| HTTP 418 | IP banned | halt; alert; do not retry until `Retry-After` |
| HTTP 403 | WAF | halt; alert (usually a header/body format problem or an IP issue) |
| HTTP 5xx / timeout on place or cancel | outcome unknown | Section 6.4 |
| 700003 | timestamp outside recvWindow | resync clock, retry once |
| 700002 / 602 | bad signature | halt (config bug) |
| 700006 | IP not whitelisted | halt |
| 700007 / 401 | permission | halt |
| 30002 / 30003 | below min / above max volume | bug in filter checks; halt the side, alert |
| 30004 / 10101 | insufficient balance | drop that side until reconciliation refreshes balances |
| 30016 / 30000 / 30020 | trading disabled / suspended / no permission for symbol | halt |
| 30041 | order type cannot place | likely `LIMIT_MAKER` would cross; treat as no-op |
| -2011 | unknown order on cancel | already gone; query by `origClientOrderId`, then mark terminal |
| 10007 / 30014 / 30021 | bad symbol | halt (config) |

---

## 5. Architecture

### 5.1 Components

| Component | Responsibility | Talks to |
| --- | --- | --- |
| `Clock` | server-time offset, monotonic timers | REST `/time` |
| `MarketData` | public WS connection(s), protobuf decode, top-of-book / 5-level / trades state, staleness tracking | `QuoteEngine`, `Markouts` |
| `UserData` | listenKey lifecycle, private WS, decode of orders / deals / account | `OrderManager`, `Accounting` |
| `RestGateway` | signing, HTTP client with 3 s timeout, token buckets (order bucket 8/s, query bucket), retry policy, error mapping | everything that writes |
| `Instrument` | filters, tick, step, mins, fees; rounding helpers | all |
| `QuoteEngine` | pure function: (book, trades, inventory, vol) -> desired quotes | `Strategy` |
| `OrderManager` | per-side order state machines, cancel-then-place sequencing, UNKNOWN resolution, reconciliation | `RestGateway`, `UserData` |
| `RiskManager` | pre-trade checks, guards, kill switch, loss limits | `Strategy`, `OrderManager` |
| `Accounting` | inventory, cash, MTM, session PnL from deals | `RiskManager`, `Metrics` |
| `Markouts` | per-fill markout capture at Δ horizons | `Persistence` |
| `Persistence` | SQLite WAL: orders, fills, markouts, events, config snapshot | all |
| `Metrics` | Prometheus endpoint or JSON lines: PnL, inventory, fill counts, edge, bucket usage, latencies, guard states | ops |
| `Supervisor` | task supervision, restart with backoff, health endpoint, graceful shutdown (cancel-all first) | all |

### 5.2 Data flow

```
 wss://wbs-api.mexc.com/ws (public)          wss://wbs-api.mexc.com/ws?listenKey= (private)
   bookTicker@100ms / @10ms, deals, depth5       orders, deals, account
            |                                            |
            v                                            v
      [MarketData] --state+event-->  [Strategy loop]  <--fills/status-- [UserData]
            |                           |      ^                            |
            |   mids for markouts       v      |                            v
            +--------------------> [QuoteEngine]   [RiskManager] <---- [Accounting]
                                        |              |
                                        v              v (halt / cancel-all)
                                   [OrderManager] --(cancel first, then place)--> [RestGateway] --HTTPS--> api.mexc.com
                                        |                                               ^
                                        +---- reconcile (openOrders, account, order) ----+
                                        |
                                        v
                                  [Persistence / Metrics]
```

### 5.3 Event loop and coalescing

- One `asyncio` loop. WS callbacks only decode and update in-memory state, then `set()` an
  `asyncio.Event`. No I/O and no strategy logic inside callbacks.
- The strategy task loops: `await event.wait(); event.clear(); snapshot = state.latest()`;
  compute desired quotes; submit to `OrderManager`. If 20 updates arrive during one REST
  round trip, the strategy acts once on the newest state. This is what keeps request
  usage under the budget.
- `OrderManager` keeps one in-flight action per side. A new desired price while an action
  is in flight is stored as "pending desired" and evaluated when the action completes.
- Blocking work (SQLite writes, metrics) goes to a thread executor or a batched writer task.

### 5.4 Latency budget (measure, then set the config from the measurements)

| Segment | Expected | Measure as |
| --- | --- | --- |
| Exchange -> bot WS delivery | 1-5 ms intra-region; the 100 ms channel adds up to 100 ms aggregation | `local_recv_ms - sendTime` (needs clock sync) |
| Protobuf decode + state update | < 0.1 ms | timer |
| Decision | < 0.5 ms | timer |
| REST place / cancel RTT | 20-80 ms from Tokyo/Singapore, 150-300 ms from EU/US | timer around the HTTP call; histogram |
| Private stream confirms fill | 10-50 ms after match | `local_recv_ms - deal.time` |

Consequence: a quote is always `RTT + aggregation interval` behind the market. The
`requote_tol_ticks` and `min_half_spread_ticks` defaults assume ~100 ms total staleness.
If measured staleness exceeds 250 ms, widen `min_half_spread_ticks` by one tick per 100 ms.

### 5.5 Persistence schema (SQLite, WAL)

```sql
CREATE TABLE orders   (cid TEXT PRIMARY KEY, exchange_id TEXT, side TEXT, px TEXT, qty TEXT,
                       state TEXT, reason TEXT, created_ms INTEGER, updated_ms INTEGER,
                       ack_latency_ms INTEGER, cancel_latency_ms INTEGER);
CREATE TABLE fills    (trade_id TEXT PRIMARY KEY, cid TEXT, exchange_id TEXT, side TEXT,
                       px TEXT, qty TEXT, fee TEXT, fee_ccy TEXT, is_maker INTEGER,
                       is_self INTEGER, t_ms INTEGER, mid0 TEXT);
CREATE TABLE markouts (trade_id TEXT, horizon_s INTEGER, mid TEXT, markout_bp REAL,
                       PRIMARY KEY (trade_id, horizon_s));
CREATE TABLE events   (t_ms INTEGER, kind TEXT, detail TEXT);  -- guards, halts, 429s, reconciles
CREATE TABLE sessions (session TEXT PRIMARY KEY, started_ms INTEGER, config_json TEXT,
                       mtm_start TEXT, mtm_end TEXT, fills INTEGER, requests INTEGER);
```

### 5.6 Logging

JSON lines, one object per event, with `t_ms, session, kind, cid, side, px, qty, reason,
bucket_tokens, latency_ms, code, msg`. Every order action logs its *reason* (which rule
fired). Every guard transition logs on and off. Never log the API secret or the listenKey.

---

## 6. Order management

### 6.1 Client order IDs

`cid = f"{session_tag}{seq:08d}"` where `session_tag` is 4-6 random alphanumerics
generated at process start and `seq` is a persisted counter. Example: `k7q2a00000317`.
Alphanumeric only, <= 16 characters. Orders found on the exchange whose `clientOrderId`
does not start with the current `session_tag` are orphans from a previous run and are
canceled during startup reconciliation.

### 6.2 State machine (one per order)

```
            place()                 REST 2xx ack / private NEW
  DESIRED ---------> PENDING_NEW ------------------------------> OPEN
                         |                                        |  deal (partial)
                         | REST 5xx/timeout                       v
                         +-----------> UNKNOWN_NEW            PARTIALLY_FILLED
                                           | query               |          \
                                           v                     | deal      \ cancel()
                                     OPEN or NEVER_PLACED        v            v
                                                               FILLED    CANCEL_PENDING
                                                                              |  ack / private status 4 or 5
                                                                              v
                                                             CANCELED / PARTIALLY_CANCELED
  Any state ---- REST 4xx reject (LIMIT_MAKER would cross, filters, balance) ---> REJECTED
  CANCEL_PENDING ---- REST 5xx/timeout ---> UNKNOWN_CANCEL --query--> (still open: cancel again) | terminal
```

Terminal states: `FILLED`, `CANCELED`, `PARTIALLY_CANCELED`, `REJECTED`, `NEVER_PLACED`.
Illegal transitions raise and halt the side; they indicate a logic bug, not a market event.

Inputs that drive transitions: REST responses, `spot@private.orders` (status 1..5, keyed
by `clientId`), `spot@private.deals` (keyed by `clientOrderId`), reconciliation queries.
Fill quantity comes from deals, not from order-status messages; order status only moves
the state.

### 6.3 Action sequences and their cost

| Action | Requests (order bucket) | Timeout | On timeout |
| --- | --- | --- | --- |
| Place | 1 | 2 s | `UNKNOWN_NEW` |
| Cancel one | 1 | 2 s | `UNKNOWN_CANCEL` |
| Requote one side | 2 (cancel, then place after ack) | | |
| Cancel-all (symbol) | 1 | 2 s | query `openOrders`; repeat once |
| Flatten | 1 cancel-all + 1 IOC | | |

The replacement is placed only after the cancel is acknowledged (REST 2xx or private
status 4/5). If the cancel comes back `-2011` (unknown order) the order was already filled
or canceled: query it, apply the result, then decide whether a new order is still desired.

### 6.4 UNKNOWN outcome protocol (HTTP 5xx, timeouts, connection resets)

MEXC's docs say explicitly that 5xx means the execution status is unknown and may have
succeeded. Therefore:

1. Never re-send a placement blindly.
2. `GET /api/v3/order?symbol=...&origClientOrderId=<cid>` (query bucket, not the order
   bucket). Retry the query up to 3 times over 1.5 s.
3. Found -> adopt its status and quantities. Not found after the retries ->
   `NEVER_PLACED`, the cid is retired (never reused), and the side may place a fresh order
   with a new cid.
4. For `UNKNOWN_CANCEL`: query; if still open, cancel again (idempotent); if terminal,
   apply.
5. While any order on a side is UNKNOWN, that side places nothing.

### 6.5 Reconciliation (runs continuously; also after every reconnect)

| Check | Interval | Drift handling |
| --- | --- | --- |
| `GET /api/v3/openOrders?symbol=` vs local open set | 5 s | exchange-only order -> cancel it, log `orphan`; local-only `OPEN` -> query, apply terminal state |
| `GET /api/v3/account` balances vs local inventory/cash | 10 s (limit is 2/s) | `|drift| <= qty_step`: ignore; `qty_step < |drift| <= order_qty`: adopt exchange values, log; `|drift| > order_qty`: halt and alert (missing fills) |
| `GET /api/v3/myTrades?symbol=&startTime=` backfill | after reconnect | insert any fills missing from `fills`, replay them through accounting and markouts (markouts marked `backfilled`) |
| listenKey validity | every 30 min `PUT`; on any error create a new key and reconnect | |

### 6.6 Procedures

Startup:
1. Load config; open SQLite; create session row.
2. Sync clock. Load `exchangeInfo`, `tradeFee`, `defaultSymbols`, `selfSymbols`; run the
   instrument assertions (Section 2.2) and the fee-mode guard (Section 7).
3. `DELETE /api/v3/openOrders?symbol=` (cancel everything, including orphans).
4. `GET /api/v3/account`: set `cash`, `inv = base_balance - base_target`, `MTM(t_0)`.
5. Create listenKey, connect private WS, subscribe to the three private channels, wait
   for subscription acks.
6. Connect public WS, subscribe, wait until bookTicker is fresh (< 500 ms) for 2 s.
7. Start reconciliation, keepalive, markout, metrics tasks.
8. Enter the strategy loop with all guards evaluated once before the first quote.

Shutdown (SIGTERM, kill file, fatal error): stop the strategy loop; cancel-all via REST
(retry with query until `openOrders` is empty or 10 s elapse); persist session end MTM;
`DELETE` listenKey; close sockets; exit non-zero if cancel-all could not be confirmed.

Disconnect: public WS gap > `stale_ms` or private WS gap > 5 s -> cancel-all via REST,
pause; reconnect with exponential backoff (0.5 s to 10 s); after reconnect run full
reconciliation and backfill, then wait for 2 s of fresh data before quoting.

---

## 7. Risk management (hard rules, enforced in code before every request and on every fill)

| # | Rule | Trigger | Action | Config key |
| --- | --- | --- | --- | --- |
| R1 | Fee-mode guard | `takerCommission > 0` and any non-flatten path could take | refuse to start | `allow_taker_except_flatten: false` |
| R2 | Order-type guard | `LIMIT_MAKER` not in `orderTypes` | refuse to start | |
| R3 | Max notional per order | `price*qty > max_order_notional` | reject the action | `max_order_notional_quote` |
| R4 | Max open orders per side | > 1 live or pending order on a side | reject the action | `max_open_per_side: 1` |
| R5 | Soft inventory cap | `|inv| >= max_inventory_base` | drop the adding side | `max_inventory_base` |
| R6 | Hard inventory cap | `|inv| >= hard_inventory_base` | cancel-all, flatten excess (IOC), log risk event | `hard_inventory_base` |
| R7 | Daily loss limit | `session_pnl <= -daily_loss_limit_quote` | cancel-all, flatten to target, **halt for the day** (manual restart) | `daily_loss_limit_quote` |
| R8 | Drawdown from session high | `MTM_high - MTM >= max_drawdown_quote` | cancel-all, halt | `max_drawdown_quote` |
| R9 | Reject storm | >= 5 consecutive rejects or reject rate > 20% over 60 s | pause the side 30 s; after 3 pauses halt | `max_consecutive_rejects` |
| R10 | Rate-limit events | any 429/418 | Section 4.8 | |
| R11 | Stale data | Section 3.5 | cancel-all, pause | `stale_ms`, `lag_ms` |
| R12 | Clock drift | `|offset| > 1000 ms` or two consecutive 700003 | resync; if persistent, halt | |
| R13 | Price sanity | desired price outside `±max_dev_bp` of both last trade and current mid, or `tick`-rounded price crosses the touch | reject the action, log | `max_dev_bp: 50` |
| R14 | Filter checks | below `baseSizePrecision` / `quoteAmountPrecision`, above `maxQuoteAmount` | reject the action (should never fire; alert if it does) | |
| R15 | Balance check | insufficient free balance for the side (from private account stream + reconciliation) | drop the side until refreshed | |
| R16 | Kill switch | file `KILL` present, or `/halt` endpoint, or SIGTERM | shutdown procedure | `kill_file` |
| R17 | Session length | `runtime > max_session_hours` | graceful restart (fresh listenKey, fresh session tag) | `max_session_hours: 23` |
| R18 | Flatten failure | IOC flatten unfilled twice | one `MARKET` with `quantity`; if that fails, halt and alert | |
| R19 | Startup sanity | any open order for the symbol after the startup cancel-all | halt | |
| R20 | Self-trade | any deal with `isSelfTrade = true` | log, count; halt if > 2 per session (indicates a sequencing bug) | |

Halting always means: cancel-all confirmed via `openOrders`, then stop quoting. Halts
never auto-resume on the same day for R7 and R8.

---

## 8. Configuration schema (YAML; every value above comes from here)

```yaml
exchange:
  rest_base: https://api.mexc.com
  ws_public: wss://wbs-api.mexc.com/ws
  api_key_env: MEXC_API_KEY          # read from the environment, never from the file
  api_secret_env: MEXC_API_SECRET
  recv_window_ms: 5000
  http_timeout_s: 3.0
  order_bucket_rate_per_s: 8         # MEXC hard limit 12/s shared by place+cancel+cancel-all
  order_bucket_burst: 10
  order_bucket_reserve_tokens: 2     # only cancels / cancel-all / flatten may use these
  clock_sync_interval_s: 60

instrument:
  symbol: SOLUSDT
  base_target: 2.0                   # SOL held at start; inventory is measured relative to this
  refresh_exchange_info_s: 3600

strategy:
  order_notional_quote: 20           # per order
  min_half_spread_ticks: 1
  vol_mult: 0.5                      # half_spread >= vol_mult * sigma_bp
  skew_ticks: 2                      # reference shift per unit of q_norm
  requote_tol_ticks: 1
  max_quote_age_s: 3
  lean_enabled: false
  lean_ticks: 1
  flow_threshold: 0.6
  vol_window_s: 60
  flow_window_s: 5

inventory:
  max_inventory_base: 1.0            # soft cap (SOL)
  hard_inventory_base: 1.5           # flatten above this
  max_inventory_age_s: 120

guards:
  stale_ms: 1000
  lag_ms: 500
  max_spread_ticks: 10
  vol_halt_bp: 25
  vol_resume_bp: 12
  flow_halt: 0.85
  move_halt_bp: 15

risk:
  max_order_notional_quote: 50
  max_open_per_side: 1
  daily_loss_limit_quote: 15
  max_drawdown_quote: 10
  max_consecutive_rejects: 5
  max_dev_bp: 50
  kill_file: ./KILL
  max_session_hours: 23
  allow_taker_except_flatten: false

mode: shadow                         # replay | shadow | canary | live

ops:
  sqlite_path: ./data/scalper.sqlite
  metrics_port: 9100
  log_level: INFO
```

---

## 9. Testing, validation, and rollout

### 9.1 Unit tests (run in CI, no network)

- **Signing**: with secret `45d0b3c26f2644f19bfb98b07741b2f5` and `totalParams`
  `symbol=BTCUSDT&side=BUY&type=LIMIT&quantity=1&price=11&recvWindow=5000&timestamp=1644489390087`
  the signature must be
  `fd3e4e8543c5188531eb7279d68ae7d26a573d0fc5ab0d18eb692451654d837a`
  (verified locally with `openssl dgst -sha256 -hmac`; the official docs print a different
  hash in their "Example 1" for the same input, which is wrong, and this value in
  "Example 2", which is right). Also test that a `batchOrders` JSON value is URL-encoded
  with uppercase hex before signing and that the signed string equals the sent query
  string byte for byte.
- **Rounding**: bids round down, asks round up, quantities round down; min-notional and
  min-qty checks; `Decimal` formatting never emits exponent notation (`1E-7`).
- **Token bucket**: at most `rate` acquisitions per second over a 10 s simulated run;
  reserved tokens are only granted to risk-reducing actions.
- **Quote engine invariants** (property-based): `bid < ask`; `bid <= best_ask - tick`;
  `ask >= best_bid + tick`; sizes respect filters; at `|q_norm| >= 1` the adding side is
  `None`; wider half-spread when `sigma_bp` rises; skew direction opposes inventory.
- **State machine**: every documented transition succeeds; every undocumented one raises.
- **Order book sync** (if used): the seven-step version algorithm from 4.6 with gap,
  stale, and overlap cases.
- **Accounting**: buy then sell at a higher price yields the expected PnL net of fees;
  MX-denominated fees are converted; MTM equals cash + inventory * mid.
- **Decode**: protobuf fixtures recorded from the real stream decode to the expected
  fields (`WhichOneof("body")` dispatch).

### 9.2 Recorder and replay backtest

- **Recorder**: a standalone process that subscribes to bookTicker@10ms, deals@10ms, and
  depth5 and writes raw binary frames with a local receive timestamp (length-prefixed,
  gzip). Run it continuously on the trading server; it costs nothing and produces the
  evidence base. Keep at least 30 days.
- **Replay**: feeds frames through the same `MarketData` decoder into the same
  `QuoteEngine` and `OrderManager`, with `RestGateway` replaced by a simulator.
- **Simulator latency model**: an order becomes visible `RTT_place` after the decision; a
  cancel takes effect `RTT_cancel` after it is issued; both drawn from the measured
  histograms (5.4), not constants.
- **Fill model** (conservative, no queue data): a resting bid fills when a public trade
  prints at a price **strictly below** it, or at its price after at least `qty_ahead`
  (the displayed size at that level when the order arrived) has traded at that price.
  Optimistic variant (touch = fill) is reported alongside but never used for the gate.
- **Outputs**: fills, `edge_bp(Δ)` table, PnL curve, max drawdown, inventory histogram,
  time-at-cap, requests per second used, guard activations. Parameter sweeps over
  `min_half_spread_ticks`, `skew_ticks`, `order_notional_quote`, and trading hours.

### 9.3 Shadow mode

Same binary, `mode: shadow`: live market data, real listenKey (to validate the private
stream plumbing), but `RestGateway` is replaced by the simulator from 9.2, so no order ever
reaches the exchange. Runs for at least 2 full days spanning Asia, Europe, and US hours.
Produces the shadow fill set used by the gate in 2.5.

### 9.4 Canary live

`mode: canary`: real orders, `order_notional_quote` at the exchange minimum,
`max_open_per_side: 1`, `daily_loss_limit_quote` at 2 USDT. One hour, then one day. Compare
realized `edge_bp(30 s)`, fill rate, and reject rate to shadow. Any 429, any self-trade,
any reconciliation drift > `qty_step`, or any UNKNOWN that did not resolve is a failed
canary regardless of PnL.

### 9.5 Rollback rule

Any failed gate, any R7/R8 halt, or any day with `edge_bp(30 s) < 0` over >= 100 fills
returns the deployment to shadow mode until the cause is identified and a replay shows the
fix. Scaling is only ever by 2x per step with the canary comparison repeated.

---

## 10. Audit checklist: diff an existing bot against this plan

Mark each item PASS / FAIL / N/A. A FAIL on any item marked **(blocking)** means the bot
should not run with real size until it is fixed.

**A. Account and connectivity**

- [ ] A1 API key has spot-trade and account-read permissions only (no withdrawal), is IP-whitelisted, lives in the environment or a secret manager, never in the repo or logs.
- [ ] A2 **(blocking)** Signs HMAC-SHA256 over exactly the bytes sent (`totalParams` = query string + body) and passes the test vector in 9.1.
- [ ] A3 Maintains a server-clock offset from `/api/v3/time`, resyncs on error 700003, uses `recvWindow` <= 5000.
- [ ] A4 Runs from AWS Tokyo or Singapore; a REST round-trip histogram exists and the quoting tolerances were set from it.
- [ ] A5 **(blocking)** Treats 429/418 with `Retry-After`, 403 as a halt, and 5xx/timeouts as UNKNOWN outcomes (6.4), never as "failed, retry".

**B. Market data**

- [ ] B1 Connects to `wss://wbs-api.mexc.com/ws` and decodes protobuf `PushDataV3ApiWrapper` (not a legacy JSON stream, not REST polling of `/ticker` or `/depth` for quoting).
- [ ] B2 Quoting is driven by `spot@public.aggre.bookTicker.v3.api.pb` (or a correctly synced local book), with the 10 ms stream used only for measurement.
- [ ] B3 Sends `{"method":"PING"}` at least every 30 s, answers server pings, reconnects with backoff, keeps <= 30 subscriptions per connection.
- [ ] B4 Tracks staleness (local receive time vs `sendTime`) and pulls quotes when data is older than 1 s.
- [ ] B5 If it maintains a full local book, it implements the `fromVersion`/`toVersion` procedure in 4.6; otherwise it uses `limit.depth` snapshots.

**C. Strategy**

- [ ] C1 **(blocking)** Every quoting order is `LIMIT_MAKER`. `MARKET` / `IMMEDIATE_OR_CANCEL` appear only in the hard-cap flatten path.
- [ ] C2 Prices are computed from the current microprice/mid at decision time with a half-spread floor of >= 1 tick, not from fixed "set prices".
- [ ] C3 Inventory is measured against a target, with a soft cap that drops the adding side and a hard cap that flattens.
- [ ] C4 Half-spread widens with realized volatility; quotes are pulled above a volatility threshold.
- [ ] C5 **(blocking)** No order is ever submitted on a timer or "regardless of opportunity".
- [ ] C6 No averaging down, grid, or martingale sizing without the caps in C3.
- [ ] C7 Guards exist for stale data, crossed book, wide spread, momentum, connectivity loss, and rate-limit events, and each guard cancels resting quotes.
- [ ] C8 Markouts are captured per fill at 1 s / 5 s / 30 s / 120 s and persisted.
- [ ] C9 The go/no-go gate in 2.5 has been run and its numbers are written down.

**D. Execution and order management**

- [ ] D1 **(blocking)** At most one live order per side; a replacement is placed only after the cancel is acknowledged.
- [ ] D2 A local token bucket caps place + cancel + cancel-all at <= 8/s with tokens reserved for risk-reducing actions (MEXC limit: 12/s shared).
- [ ] D3 Client order IDs are unique per session and used for cancel and query (`origClientOrderId`).
- [ ] D4 **(blocking)** An explicit order state machine with UNKNOWN states; a placement is never re-sent without querying first.
- [ ] D5 Fills, fees, and maker/taker flags come from `spot@private.deals.v3.api.pb`; order status comes from `spot@private.orders.v3.api.pb`; neither is inferred from REST responses alone.
- [ ] D6 Reconciliation: `openOrders` every 5 s, `account` every 10 s, `myTrades` backfill after any reconnect; orders not created by this session are canceled.
- [ ] D7 **(blocking)** Startup cancel-all; shutdown cancel-all confirmed by an empty `openOrders`.
- [ ] D8 Tick, step, minimum quantity, minimum notional, and maximum notional are read from `exchangeInfo` and enforced with `Decimal`; no floats touch prices or quantities.
- [ ] D9 listenKey is extended every 30 min and the connection is recycled before the 24 h cut.
- [ ] D10 Self-trade is impossible by construction: bid < ask always; both quotes are canceled before a flatten order; `stpMode` is sent when supported; `isSelfTrade` deals are counted and alert.

**E. Risk**

- [ ] E1 **(blocking)** Daily loss limit on mark-to-market PnL halts for the rest of the day without auto-resume.
- [ ] E2 Session drawdown limit halts.
- [ ] E3 Max notional per order and max open orders are enforced before every request.
- [ ] E4 Price sanity (within +/- 50 bp of mid; never crossing the touch) is enforced before every request.
- [ ] E5 Reject-storm and rate-limit guards pause the bot.
- [ ] E6 **(blocking)** A kill switch (file, endpoint, or signal) runs the full shutdown procedure.
- [ ] E7 The bot refuses to start if its configuration could pay taker fees outside the flatten path.

**F. Accounting and reporting**

- [ ] F1 PnL is `cash + inventory * mid - start`, computed from fills, never from intended trades.
- [ ] F2 Fees are converted to the quote currency, or MX deduction is disabled on the trading account.
- [ ] F3 A daily report shows fills, maker ratio (expected 100%), `edge_bp` per horizon, PnL, max drawdown, time at inventory cap, requests per second, guard activations, and 429 count.

**G. Testing and operations**

- [ ] G1 The unit tests in 9.1 exist and run in CI.
- [ ] G2 A recorder runs continuously and a replay harness with the conservative fill model exists.
- [ ] G3 Shadow mode uses the production code path with only the gateway swapped.
- [ ] G4 A canary at minimum notional has been run and compared with shadow.
- [ ] G5 Structured logs record the reason for every order action; secrets and listenKeys are never logged.
- [ ] G6 A supervisor restarts the process on crash; alerts fire on halt, 429/418, reconciliation drift, unresolved UNKNOWN, and repeated reconnects.

---

## 11. Reference implementation skeleton (Python 3.11+, asyncio)

All blocks below concatenate into one module, `mexc_scalper.py`. It needs `aiohttp`,
`websockets`, `protobuf`, and the official protos compiled into a package `mexc_proto`
(`protoc --python_out=mexc_proto *.proto` from `github.com/mexcdevelop/websocket-proto`).
`python mexc_scalper.py selftest` runs the no-network checks from 9.1 (signing vector,
rounding, bucket, quote invariants, state machine, accounting). `main()` wires the live
loop; the shadow/replay gateway swap and the SQLite persistence are left as the obvious
extension points and are described in Sections 5.5 and 9.

Verification status of this skeleton: `selftest` passes; the decode path was exercised with
messages serialized from the official `.proto` files (`WhichOneof("body")` dispatch,
`sendTime`, `bidPrice`, `clientId`, `clientOrderId`, `isMaker`); and a fake-exchange harness
exercised the order manager (place, partial fill, cancel-first requote, UNKNOWN resolution by
query, `-2011` on cancel, cancel-all, orphan reconciliation, `LIMIT_MAKER` rejection, 429
propagation) and the strategy loop (requotes on a moving book, stale-data cancel-all fired
exactly once). It has not been run against the live exchange. Note that `protoc` emits flat
sibling imports (`import PublicDealsV3Api_pb2`), so put the `mexc_proto` directory itself on
`PYTHONPATH` as well, or rewrite those imports to package-relative ones.

### 11.1 Imports, errors, Decimal helpers

```python
"""mexc_scalper.py: reference skeleton for docs/MEXC_SCALPER_PLAN.md (MEXC spot, maker-only)."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
import random
import string
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from enum import Enum
from typing import Any, Callable, Optional
from urllib.parse import urlencode

import aiohttp
import websockets

try:
    from mexc_proto import PushDataV3ApiWrapper_pb2 as pb  # compiled official protos
except ImportError:  # the selftest does not need them
    pb = None

log = logging.getLogger("scalper")
D = Decimal
ZERO = D(0)


def fmt(x: Decimal) -> str:
    """Decimal -> API string, never exponent notation ('1E-7' would be rejected)."""
    return format(x, "f")


class MexcError(Exception):
    def __init__(self, status: int, payload: Any):
        super().__init__(f"HTTP {status}: {payload}")
        self.status, self.payload = status, payload

    @property
    def code(self) -> Optional[int]:
        return self.payload.get("code") if isinstance(self.payload, dict) else None


class RateLimited(MexcError):
    def __init__(self, status: int, payload: Any, retry_after: float):
        super().__init__(status, payload)
        self.retry_after = retry_after


class UnknownOutcome(Exception):
    """5xx or timeout: the request may or may not have executed (Section 6.4)."""
```

### 11.2 Instrument filters and rounding (Sections 2.2, 4.5)

```python
@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick: Decimal          # 10^-quotePrecision
    qty_step: Decimal      # 10^-baseAssetPrecision
    min_qty: Decimal       # baseSizePrecision
    min_notional: Decimal  # quoteAmountPrecision
    max_notional: Decimal  # maxQuoteAmount
    maker_fee: Decimal
    taker_fee: Decimal

    @classmethod
    def from_exchange_info(cls, s: dict) -> "Instrument":
        return cls(
            symbol=s["symbol"],
            tick=D(1).scaleb(-int(s["quotePrecision"])),
            qty_step=D(1).scaleb(-int(s["baseAssetPrecision"])),
            min_qty=D(str(s["baseSizePrecision"])),
            min_notional=D(str(s["quoteAmountPrecision"])),
            max_notional=D(str(s["maxQuoteAmount"])),
            maker_fee=D(str(s["makerCommission"])),
            taker_fee=D(str(s["takerCommission"])),
        )

    def round_price(self, px: Decimal, side: str) -> Decimal:
        """Bids round down, asks round up: rounding never moves a quote toward the touch."""
        rounding = ROUND_FLOOR if side == "BUY" else ROUND_CEILING
        return ((px / self.tick).to_integral_value(rounding=rounding) * self.tick).quantize(self.tick)

    def round_qty(self, qty: Decimal) -> Decimal:
        return ((qty / self.qty_step).to_integral_value(rounding=ROUND_FLOOR) * self.qty_step).quantize(self.qty_step)

    def reject_reason(self, px: Decimal, qty: Decimal) -> Optional[str]:
        if px <= ZERO:
            return "price<=0"
        if qty < self.min_qty:
            return "qty<min_qty"
        notional = px * qty
        if notional < self.min_notional:
            return "notional<min"
        if notional > self.max_notional:
            return "notional>max"
        return None
```

### 11.3 Token bucket and REST gateway (Sections 3.4, 4.1-4.3)

```python
class TokenBucket:
    """Local budget for MEXC's shared order bucket (12 req/s per UID for place + cancel + cancel-all).
    The last `reserve` tokens are only available to risk-reducing actions."""

    def __init__(self, rate_per_s: float, burst: int, reserve: int = 2,
                 clock: Callable[[], float] = time.monotonic):
        self.rate, self.capacity, self.reserve, self.clock = rate_per_s, float(burst), float(reserve), clock
        self.tokens, self.t = float(burst), clock()

    def try_acquire(self, risk_reducing: bool = False) -> bool:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
        self.t = now
        floor = 0.0 if risk_reducing else self.reserve
        if self.tokens - 1.0 >= floor:
            self.tokens -= 1.0
            return True
        return False


class RestGateway:
    def __init__(self, session: aiohttp.ClientSession, base: str, api_key: str, api_secret: str,
                 recv_window_ms: int = 5000, timeout_s: float = 3.0):
        self._s, self._base, self._key = session, base.rstrip("/"), api_key
        self._secret = api_secret.encode()
        self._recv, self._timeout = recv_window_ms, aiohttp.ClientTimeout(total=timeout_s)
        self.offset_ms = 0.0

    def sign(self, params: dict) -> str:
        """Everything is sent in the query string, so totalParams == query string, byte for byte."""
        qs = urlencode(params)  # uppercase %XX encoding, as MEXC requires for signing
        sig = hmac.new(self._secret, qs.encode(), hashlib.sha256).hexdigest()
        return f"{qs}&signature={sig}"

    async def sync_clock(self) -> None:
        t0 = time.time() * 1000
        data = await self.public("GET", "/api/v3/time")
        t1 = time.time() * 1000
        self.offset_ms = float(data["serverTime"]) - (t0 + t1) / 2

    async def public(self, method: str, path: str, params: Optional[dict] = None) -> Any:
        url = self._base + path + (("?" + urlencode(params)) if params else "")
        return await self._request(method, url, {})

    async def signed(self, method: str, path: str, params: Optional[dict] = None) -> Any:
        p = {k: v for k, v in (params or {}).items() if v is not None}
        p["recvWindow"] = self._recv
        p["timestamp"] = int(time.time() * 1000 + self.offset_ms)
        url = f"{self._base}{path}?{self.sign(p)}"
        return await self._request(method, url, {"X-MEXC-APIKEY": self._key})

    async def _request(self, method: str, url: str, headers: dict) -> Any:
        try:
            async with self._s.request(method, url, headers=headers, timeout=self._timeout) as resp:
                status, text = resp.status, await resp.text()
                retry_after = resp.headers.get("Retry-After", "1")
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            raise UnknownOutcome(f"{method} {url.split('?')[0]}: {e}") from e
        if status in (418, 429):
            raise RateLimited(status, text[:200], float(retry_after))
        if status >= 500:
            raise UnknownOutcome(f"HTTP {status}: {text[:200]}")
        try:
            data = json.loads(text) if text else {}
        except ValueError:
            raise MexcError(status, text[:200]) from None
        if status != 200:
            raise MexcError(status, data)
        if isinstance(data, dict) and "code" in data and data["code"] not in (0, 200):
            raise MexcError(status, data)
        return data

    # Endpoints used by the bot (Section 4.3)
    async def exchange_info(self, symbol: str) -> dict:
        return await self.public("GET", "/api/v3/exchangeInfo", {"symbol": symbol})

    async def trade_fee(self, symbol: str) -> dict:
        return await self.signed("GET", "/api/v3/tradeFee", {"symbol": symbol})

    async def self_symbols(self) -> dict:
        return await self.signed("GET", "/api/v3/selfSymbols")

    async def account(self) -> dict:
        return await self.signed("GET", "/api/v3/account")

    async def open_orders(self, symbol: str) -> list:
        return await self.signed("GET", "/api/v3/openOrders", {"symbol": symbol})

    async def query_order(self, symbol: str, cid: str) -> dict:
        return await self.signed("GET", "/api/v3/order", {"symbol": symbol, "origClientOrderId": cid})

    async def place_limit_maker(self, symbol: str, side: str, px: Decimal, qty: Decimal, cid: str,
                                stp_mode: Optional[str] = None) -> dict:
        return await self.signed("POST", "/api/v3/order", {
            "symbol": symbol, "side": side, "type": "LIMIT_MAKER",
            "quantity": fmt(qty), "price": fmt(px), "newClientOrderId": cid, "stpMode": stp_mode})

    async def place_ioc(self, symbol: str, side: str, px: Decimal, qty: Decimal, cid: str) -> dict:
        """Flatten path only (Section 3.3): pays the taker fee by design."""
        return await self.signed("POST", "/api/v3/order", {
            "symbol": symbol, "side": side, "type": "IMMEDIATE_OR_CANCEL",
            "quantity": fmt(qty), "price": fmt(px), "newClientOrderId": cid})

    async def cancel(self, symbol: str, cid: str) -> dict:
        return await self.signed("DELETE", "/api/v3/order", {"symbol": symbol, "origClientOrderId": cid})

    async def cancel_all(self, symbol: str) -> Any:
        return await self.signed("DELETE", "/api/v3/openOrders", {"symbol": symbol})

    async def listen_key_create(self) -> str:
        return (await self.signed("POST", "/api/v3/userDataStream"))["listenKey"]

    async def listen_key_keepalive(self, key: str) -> None:
        await self.signed("PUT", "/api/v3/userDataStream", {"listenKey": key})

    async def listen_key_close(self, key: str) -> None:
        await self.signed("DELETE", "/api/v3/userDataStream", {"listenKey": key})
```

### 11.4 Market data (Section 4.6) and signals (Section 3.1)

```python
@dataclass
class TopOfBook:
    bid: Decimal = ZERO
    ask: Decimal = ZERO
    bid_qty: Decimal = ZERO
    ask_qty: Decimal = ZERO
    send_time_ms: int = 0
    recv_ms: float = 0.0

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def micro(self) -> Decimal:
        tot = self.bid_qty + self.ask_qty
        return self.mid if tot <= ZERO else (self.bid * self.ask_qty + self.ask * self.bid_qty) / tot

    def fresh(self, stale_ms: float, now_ms: Optional[float] = None) -> bool:
        now_ms = time.time() * 1000 if now_ms is None else now_ms
        return self.bid > ZERO and self.ask > self.bid and (now_ms - self.recv_ms) < stale_ms


class MarketData:
    """Public WS client: decodes PushDataV3ApiWrapper, keeps top-of-book, trade flow and realized
    volatility, and sets `updated` for the strategy loop (Section 5.3). Callbacks do no I/O."""

    def __init__(self, url: str, symbol: str, vol_window_s: float = 60.0, flow_window_s: float = 5.0):
        self.url, self.symbol = url, symbol
        self.vol_window_s, self.flow_window_s = vol_window_s, flow_window_s
        self.book = TopOfBook()
        self.depth5: Any = None
        self.updated = asyncio.Event()
        self._mids: deque = deque()    # (t_s, mid)
        self._trades: deque = deque()  # (t_s, signed qty: + buy / - sell)

    def channels(self) -> list[str]:
        return [f"spot@public.aggre.bookTicker.v3.api.pb@100ms@{self.symbol}",
                f"spot@public.aggre.deals.v3.api.pb@100ms@{self.symbol}",
                f"spot@public.limit.depth.v3.api.pb@{self.symbol}@5"]

    async def run(self) -> None:
        backoff = 0.5
        while True:
            try:
                async with websockets.connect(self.url, ping_interval=None, max_size=2 ** 20) as ws:
                    await ws.send(json.dumps({"method": "SUBSCRIPTION", "params": self.channels()}))
                    pinger = asyncio.create_task(self.pinger(ws))
                    try:
                        async for frame in ws:
                            if isinstance(frame, (bytes, bytearray)):
                                self.on_binary(bytes(frame))
                            else:
                                log.debug("ws control: %s", frame)  # subscription acks, PONG
                    finally:
                        pinger.cancel()
            except Exception as e:  # reconnect on anything; the strategy loop sees stale data meanwhile
                log.warning("public ws disconnected: %s", e)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

    @staticmethod
    async def pinger(ws: Any) -> None:
        while True:
            await asyncio.sleep(20)
            await ws.send('{"method":"PING"}')

    def on_binary(self, raw: bytes) -> None:
        w = pb.PushDataV3ApiWrapper()
        w.ParseFromString(raw)
        now = time.time()
        kind = w.WhichOneof("body")
        if kind == "publicAggreBookTicker":
            m = w.publicAggreBookTicker
            self.book = TopOfBook(D(m.bidPrice), D(m.askPrice), D(m.bidQuantity), D(m.askQuantity),
                                  w.sendTime, now * 1000)
            self._mids.append((now, float(self.book.mid)))
            self._trim(now)
            self.updated.set()
        elif kind == "publicAggreDeals":
            for d in w.publicAggreDeals.deals:
                self._trades.append((d.time / 1000, float(d.quantity) * (1 if d.tradeType == 1 else -1)))
            self._trim(now)
        elif kind == "publicLimitDepths":
            self.depth5 = w.publicLimitDepths

    def _trim(self, now: float) -> None:
        while self._mids and now - self._mids[0][0] > self.vol_window_s:
            self._mids.popleft()
        while self._trades and now - self._trades[0][0] > self.flow_window_s:
            self._trades.popleft()

    def sigma_bp(self) -> float:
        """Std of 1 s mid log-returns over the window, in bp; 0 until there are enough samples."""
        samples: list[float] = []
        last_t = None
        for t, m in self._mids:
            if last_t is None or t - last_t >= 1.0:
                samples.append(m)
                last_t = t
        if len(samples) < 5:
            return 0.0
        rets = [math.log(b / a) for a, b in zip(samples, samples[1:]) if a > 0 and b > 0]
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
        return math.sqrt(var) * 1e4

    def flow(self) -> float:
        tot = sum(abs(q) for _, q in self._trades)
        return 0.0 if tot == 0 else sum(q for _, q in self._trades) / tot
```

### 11.5 Quote engine (Section 3.2, pure function)

```python
@dataclass(frozen=True)
class QuoteParams:
    order_notional: Decimal
    min_half_spread_ticks: int
    vol_mult: float
    skew_ticks: int
    max_inventory_base: Decimal
    max_spread_ticks: int


@dataclass(frozen=True)
class Quote:
    bid_px: Optional[Decimal]
    bid_qty: Decimal
    ask_px: Optional[Decimal]
    ask_qty: Decimal
    reason: str


def compute_quote(inst: Instrument, p: QuoteParams, book: TopOfBook, inv: Decimal, sigma_bp: float) -> Quote:
    """`inv` is base inventory relative to target. Returns None on a side that must not be quoted."""
    if book.bid <= ZERO or book.ask <= book.bid:
        return Quote(None, ZERO, None, ZERO, "crossed_or_empty")
    if (book.ask - book.bid) / inst.tick > p.max_spread_ticks:
        return Quote(None, ZERO, None, ZERO, "spread_too_wide")
    q_norm = max(D(-1), min(D(1), inv / p.max_inventory_base))
    ref = book.micro - q_norm * p.skew_ticks * inst.tick                       # lean away from inventory
    half = max(p.min_half_spread_ticks * inst.tick, D(repr(p.vol_mult * sigma_bp / 1e4)) * book.mid)
    bid_px = min(inst.round_price(ref - half, "BUY"), book.ask - inst.tick)    # post-only clamp
    ask_px = max(inst.round_price(ref + half, "SELL"), book.bid + inst.tick)
    bid_qty = inst.round_qty(p.order_notional / bid_px * (D(1) - max(ZERO, q_norm))) if bid_px > ZERO else ZERO
    ask_qty = inst.round_qty(p.order_notional / ask_px * (D(1) - max(ZERO, -q_norm)))
    bid = None if (q_norm >= 1 or inst.reject_reason(bid_px, bid_qty)) else bid_px
    ask = None if (q_norm <= -1 or inst.reject_reason(ask_px, ask_qty)) else ask_px
    return Quote(bid, bid_qty, ask, ask_qty, "ok")
```

### 11.6 Order state machine and order manager (Section 6)

```python
class S(str, Enum):
    DESIRED = "DESIRED"
    PENDING_NEW = "PENDING_NEW"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELED = "CANCELED"
    PARTIALLY_CANCELED = "PARTIALLY_CANCELED"
    REJECTED = "REJECTED"
    UNKNOWN_NEW = "UNKNOWN_NEW"
    UNKNOWN_CANCEL = "UNKNOWN_CANCEL"
    NEVER_PLACED = "NEVER_PLACED"


TERMINAL = {S.FILLED, S.CANCELED, S.PARTIALLY_CANCELED, S.REJECTED, S.NEVER_PLACED}
TRANSITIONS: dict[S, set[S]] = {
    S.DESIRED: {S.PENDING_NEW},
    S.PENDING_NEW: {S.OPEN, S.PARTIALLY_FILLED, S.FILLED, S.CANCELED, S.REJECTED, S.UNKNOWN_NEW},
    S.OPEN: {S.PARTIALLY_FILLED, S.FILLED, S.CANCEL_PENDING, S.CANCELED, S.PARTIALLY_CANCELED},
    S.PARTIALLY_FILLED: {S.FILLED, S.CANCEL_PENDING, S.CANCELED, S.PARTIALLY_CANCELED},
    S.CANCEL_PENDING: {S.FILLED, S.CANCELED, S.PARTIALLY_CANCELED, S.UNKNOWN_CANCEL},
    S.UNKNOWN_NEW: {S.OPEN, S.PARTIALLY_FILLED, S.FILLED, S.CANCELED, S.PARTIALLY_CANCELED, S.NEVER_PLACED},
    S.UNKNOWN_CANCEL: {S.OPEN, S.PARTIALLY_FILLED, S.FILLED, S.CANCELED, S.PARTIALLY_CANCELED},
}
PRIVATE_STATUS = {1: S.OPEN, 2: S.FILLED, 3: S.PARTIALLY_FILLED, 4: S.CANCELED, 5: S.PARTIALLY_CANCELED}
REST_STATUS = {"NEW": S.OPEN, "PARTIALLY_FILLED": S.PARTIALLY_FILLED, "FILLED": S.FILLED,
               "CANCELED": S.CANCELED, "PARTIALLY_CANCELED": S.PARTIALLY_CANCELED}


class IllegalTransition(RuntimeError):
    pass


@dataclass
class Order:
    cid: str
    side: str
    px: Decimal
    qty: Decimal
    state: S = S.DESIRED
    exchange_id: Optional[str] = None
    filled: Decimal = ZERO
    created_ms: float = field(default_factory=lambda: time.time() * 1000)
    updated_ms: float = 0.0

    def age_s(self) -> float:
        return (time.time() * 1000 - self.created_ms) / 1000

    def transition(self, new: S) -> None:
        if new == self.state:
            return
        if new not in TRANSITIONS.get(self.state, set()):
            raise IllegalTransition(f"{self.cid}: {self.state.value} -> {new.value}")
        self.state, self.updated_ms = new, time.time() * 1000


class OrderManager:
    """One live order per side; cancel-then-place; UNKNOWN resolution; private-stream application."""

    def __init__(self, rest: RestGateway, inst: Instrument, bucket: TokenBucket, session_tag: Optional[str] = None):
        self.rest, self.inst, self.bucket = rest, inst, bucket
        self.session_tag = session_tag or "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
        self.seq = 0
        self.live: dict[str, Optional[Order]] = {"BUY": None, "SELL": None}
        self.in_flight: dict[str, bool] = {"BUY": False, "SELL": False}
        self.by_cid: dict[str, Order] = {}
        self.on_fill: Optional[Callable[[Order, str, Decimal, Decimal, Decimal, str], None]] = None
        self.on_reject: Optional[Callable[[], None]] = None
        self.on_ack: Optional[Callable[[], None]] = None

    def next_cid(self) -> str:
        self.seq += 1
        return f"{self.session_tag}{self.seq:08d}"

    async def sync_side(self, side: str, desired_px: Optional[Decimal], desired_qty: Decimal,
                        tol_ticks: int, max_age_s: float) -> None:
        if self.in_flight[side]:
            return
        live = self.live[side]
        if live is not None and live.state in TERMINAL:
            self.live[side] = live = None
        if live is not None and live.state not in (S.OPEN, S.PARTIALLY_FILLED):
            return  # pending or unknown: wait for the ack, the stream, or reconciliation
        if live is not None:
            stale = (desired_px is None or abs(live.px - desired_px) >= tol_ticks * self.inst.tick
                     or live.age_s() > max_age_s)
            if stale:
                await self._cancel(live)  # the replacement is placed on a later pass, after the ack
            return
        if desired_px is not None:
            await self._place(side, desired_px, desired_qty)

    async def _place(self, side: str, px: Decimal, qty: Decimal) -> None:
        if not self.bucket.try_acquire():
            return
        o = Order(self.next_cid(), side, px, qty)
        o.transition(S.PENDING_NEW)
        self.by_cid[o.cid] = o
        self.live[side] = o
        self.in_flight[side] = True
        try:
            r = await self.rest.place_limit_maker(self.inst.symbol, side, px, qty, o.cid)
            o.exchange_id = str(r.get("orderId"))
            if o.state == S.PENDING_NEW:  # a fill may already have arrived on the private stream
                o.transition(S.OPEN)
            if self.on_ack:
                self.on_ack()
        except UnknownOutcome as e:
            o.transition(S.UNKNOWN_NEW)
            log.warning("place UNKNOWN %s: %s", o.cid, e)
            asyncio.create_task(self._resolve_unknown(o))
        except MexcError as e:  # includes a LIMIT_MAKER that would have crossed, filters, balance, 429
            o.transition(S.REJECTED)
            log.info("place rejected %s: %s", o.cid, e)
            if self.on_reject:
                self.on_reject()
            if isinstance(e, RateLimited):
                raise
        finally:
            self.in_flight[side] = False

    async def _cancel(self, o: Order) -> None:
        if not self.bucket.try_acquire(risk_reducing=True):
            return
        o.transition(S.CANCEL_PENDING)
        self.in_flight[o.side] = True
        try:
            await self.rest.cancel(self.inst.symbol, o.cid)
            if o.state == S.CANCEL_PENDING:
                o.transition(S.PARTIALLY_CANCELED if o.filled > ZERO else S.CANCELED)
        except UnknownOutcome as e:
            o.transition(S.UNKNOWN_CANCEL)
            log.warning("cancel UNKNOWN %s: %s", o.cid, e)
            asyncio.create_task(self._resolve_unknown(o))
        except MexcError as e:
            if e.code == -2011:  # unknown order: already filled or canceled; let the exchange tell us which
                o.transition(S.UNKNOWN_CANCEL)
                asyncio.create_task(self._resolve_unknown(o))
            else:
                log.error("cancel failed %s: %s", o.cid, e)
                if isinstance(e, RateLimited):
                    raise
        finally:
            self.in_flight[o.side] = False

    async def _resolve_unknown(self, o: Order) -> None:
        """Section 6.4: never resend blindly; ask the exchange what happened."""
        for _ in range(3):
            try:
                r: Optional[dict] = await self.rest.query_order(self.inst.symbol, o.cid)
            except MexcError as e:
                if e.code != -2011:
                    await asyncio.sleep(0.5)
                    continue
                r = None
            except UnknownOutcome:
                await asyncio.sleep(0.5)
                continue
            self._apply_query(o, r)
            return
        log.error("could not resolve %s after 3 queries; side %s stays blocked until reconciliation", o.cid, o.side)

    def _apply_query(self, o: Order, r: Optional[dict]) -> None:
        if r is None:  # the exchange has never seen it, or it is gone
            if o.state == S.UNKNOWN_NEW:
                o.transition(S.NEVER_PLACED)
            elif o.state not in TERMINAL:
                o.transition(S.PARTIALLY_CANCELED if o.filled > ZERO else S.CANCELED)
            return
        o.exchange_id = str(r.get("orderId", o.exchange_id))
        o.filled = max(o.filled, D(str(r.get("executedQty", "0"))))
        status = REST_STATUS.get(str(r.get("status")), S.OPEN)
        if o.state in TERMINAL:
            return
        o.transition(status)  # UNKNOWN_CANCEL -> OPEN means: still resting, cancel again on the next pass

    def on_private_order(self, m: Any) -> None:
        """PrivateOrdersV3Api (Section 4.7): status moves the state; quantities come from deals."""
        o = self.by_cid.get(m.clientId)
        if o is None:
            return  # orphan or previous session: reconciliation cancels it
        if m.id:
            o.exchange_id = m.id
        new = PRIVATE_STATUS.get(m.status)
        if new is None or o.state in TERMINAL:
            return
        if o.state in (S.CANCEL_PENDING, S.UNKNOWN_CANCEL) and new in (S.OPEN, S.PARTIALLY_FILLED):
            return  # a non-terminal status must not undo a pending cancel
        o.transition(new)

    def on_private_deal(self, m: Any) -> None:
        """PrivateDealsV3Api: the source of truth for fills and fees."""
        o = self.by_cid.get(m.clientOrderId)
        if o is None:
            return
        qty, px, fee = D(m.quantity), D(m.price), D(m.feeAmount)
        o.filled += qty
        if self.on_fill:
            self.on_fill(o, "BUY" if m.tradeType == 1 else "SELL", px, qty, fee, m.feeCurrency)

    async def cancel_all(self) -> None:
        """Risk-reducing: one request cancels every open order on the symbol."""
        while not self.bucket.try_acquire(risk_reducing=True):
            await asyncio.sleep(0.05)
        await self.rest.cancel_all(self.inst.symbol)
        for o in list(self.live.values()):
            if o is not None and o.state in (S.OPEN, S.PARTIALLY_FILLED):
                o.transition(S.CANCEL_PENDING)  # finalized by the private stream or reconciliation

    async def reconcile(self) -> None:
        """Section 6.5: the exchange view wins; orphans are canceled; stuck local orders are queried."""
        exchange_open = {str(x.get("clientOrderId")): x for x in await self.rest.open_orders(self.inst.symbol)}
        for cid in exchange_open:
            if cid not in self.by_cid:
                log.warning("orphan order %s on exchange; canceling", cid)
                if self.bucket.try_acquire(risk_reducing=True):
                    try:
                        await self.rest.cancel(self.inst.symbol, cid)
                    except (MexcError, UnknownOutcome) as e:
                        log.error("orphan cancel failed %s: %s", cid, e)
        for o in list(self.live.values()):
            if o is not None and o.state not in TERMINAL and o.cid not in exchange_open:
                await self._resolve_unknown(o)
```

### 11.7 Accounting and risk (Sections 3.6, 7)

```python
@dataclass
class Accounting:
    """Positions and PnL from private deals only; mark-to-market in the quote currency."""
    cash: Decimal
    inv: Decimal
    fees: Decimal = ZERO
    mtm0: Optional[Decimal] = None
    mtm_high: Optional[Decimal] = None
    fills: int = 0

    def on_fill(self, side: str, px: Decimal, qty: Decimal, fee_quote: Decimal) -> None:
        if side == "BUY":
            self.cash -= px * qty + fee_quote
            self.inv += qty
        else:
            self.cash += px * qty - fee_quote
            self.inv -= qty
        self.fees += fee_quote
        self.fills += 1

    def mtm(self, mid: Decimal) -> Decimal:
        return self.cash + self.inv * mid

    def session_pnl(self, mid: Decimal) -> Decimal:
        m = self.mtm(mid)
        if self.mtm0 is None:
            self.mtm0, self.mtm_high = m, m
        self.mtm_high = max(self.mtm_high or m, m)
        return m - self.mtm0

    def drawdown(self, mid: Decimal) -> Decimal:
        self.session_pnl(mid)
        return (self.mtm_high or ZERO) - self.mtm(mid)


@dataclass(frozen=True)
class RiskLimits:
    max_order_notional: Decimal
    daily_loss_limit: Decimal
    max_drawdown: Decimal
    max_dev_bp: Decimal
    hard_inventory_base: Decimal
    max_consecutive_rejects: int = 5


class RiskManager:
    """Section 7: pre-trade checks before every placement; post-fill checks after every fill."""

    def __init__(self, limits: RiskLimits, inst: Instrument, acct: Accounting):
        self.l, self.inst, self.acct = limits, inst, acct
        self.halted, self.halt_reason = False, ""
        self.consecutive_rejects = 0

    def halt(self, reason: str) -> None:
        if not self.halted:
            log.error("HALT: %s", reason)
        self.halted, self.halt_reason = True, reason

    def pre_trade(self, side: str, px: Decimal, qty: Decimal, book: TopOfBook) -> Optional[str]:
        if self.halted:
            return "halted"
        reason = self.inst.reject_reason(px, qty)
        if reason is not None:
            return "R14 " + reason
        if px * qty > self.l.max_order_notional:
            return "R3 notional"
        if abs(px - book.mid) / book.mid * 10_000 > self.l.max_dev_bp:
            return "R13 price deviates from mid"
        if (side == "BUY" and px >= book.ask) or (side == "SELL" and px <= book.bid):
            return "R13 would cross"
        if abs(self.acct.inv) >= self.l.hard_inventory_base:
            return "R6 hard inventory cap"
        return None

    def post_fill(self, mid: Decimal) -> None:
        pnl = self.acct.session_pnl(mid)
        if pnl <= -self.l.daily_loss_limit:
            self.halt(f"R7 daily loss limit: pnl={pnl}")
        elif self.acct.drawdown(mid) >= self.l.max_drawdown:
            self.halt(f"R8 drawdown: {self.acct.drawdown(mid)}")

    def on_reject(self) -> None:
        self.consecutive_rejects += 1
        if self.consecutive_rejects >= self.l.max_consecutive_rejects:
            self.halt("R9 reject storm")

    def on_ack(self) -> None:
        self.consecutive_rejects = 0
```

### 11.8 Private stream, strategy loop, wiring, selftest (Sections 4.7, 5.3, 6.6, 9.1)

```python
class UserData:
    """Private WS: listenKey lifecycle, keepalive every 30 min, reconnect; dispatches orders/deals/account."""

    def __init__(self, rest: RestGateway, ws_base: str, on_order: Callable[[Any], None],
                 on_deal: Callable[[Any], None], on_account: Callable[[Any], None]):
        self.rest, self.ws_base = rest, ws_base
        self.on_order, self.on_deal, self.on_account = on_order, on_deal, on_account
        self.connected = asyncio.Event()
        self.last_rx_ms = 0.0

    async def run(self) -> None:
        while True:
            key, keep = None, None
            try:
                key = await self.rest.listen_key_create()
                keep = asyncio.create_task(self._keepalive(key))
                async with websockets.connect(f"{self.ws_base}?listenKey={key}", ping_interval=None,
                                              max_size=2 ** 20) as ws:
                    await ws.send(json.dumps({"method": "SUBSCRIPTION", "params": [
                        "spot@private.orders.v3.api.pb", "spot@private.deals.v3.api.pb",
                        "spot@private.account.v3.api.pb"]}))
                    pinger = asyncio.create_task(MarketData.pinger(ws))
                    self.connected.set()
                    try:
                        async for frame in ws:
                            self.last_rx_ms = time.time() * 1000
                            if isinstance(frame, (bytes, bytearray)):
                                self.on_binary(bytes(frame))
                    finally:
                        pinger.cancel()
            except Exception as e:
                log.warning("private ws disconnected: %s", e)
            finally:
                self.connected.clear()
                if keep is not None:
                    keep.cancel()
                if key is not None:
                    try:
                        await self.rest.listen_key_close(key)
                    except Exception:
                        pass
            await asyncio.sleep(1.0)

    async def _keepalive(self, key: str) -> None:
        while True:
            await asyncio.sleep(1800)
            await self.rest.listen_key_keepalive(key)

    def on_binary(self, raw: bytes) -> None:
        w = pb.PushDataV3ApiWrapper()
        w.ParseFromString(raw)
        kind = w.WhichOneof("body")
        if kind == "privateOrders":
            self.on_order(w.privateOrders)
        elif kind == "privateDeals":
            self.on_deal(w.privateDeals)
        elif kind == "privateAccount":
            self.on_account(w.privateAccount)


@dataclass
class StrategyConfig:
    quote: QuoteParams
    requote_tol_ticks: int = 1
    max_quote_age_s: float = 3.0
    stale_ms: float = 1000.0
    vol_halt_bp: float = 25.0


async def strategy_loop(cfg: StrategyConfig, inst: Instrument, md: MarketData, om: OrderManager,
                        risk: RiskManager, acct: Accounting) -> None:
    """Act once per newest state; guards before quotes; cancel-first requotes (Sections 3.4, 3.5, 5.3)."""
    guard_active = False
    while True:
        try:
            await asyncio.wait_for(md.updated.wait(), timeout=0.25)  # the timeout lets the stale guard fire
        except asyncio.TimeoutError:
            pass
        md.updated.clear()
        book = md.book
        blocked = risk.halted or not book.fresh(cfg.stale_ms) or md.sigma_bp() > cfg.vol_halt_bp
        if blocked:
            if not guard_active:
                guard_active = True
                log.warning("guard on: halted=%s fresh=%s sigma_bp=%.1f", risk.halted,
                            book.fresh(cfg.stale_ms), md.sigma_bp())
                try:
                    await om.cancel_all()
                except (MexcError, UnknownOutcome) as e:
                    log.error("cancel_all during guard: %s", e)
            continue
        guard_active = False
        q = compute_quote(inst, cfg.quote, book, acct.inv, md.sigma_bp())
        for side, px, qty in (("BUY", q.bid_px, q.bid_qty), ("SELL", q.ask_px, q.ask_qty)):
            if px is not None and risk.pre_trade(side, px, qty, book) is not None:
                px = None
            try:
                await om.sync_side(side, px, qty, cfg.requote_tol_ticks, cfg.max_quote_age_s)
            except RateLimited as e:
                log.error("rate limited; backing off %.1fs", e.retry_after)
                await asyncio.sleep(e.retry_after)


async def main(symbol: str, api_key: str, api_secret: str, base_target: Decimal,
               quote: QuoteParams, limits: RiskLimits) -> None:
    async with aiohttp.ClientSession() as session:
        rest = RestGateway(session, "https://api.mexc.com", api_key, api_secret)
        await rest.sync_clock()
        s = (await rest.exchange_info(symbol))["symbols"][0]
        inst = Instrument.from_exchange_info(s)
        if "LIMIT_MAKER" not in s["orderTypes"] or not s.get("isSpotTradingAllowed"):
            raise SystemExit("R2: symbol is not quotable via API with LIMIT_MAKER")
        fee = (await rest.trade_fee(symbol))["data"]
        log.info("fees maker=%s taker=%s", fee["makerCommission"], fee["takerCommission"])
        await rest.cancel_all(symbol)  # startup: clean slate (Section 6.6)
        bal = {b["asset"]: b for b in (await rest.account())["balances"]}
        base_ccy, quote_ccy = s["baseAsset"], s["quoteAsset"]
        total = lambda a: D(bal[a]["free"]) + D(bal[a]["locked"]) if a in bal else ZERO  # noqa: E731
        acct = Accounting(cash=total(quote_ccy), inv=total(base_ccy) - base_target)
        risk = RiskManager(limits, inst, acct)
        om = OrderManager(rest, inst, TokenBucket(rate_per_s=8, burst=10, reserve=2))
        md = MarketData("wss://wbs-api.mexc.com/ws", symbol)

        def on_fill(o: Order, side: str, px: Decimal, qty: Decimal, fee_amt: Decimal, fee_ccy: str) -> None:
            fee_quote = fee_amt if fee_ccy == quote_ccy else ZERO  # convert MX-denominated fees (Section 3.6)
            acct.on_fill(side, px, qty, fee_quote)
            risk.post_fill(md.book.mid)
            log.info("fill %s %s %s@%s fee=%s %s inv=%s pnl=%s", o.cid, side, qty, px, fee_amt, fee_ccy,
                     acct.inv, acct.session_pnl(md.book.mid))

        om.on_fill, om.on_reject, om.on_ack = on_fill, risk.on_reject, risk.on_ack
        ud = UserData(rest, "wss://wbs-api.mexc.com/ws", om.on_private_order, om.on_private_deal, lambda a: None)

        async def reconcile_loop() -> None:
            while True:
                await asyncio.sleep(5)
                try:
                    await om.reconcile()
                except (MexcError, UnknownOutcome) as e:
                    log.warning("reconcile: %s", e)

        async def clock_loop() -> None:
            while True:
                await asyncio.sleep(60)
                try:
                    await rest.sync_clock()
                except (MexcError, UnknownOutcome) as e:
                    log.warning("clock sync: %s", e)

        try:
            await asyncio.gather(md.run(), ud.run(), strategy_loop(StrategyConfig(quote=quote), inst, md, om, risk, acct),
                                 reconcile_loop(), clock_loop())
        finally:
            await om.cancel_all()  # shutdown: never leave resting orders behind


def selftest() -> None:
    # 1. official signing vector (Section 9.1)
    g = RestGateway.__new__(RestGateway)
    g._secret = b"45d0b3c26f2644f19bfb98b07741b2f5"
    params = {"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT", "quantity": 1, "price": 11,
              "recvWindow": 5000, "timestamp": 1644489390087}
    assert g.sign(params).endswith("&signature=fd3e4e8543c5188531eb7279d68ae7d26a573d0fc5ab0d18eb692451654d837a")
    # 2. rounding and filters
    inst = Instrument("SOLUSDT", D("0.01"), D("0.001"), D("0.01"), D("5"), D("2000000"), ZERO, D("0.0005"))
    assert inst.round_price(D("199.996"), "BUY") == D("199.99")
    assert inst.round_price(D("199.991"), "SELL") == D("200.00")
    assert inst.round_qty(D("0.123456")) == D("0.123")
    assert fmt(D("1E+1")) == "10" and fmt(D("0.00000010")) == "0.00000010"
    assert inst.reject_reason(D("200"), D("0.001")) == "qty<min_qty"
    assert inst.reject_reason(D("200"), D("0.01")) == "notional<min"
    # 3. token bucket honours the rate and the reserve
    t = [0.0]
    b = TokenBucket(rate_per_s=8, burst=10, reserve=2, clock=lambda: t[0])
    assert sum(b.try_acquire() for _ in range(20)) == 8
    assert b.try_acquire(risk_reducing=True) and b.try_acquire(risk_reducing=True)
    assert not b.try_acquire(risk_reducing=True)
    t[0] = 1.0
    assert sum(b.try_acquire() for _ in range(20)) == 6
    # 4. quote invariants on random books
    p = QuoteParams(order_notional=D("20"), min_half_spread_ticks=1, vol_mult=0.5, skew_ticks=2,
                    max_inventory_base=D("1"), max_spread_ticks=10)
    rng = random.Random(7)
    for _ in range(2000):
        bid = D(rng.randint(15000, 25000)) / 100
        ask = bid + inst.tick * rng.randint(1, 8)
        book = TopOfBook(bid, ask, D(rng.randint(1, 500)) / 10, D(rng.randint(1, 500)) / 10, 0, time.time() * 1000)
        inv = D(rng.randint(-150, 150)) / 100
        q = compute_quote(inst, p, book, inv, rng.random() * 20)
        if q.bid_px is not None:
            assert q.bid_px <= book.ask - inst.tick and inst.reject_reason(q.bid_px, q.bid_qty) is None
        if q.ask_px is not None:
            assert q.ask_px >= book.bid + inst.tick and inst.reject_reason(q.ask_px, q.ask_qty) is None
        if q.bid_px is not None and q.ask_px is not None:
            assert q.bid_px < q.ask_px
        if inv >= p.max_inventory_base:
            assert q.bid_px is None
        if inv <= -p.max_inventory_base:
            assert q.ask_px is None
    book = TopOfBook(D("200.00"), D("200.01"), D("10"), D("10"), 0, time.time() * 1000)
    flat, long_ = compute_quote(inst, p, book, ZERO, 0.0), compute_quote(inst, p, book, D("0.5"), 0.0)
    assert long_.ask_px <= flat.ask_px and long_.bid_px <= flat.bid_px  # skew leans away from inventory
    # 5. state machine
    o = Order("x1", "BUY", D("1"), D("1"))
    for st in (S.PENDING_NEW, S.OPEN, S.CANCEL_PENDING, S.CANCELED):
        o.transition(st)
    try:
        o.transition(S.OPEN)
        raise AssertionError("terminal -> OPEN must fail")
    except IllegalTransition:
        pass
    # 6. accounting
    a = Accounting(cash=D("1000"), inv=ZERO)
    a.session_pnl(D("200"))
    a.on_fill("BUY", D("200"), D("0.1"), ZERO)
    a.on_fill("SELL", D("200.02"), D("0.1"), ZERO)
    assert a.session_pnl(D("200")) == D("0.002") and a.inv == ZERO
    print("selftest OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        print("usage: python mexc_scalper.py selftest   (live wiring: call main() from your launcher)")
```

---

## 12. Deployment and operations

**Server.** One small instance in AWS Tokyo or Singapore; `chrony` enabled; Python 3.11+;
the recorder (9.2) and the bot as separate `systemd` services with `Restart=always`,
`RestartSec=2`, and an `EnvironmentFile` (mode 0600) holding `MEXC_API_KEY` and
`MEXC_API_SECRET`. Logs to journald and a rotated JSON file; metrics scraped locally.

**API key.** Create a sub-account for the bot; permissions spot trade + account read only;
IP whitelist the server; rotate every 90 days regardless. Check `GET /api/v3/kyc/status`
and `GET /api/v3/selfSymbols` once after creation and keep the output with the config.

**Alerts (page a human).** Any halt (R7, R8, R9); any 418; more than 2 x 429 in 10 min;
reconciliation drift above `qty_step`; an UNKNOWN order unresolved for more than 10 s; more
than 3 WebSocket reconnects in 10 min; maker ratio below 99% over the last 100 fills;
session PnL below 50% of the daily loss limit.

**Runbooks.**

1. 429 storm: confirm there is exactly one process running against the key; lower
   `order_bucket_rate_per_s`; check the per-second request histogram for bursts from
   reconciliation coinciding with requotes.
2. R7 / R8 halt: do not restart the same day. Export the day's fills and markouts; compare
   `edge_bp(30 s)` by hour; decide whether the spread floor, the volatility halt, or the
   trading hours change before shadow mode resumes.
3. Reconciliation drift: compare `GET /api/v3/myTrades` for the session window against the
   `fills` table; insert the missing deals; restart only after the balances match.
4. Exchange maintenance or symbol status != online: cancel-all, stop, wait for
   `exchangeInfo.status` to return to online.

**Daily report.** Fills, maker ratio, `edge_bp` at each horizon with standard errors, PnL,
max drawdown, time at inventory cap, average and p99 request rate, guard activations with
durations, 429/418 count, reconnects, and the current gate status (2.5).

---

## Appendix A. Verified facts, sources, and what to re-verify

| Fact | Value used in this plan | Source | Status |
| --- | --- | --- | --- |
| REST base URL | `https://api.mexc.com` | official docs mirror (`mexcdevelop/apidocs`, branch `spot_v3_en`, changelog through 2025-02-24) | verified |
| Signature scheme | HMAC-SHA256 over `totalParams`, lowercase hex, header `X-MEXC-APIKEY` | same | verified; the docs' "Example 1" hash is inconsistent with its own input, "Example 2" (`fd3e4e85...`) is correct (checked with `openssl`) |
| Timing rule | `timestamp < serverTime + 1000` and `serverTime - timestamp <= recvWindow`; default 5000, max 60000 | same | verified |
| HTTP codes | 403 WAF, 429 rate limit, 418 ban, 5xx = UNKNOWN; `Retry-After` on 418/429; bans 2 min to 3 days | same | verified |
| Order types | `LIMIT, MARKET, LIMIT_MAKER, IMMEDIATE_OR_CANCEL, FILL_OR_KILL`; no `timeInForce` parameter | same | verified |
| Order statuses | `NEW, PARTIALLY_FILLED, FILLED, CANCELED, PARTIALLY_CANCELED`; private stream 1..5; `orderType` 1 LIMIT, 2 POST_ONLY, 3 IOC, 4 FOK, 5 MARKET, 100 stop | same | verified |
| `exchangeInfo` fields | `quotePrecision`, `baseAssetPrecision`, `baseSizePrecision` (min qty), `quoteAmountPrecision` (min notional), `maxQuoteAmount`, `orderTypes`, `isSpotTradingAllowed`, `makerCommission`, `takerCommission`, `tradeSideType` | same | verified field names; **VERIFY** live values for SOLUSDT |
| Cancel-all | `DELETE /api/v3/openOrders?symbol=` (up to 5 symbols) | same | verified |
| Account endpoint limit | 2 requests/s, weight 10 | same | verified |
| `myTrades` | last 1 month only, max 100 per call | same | verified |
| Order + cancel bucket | 12 requests/s per UID, shared by place, cancel, cancel-all | live changelog entries dated 2026-03-11, 2026-05-03, 2026-05-22 (seen via search; the live docs host was not reachable from this environment) | **VERIFY** on `mexc.com/api-docs/spot-v3/change-log` |
| Batch orders | `POST /api/v3/batchOrders`, 20 orders per request, 2 requests/s | mirror + live changelog 2025-12-24; a 2025 PR to the Chinese docs raised it to 30 | **VERIFY** current maximum |
| IP-weighted budget | mirror: 500 weight per 10 s per endpoint; live intro page: 300 weight per 10 s shared | search result from the live page | **VERIFY** |
| Open order cap | 500 unfilled orders per account | FAQ in the mirror | verified |
| Public WebSocket | `wss://wbs-api.mexc.com/ws`, protobuf `PushDataV3ApiWrapper`, `SUBSCRIPTION` / `UNSUBSCRIPTION` / `PING` JSON control messages, 30 subscriptions per connection, 100 msgs/s, 30 s / 60 s idle disconnects | mirror + `mexcdevelop/websocket-proto` (`main`) + `pymexc` 1.3.1 | verified |
| Channel names | `spot@public.aggre.bookTicker.v3.api.pb@(100ms|10ms)@SYM`, `spot@public.limit.depth.v3.api.pb@SYM@(5|10|20)`, `spot@public.aggre.deals.v3.api.pb@(100ms|10ms)@SYM`, `spot@public.aggre.depth.v3.api.pb@(100ms|10ms)@SYM`, `spot@private.orders.v3.api.pb`, `spot@private.deals.v3.api.pb`, `spot@private.account.v3.api.pb` | mirror | verified |
| Proto fields | as listed in 4.6 / 4.7 (`PublicAggreBookTickerV3Api`, `PublicLimitDepthsV3Api`, `PublicAggreDepthsV3Api`, `PublicAggreDealsV3Api`, `PrivateOrdersV3Api`, `PrivateDealsV3Api`, `PrivateAccountV3Api`) | `mexcdevelop/websocket-proto` | verified |
| Local book sync | `fromVersion == prev.toVersion + 1`; snapshot `limit=1000`; stale / gap rules | mirror | verified |
| listenKey | 60 min validity, `PUT` extends 60 min (send every 30 min), 60 keys per UID, 5 connections per key, 24 h connection cut | mirror | verified |
| Self-trade prevention | `stpMode` in `{"", cancel_maker, cancel_taker, cancel_both}`; `isSelfTrade` on deals | `pymexc` 1.3.1 (documents `stpMode`); mirror (deals field) | **VERIFY** `stpMode` on the live "New Order" page |
| Spot fees | 0% maker / 0.05% taker regular tier, 0.04% taker with MX deduction | multiple 2026 fee guides, including MEXC's own articles; the fee page itself was not reachable | **VERIFY** via `GET /api/v3/tradeFee?symbol=SOLUSDT` |
| Futures API | order endpoints marked "Under maintenance"; bulk order "not available for all users" | `mexcdevelop/apidocs`, branch `contract_v1_en` | verified |
| Hosting advice | Japan or Singapore AWS | FAQ in the mirror | verified |
| `LIMIT_MAKER` rejection code when it would cross | not documented; 30041 is the closest listed code | | **VERIFY** empirically with a minimum-size order |
| `newClientOrderId` length limit | not documented | | **VERIFY** empirically |

Re-verify before go-live, in this order: live `exchangeInfo` for the symbol; `tradeFee`;
the current order/cancel rate limit and batch maximum on the live changelog; `stpMode`
acceptance; the `LIMIT_MAKER` rejection behavior; `newClientOrderId` limits; measured REST
RTT and WebSocket lag from the chosen region.
