# When to Consider BULLISH vs BEARISH

**Source: database and codebase only.**

---

## 1. Definition from database (exact)

From **update_ml_features_from_multi_expiry.sql** and **database_new.py**:

- **oi_next_sentiment** (ml_features column) is defined as:
  - **oi_next_sentiment = nse_next_oi_change_put_total − nse_next_oi_change_call_total**
  - i.e. **PUT OI change − CALL OI change** (next-expiry; raw change, not %).

- **nse_next_oi_change_diff_put_call** is the same value:
  - `(total_oi_change_put_all_expiries - total_oi_change_call_all_expiries)` in SQL;
  - in **database_new.py** (line 2063): `oi_change_diff = (total_oi_change_put or 0.0) - (total_oi_change_call or 0.0)`; `oi_next_sentiment = oi_change_diff`.

So:
- **oi_next_sentiment > 0** ⇒ put OI change > call OI change ⇒ **puts building more** ⇒ **BEARISH**.
- **oi_next_sentiment < 0** ⇒ call OI change > put OI change ⇒ **calls building more** ⇒ **BULLISH**.

No other interpretation: the sign of oi_next_sentiment is defined by this formula only.

---

## 2. Rules derived only from the DB formula

| Consider   | Condition (from DB) |
|-----------|----------------------|
| **BULLISH** | **oi_next_sentiment < 0** (nse_next_oi_change_put_total < nse_next_oi_change_call_total → calls building more) |
| **BEARISH** | **oi_next_sentiment > 0** (nse_next_oi_change_put_total > nse_next_oi_change_call_total → puts building more) |

---

## 3. 3m % change (script-only, not DB columns)

The analysis script also computes **3-minute % change** from **OI levels** (nse_next_oi_call_total, nse_next_oi_put_total), not from the DB “change” columns. So:

- **pct_3m_oi_diff_ce_pe** = (CE OI 3m % change) − (PE OI 3m % change) from **levels**.
- **Positive** = CE (call) OI level rising more than PE (put) over 3m → same **direction** as “calls building” → **bullish**.
- **Negative** = PE (put) OI level rising more than CE (call) over 3m → same **direction** as “puts building” → **bearish**.

So for the script’s 3m diff (from levels), the rule is consistent with the DB: **CE > PE (diff > 0) ⇒ bullish bias; CE < PE (diff < 0) ⇒ bearish bias.**

---

## 4. Your trade outcomes (NSE)

- When at entry **oi_diff_CE_gt_PE** (CE 3m % > PE 3m % → bullish bias): 39.5% win rate.
- When at entry **oi_diff_CE_lt_PE** (CE 3m % < PE 3m % → bearish bias): 61.0% win rate.

So in your history, entries in bearish setups (CE < PE) did better; entries in bullish setups (CE > PE) did worse. The **definition** of bullish/bearish above is unchanged; only the **outcome** of when to favor trades differs.

---

## 5. Quick reference (DB-only rule)

| Consider   | DB condition |
|-----------|---------------|
| **BULLISH** | **oi_next_sentiment < 0** (put_change − call_change < 0) |
| **BEARISH** | **oi_next_sentiment > 0** (put_change − call_change > 0) |

Formula (from DB/code): **oi_next_sentiment = nse_next_oi_change_put_total − nse_next_oi_change_call_total**.
