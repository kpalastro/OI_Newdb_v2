# File: nifty_sentiment.py
import requests
from datetime import datetime
import json

def get_market_sentiment(index="50"):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://www.nseindia.com/market-data/live-equity-market",
        "X-Requested-With": "XMLHttpRequest"
    }

    # Step 1: Get A/D + total value
    url = f"https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20{index}"
    data = requests.get(url, headers=headers, timeout=15).json()

    adv_issues = int(data["advance"]["advances"])
    dec_issues = int(data["advance"]["declines"])
    total_value = float(data["data"][0]["totalTradedValue"])  # in ₹

    # Step 2: Get actual advancing/declining volume (₹ Cr) from live analysis
    def get_volume_cr(url):
        try:
            text = requests.get(url, headers=headers, timeout=15).text.strip()
            start = text.find('[')
            end = text.rfind(']') + 1
            array_str = text[start:end] if start != -1 and end > start else "[]"
            items = json.loads(array_str)
            key = "advData" if "advance" in url else "decData"
            data_obj = next((x for x in items if isinstance(x, dict) and key in x), {})
            volume_cr = float(data_obj.get(key, [{}])[0].get("tradedValue", 0)) if data_obj.get(key) else 0
            return volume_cr
        except:
            return 0

    adv_vol_cr = get_volume_cr("https://www.nseindia.com/api/live-analysis-advance")
    dec_vol_cr = get_volume_cr("https://www.nseindia.com/api/live-analysis-decline")

    # Fallback: Proportional split from total traded value
    total_issues = adv_issues + dec_issues
    if (adv_vol_cr + dec_vol_cr) == 0 and total_issues > 0:
        adv_vol_cr = total_value / 10000000 * (adv_issues / total_issues)
        dec_vol_cr = total_value / 10000000 * (dec_issues / total_issues)

    # Calculations
    ad_ratio = adv_issues / dec_issues if dec_issues else 999
    vol_ratio = adv_vol_cr / dec_vol_cr if dec_vol_cr else 999
    combined_ratio = (ad_ratio + vol_ratio) / 2

    sentiment_score = 50 + 50 * (combined_ratio - 1) / (combined_ratio + 1)
    sentiment_score = max(0, min(100, round(sentiment_score, 1)))

    trin = round((adv_issues / dec_issues) / (adv_vol_cr / dec_vol_cr) if dec_vol_cr else 1.0, 2)
    confidence = 90 if abs(trin - 1) < 0.3 else 75 if abs(trin - 1) < 0.7 else 60

    sentiment_label = "STRONG BULLISH" if sentiment_score > 70 else "BULLISH" if sentiment_score > 55 else "NEUTRAL" if sentiment_score > 45 else "BEARISH"

    result = {
        "index": f"NIFTY {index}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Advances": adv_issues,
        "Declines": dec_issues,
        "Adv Volume (₹ Cr)": round(adv_vol_cr),
        "Dec Volume (₹ Cr)": round(dec_vol_cr),
        "Sentiment Score": f"{sentiment_score}/100 → {sentiment_label}",
        "TRIN": trin,
        "Confidence": f"{confidence}%"
    }

    # Pretty Terminal Output
    print(f"\n{'='*56}")
    print(f"    LIVE MARKET SENTIMENT - NIFTY {index} ")
    print(f"{'='*56}")
    for k, v in result.items():
        if k != "index" and k != "timestamp":
            print(f"{k:20}: {v}")
    print(f"{'='*56}\n")

    return result

# MAIN: Run both NIFTY 50 and NIFTY 100
if __name__ == "__main__":
    print(f"Fetching live data @ {datetime.now().strftime('%d-%b-%Y %H:%M:%S')}\n")
    
    nifty50 = get_market_sentiment("50")
    nifty100 = get_market_sentiment("100")

    # Final JSON output (copy-paste ready)
    final_json = {
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "NIFTY50": nifty50,
        "NIFTY100": nifty100
    }

    print(json.dumps(final_json, indent=2))