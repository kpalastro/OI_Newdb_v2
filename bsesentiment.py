# File: bsesentiment.py
"""
BSE Sentiment Fetcher
Fetches market breadth data for BSE 100 and BSE 200 indices from BSE API.
"""
import requests
from datetime import datetime
import json
import logging

LOGGER = logging.getLogger(__name__)


def get_bse_sentiment():
    """
    Fetch BSE 100 and BSE 200 sentiment data from BSE API.
    
    Returns:
        Dictionary with:
        - bse_100: BSE 100 sentiment data
        - bse_200: BSE 200 sentiment data
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://www.bseindia.com/",
        "Accept": "application/json"
    }
    
    url = "https://api.bseindia.com/BseIndiaAPI/api/advanceDeclineTO/w?val=Index&TOFlag=1"
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if not isinstance(data, list):
            LOGGER.error(f"[BSE_SENTIMENT] Unexpected response format: {type(data)}")
            return None
        
        # Find BSE 100 and BSE 200 in the response
        bse_100_data = None
        bse_200_data = None
        
        for item in data:
            sens_ind = item.get("Sens_ind", "")
            if sens_ind == "BSE 100":
                bse_100_data = item
            elif sens_ind == "BSE 200":
                bse_200_data = item
        
        result = {}
        
        # Process BSE 100
        if bse_100_data:
            result["bse_100"] = _calculate_sentiment(bse_100_data, "BSE 100")
        else:
            LOGGER.warning("[BSE_SENTIMENT] BSE 100 data not found in API response")
            result["bse_100"] = None
        
        # Process BSE 200
        if bse_200_data:
            result["bse_200"] = _calculate_sentiment(bse_200_data, "BSE 200")
        else:
            LOGGER.warning("[BSE_SENTIMENT] BSE 200 data not found in API response")
            result["bse_200"] = None
        
        return result
        
    except requests.exceptions.RequestException as e:
        LOGGER.error(f"[BSE_SENTIMENT] API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        LOGGER.error(f"[BSE_SENTIMENT] Failed to parse JSON response: {e}")
        return None
    except Exception as e:
        LOGGER.error(f"[BSE_SENTIMENT] Unexpected error: {e}", exc_info=True)
        return None


def _calculate_sentiment(index_data: dict, index_name: str) -> dict:
    """
    Calculate sentiment score from BSE index data.
    
    Args:
        index_data: Dictionary with UP, DN, TOTAL, Advance_PER, Decline_PER
        index_name: Name of the index (e.g., "BSE 100")
    
    Returns:
        Dictionary with sentiment metrics
    """
    try:
        up = int(index_data.get("UP", 0))
        dn = int(index_data.get("DN", 0))
        total = int(index_data.get("TOTAL", 0))
        advance_per = float(index_data.get("Advance_PER", "0.0").replace("%", ""))
        decline_per = float(index_data.get("Decline_PER", "0.0").replace("%", ""))
        
        # Calculate A/D ratio
        if dn > 0:
            ad_ratio = up / dn
        else:
            ad_ratio = 999 if up > 0 else 1.0
        
        # Calculate sentiment score using same formula as NSE
        # Since we don't have volume data, use A/D ratio only
        sentiment_score = 50 + 50 * (ad_ratio - 1) / (ad_ratio + 1)
        sentiment_score = max(0, min(100, round(sentiment_score, 1)))
        
        # TRIN calculation (without volume, set to 1.0 as neutral)
        # TRIN = (Advancing Issues / Declining Issues) / (Advancing Volume / Declining Volume)
        # Since we don't have volume, we'll use a simplified approach
        trin = 1.0  # Neutral since no volume data available
        
        # Confidence based on A/D ratio strength
        if ad_ratio > 2.0 or ad_ratio < 0.5:
            confidence = 85  # Strong signal
        elif ad_ratio > 1.5 or ad_ratio < 0.67:
            confidence = 75  # Moderate signal
        else:
            confidence = 65  # Weak signal
        
        sentiment_label = (
            "STRONG BULLISH" if sentiment_score > 70 else
            "BULLISH" if sentiment_score > 55 else
            "NEUTRAL" if sentiment_score > 45 else
            "BEARISH"
        )
        
        return {
            "index": index_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Advances": up,
            "Declines": dn,
            "Unchanged": int(index_data.get("UC", 0)),
            "Total": total,
            "Advance %": f"{advance_per:.2f}",
            "Decline %": f"{decline_per:.2f}",
            "Sentiment Score": f"{sentiment_score}/100 → {sentiment_label}",
            "TRIN": trin,
            "Confidence": f"{confidence}%",
            # Raw values for extraction
            "_sentiment_score": sentiment_score,
            "_confidence": confidence,
            "_trin": trin
        }
        
    except (ValueError, KeyError, TypeError) as e:
        LOGGER.error(f"[BSE_SENTIMENT] Error calculating sentiment for {index_name}: {e}")
        return None


# MAIN: Test the function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Fetching BSE sentiment data @ {datetime.now().strftime('%d-%b-%Y %H:%M:%S')}\n")
    
    result = get_bse_sentiment()
    
    if result:
        print("\n" + "="*60)
        print("    BSE MARKET SENTIMENT")
        print("="*60)
        
        if result.get("bse_100"):
            print("\nBSE 100:")
            for k, v in result["bse_100"].items():
                if not k.startswith("_"):
                    print(f"  {k:20}: {v}")
        
        if result.get("bse_200"):
            print("\nBSE 200:")
            for k, v in result["bse_200"].items():
                if not k.startswith("_"):
                    print(f"  {k:20}: {v}")
        
        print("="*60 + "\n")
        
        # JSON output
        print(json.dumps({
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "BSE_100": result.get("bse_100"),
            "BSE_200": result.get("bse_200")
        }, indent=2, default=str))
    else:
        print("Failed to fetch BSE sentiment data")
