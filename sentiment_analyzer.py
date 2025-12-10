"""
Sentiment analysis module for OI Gemini.
Combines institutional flow (FII/DII) and market breadth (NIFTY 50/100) to gauge market sentiment.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analyzes market sentiment using FII/DII data and Market Breadth.
    """
    def __init__(self):
        self.fii_dii_cache: Dict[str, Any] = {}
        self.breadth_cache: Dict[str, Any] = {}
        self.last_breadth_update: Optional[datetime] = None
        self.last_fii_update: Optional[datetime] = None
        
        # Headers for NSE requests
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://www.nseindia.com/market-data/live-equity-market",
            "X-Requested-With": "XMLHttpRequest"
        }

    def fetch_market_breadth(self) -> Dict[str, Any]:
        """
        Fetches live market breadth for NIFTY 50 and NIFTY 100.
        Returns a dictionary with sentiment scores and metrics.
        """
        # Rate limit updates (e.g., every 1 minute)
        if self.last_breadth_update and (datetime.now() - self.last_breadth_update).total_seconds() < 60:
            return self.breadth_cache

        try:
            nifty50 = self._get_index_sentiment("50")
            nifty100 = self._get_index_sentiment("100")
            
            self.breadth_cache = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "nifty50": nifty50,
                "nifty100": nifty100,
                "overall_sentiment_score": (nifty50['sentiment_score_raw'] + nifty100['sentiment_score_raw']) / 2
            }
            self.last_breadth_update = datetime.now()
            return self.breadth_cache
        except Exception as e:
            LOGGER.error(f"Error fetching market breadth: {e}")
            return self.breadth_cache  # Return last known state

    def fetch_fii_dii(self) -> Dict[str, float]:
        """
        Fetches latest FII/DII daily net values.
        Updates only once per day or on restart.
        """
        today = datetime.now().date()
        if self.last_fii_update and self.last_fii_update.date() == today and self.fii_dii_cache:
            return self.fii_dii_cache

        try:
            # Attempt to fetch previous day's data (since today's comes EOD)
            # or today's provisional if available (NSE usually updates EOD)
            # We look back up to 3 days to find the last valid trading day
            for delta in range(0, 4):
                target_date = datetime.now() - timedelta(days=delta)
                data = self._fetch_fii_dii_for_date(target_date)
                if data:
                    self.fii_dii_cache = data
                    self.last_fii_update = datetime.now()
                    return data
            
            return {"fii_net": 0.0, "dii_net": 0.0}
        except Exception as e:
            LOGGER.error(f"Error fetching FII/DII data: {e}")
            return {"fii_net": 0.0, "dii_net": 0.0}

    def get_sentiment_metrics(self) -> Dict[str, float]:
        """
        Returns a flattened dictionary of sentiment metrics for feature engineering.
        """
        breadth = self.fetch_market_breadth()
        fii_dii = self.fetch_fii_dii()
        
        metrics = {
            'sentiment_ad_ratio_50': breadth.get('nifty50', {}).get('ad_ratio', 1.0),
            'sentiment_ad_ratio_100': breadth.get('nifty100', {}).get('ad_ratio', 1.0),
            'sentiment_trin_50': breadth.get('nifty50', {}).get('trin', 1.0),
            'sentiment_trin_100': breadth.get('nifty100', {}).get('trin', 1.0),
            'sentiment_score': breadth.get('overall_sentiment_score', 50.0),
            'sentiment_fii_net_crores': fii_dii.get('fii_net', 0.0),
            'sentiment_dii_net_crores': fii_dii.get('dii_net', 0.0),
            'sentiment_inst_net_crores': fii_dii.get('fii_net', 0.0) + fii_dii.get('dii_net', 0.0)
        }
        return metrics

    def _get_index_sentiment(self, index: str) -> Dict[str, Any]:
        """Helper to fetch and calculate sentiment for a specific index."""
        url = f"https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20{index}"
        
        try:
            # Need a session to handle cookies potentially, or just robust headers
            session = requests.Session()
            session.headers.update(self.headers)
            # Visit homepage to set cookies (NSE anti-scraping)
            session.get("https://www.nseindia.com", timeout=5)
            
            resp = session.get(url, timeout=10)
            if resp.status_code != 200:
                raise ValueError(f"Status {resp.status_code}")
                
            data = resp.json()
            
            adv_issues = int(data["advance"]["advances"])
            dec_issues = int(data["advance"]["declines"])
            total_value = float(data["data"][0]["totalTradedValue"]) # in rupees
            
            # Fetch Volume breakdown (optional, can be brittle)
            # Using proportional fallback for robustness if separate call fails
            adv_vol_cr = 0.0
            dec_vol_cr = 0.0
            
            # Helper to get vol from live analysis
            def get_vol(url_analysis):
                try:
                    t = session.get(url_analysis, timeout=5).text.strip()
                    # Logic from adsentiment.py
                    start = t.find('[')
                    end = t.rfind(']') + 1
                    array_str = t[start:end] if start != -1 and end > start else "[]"
                    items = json.loads(array_str)
                    key = "advData" if "advance" in url_analysis else "decData"
                    # Safe access
                    for item in items:
                        if isinstance(item, dict) and key in item:
                             return float(item[key][0].get("tradedValue", 0))
                    return 0.0
                except:
                    return 0.0

            try:
                adv_vol_cr = get_vol("https://www.nseindia.com/api/live-analysis-advance")
                dec_vol_cr = get_vol("https://www.nseindia.com/api/live-analysis-decline")
            except Exception:
                pass # Fallback to proportional

            total_issues = adv_issues + dec_issues
            if (adv_vol_cr + dec_vol_cr) == 0 and total_issues > 0:
                # Fallback
                adv_vol_cr = (total_value / 1e7) * (adv_issues / total_issues)
                dec_vol_cr = (total_value / 1e7) * (dec_issues / total_issues)

            # Calculations
            ad_ratio = adv_issues / dec_issues if dec_issues else 10.0
            vol_ratio = adv_vol_cr / dec_vol_cr if dec_vol_cr else 10.0
            combined_ratio = (ad_ratio + vol_ratio) / 2
            
            # 0 to 100 score
            score = 50 + 50 * (combined_ratio - 1) / (combined_ratio + 1)
            score = max(0, min(100, round(score, 1)))
            
            trin = round((adv_issues / dec_issues) / (adv_vol_cr / dec_vol_cr) if dec_vol_cr and adv_vol_cr else 1.0, 2)
            
            return {
                "ad_ratio": ad_ratio,
                "trin": trin,
                "sentiment_score_raw": score
            }
            
        except Exception as e:
            LOGGER.warning(f"Failed to fetch sentiment for NIFTY {index}: {e}")
            return {"ad_ratio": 1.0, "trin": 1.0, "sentiment_score_raw": 50.0}

    def _fetch_fii_dii_for_date(self, date_obj: datetime) -> Optional[Dict[str, float]]:
        """Fetch parsed FII/DII data for a specific date."""
        base = "https://archives.nseindia.com/content/equities/FIIDII_"
        date_str = date_obj.strftime("%d-%b-%Y").replace(" ", "")
        url = base + date_str + ".csv"
        
        try:
            df = pd.read_csv(url)
            # Expected columns: Category, Date, Buy Value, Sell Value, Net Value
            # Need to parse 'Net Value' for FII and DII rows
            
            fii_row = df[df['Category'].str.contains('FII|FPI', case=False, na=False)]
            dii_row = df[df['Category'].str.contains('DII', case=False, na=False)]
            
            fii_net = float(fii_row.iloc[0]['Net Value (Rs. Crores)']) if not fii_row.empty else 0.0
            dii_net = float(dii_row.iloc[0]['Net Value (Rs. Crores)']) if not dii_row.empty else 0.0
            
            return {"fii_net": fii_net, "dii_net": dii_net}
            
        except Exception:
            return None

