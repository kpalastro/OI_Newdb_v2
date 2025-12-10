import logging
import requests
import csv
import io
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

# Constants
NSE_FII_JSON_URL = "https://www.nseindia.com/api/fiidiiTradeReact"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/reports/fii-dii"
}

def _fetch_fii_dii_nse_json(session) -> Optional[Dict[str, float]]:
    """
    Fetch FII/DII data from NSE API (JSON format).
    
    API Response format:
    [
        {"category":"DII **","date":"04-Dec-2025","buyValue":"16489.46","sellValue":"12828.41","netValue":"3661.05"},
        {"category":"FII/FPI *","date":"04-Dec-2025","buyValue":"11500.09","sellValue":"13444.28","netValue":"-1944.19"}
    ]
    
    Returns:
        Dictionary with fii_flow, dii_flow, and fii_dii_net (sum of both)
    """
    try:
        # NSE requires a visit to the homepage or report page first to set cookies
        session.get("https://www.nseindia.com/reports/fii-dii", headers=HEADERS, timeout=5)
        
        response = session.get(NSE_FII_JSON_URL, headers=HEADERS, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            fii_flow = 0.0
            dii_flow = 0.0
            
            # Extract netValue from JSON response
            # Format: [{"category":"DII **", "netValue":"3661.05"}, {"category":"FII/FPI *", "netValue":"-1944.19"}]
            for item in data:
                category = item.get('category', '')
                try:
                    net_val = float(item.get('netValue', 0))
                    if "FII/FPI" in category:
                        fii_flow = net_val
                    elif "DII" in category:
                        dii_flow = net_val
                except (ValueError, TypeError):
                    continue
            
            # Calculate fii_dii_net as sum of both flows
            fii_dii_net = fii_flow + dii_flow
            
            return {
                'fii_flow': fii_flow,
                'dii_flow': dii_flow,
                'fii_dii_net': fii_dii_net
            }
    except Exception as e:
        logging.debug(f"NSE API JSON fetch failed: {e}")
    return None

def fetch_fii_dii_data() -> Dict[str, float]:
    """
    Fetch FII/DII provisional data from NSE India API.
    
    Strategy: Try NSE JSON API endpoint.
    
    Returns:
        Dictionary with:
        - fii_flow: Net FII/FPI flow (from netValue field)
        - dii_flow: Net DII flow (from netValue field)
        - fii_dii_net: Sum of fii_flow and dii_flow
    """
    try:
        session = requests.Session()
        
        # Attempt 1: NSE JSON API
        data = _fetch_fii_dii_nse_json(session)
        if data:
            logging.info(f"✓ Macro Data (NSE API): FII Flow: {data['fii_flow']} Cr, DII Flow: {data['dii_flow']} Cr, Net: {data['fii_dii_net']} Cr")
            return data
            
        logging.warning("⚠ Could not fetch FII/DII data. Using neutral bias.")
        return {'fii_flow': 0.0, 'dii_flow': 0.0, 'fii_dii_net': 0.0}

    except Exception as e:
        logging.error(f"FII/DII fetch error: {e}")
        return {'fii_flow': 0.0, 'dii_flow': 0.0, 'fii_dii_net': 0.0}

def find_macro_tokens(kite_client) -> Dict[str, int]:
    """
    Find instrument tokens for active USDINR and CRUDEOIL futures.
    Logic:
    - USDINR (CDS): Exclude weekly contracts (regex check). Pick NEXT MONTH expiry (Month 2).
    - CRUDEOIL (MCX): Pick NEXT MONTHLY expiry (Front Month).
    """
    macro_tokens = {}
    try:
        # 1. USDINR (NSE-CDS)
        try:
            instruments_cds = kite_client.instruments('CDS')
            logging.info(f"[MACRO_TOKENS] CDS instruments fetched: {len(instruments_cds)}")
            # Filter for USDINR Futures
            usdinr_futs = [
                i for i in instruments_cds 
                if i['name'] == 'USDINR' and i['instrument_type'] == 'FUT'
            ]
            logging.info(f"[MACRO_TOKENS] USDINR FUT count: {len(usdinr_futs)}")
            
            # Filter out weekly contracts
            # Monthly symbols: "USDINR25NOVFUT" (Standard format)
            # Weekly symbols: "USDINR25N21FUT", "USDINR25D05FUT" (Contain specific dates/week codes)
            monthly_futs = []
            for i in usdinr_futs:
                symbol = i['tradingsymbol']
                # Regex: Look for MMMFUT pattern (e.g., NOVFUT, DECFUT) avoiding digits in the month part
                if re.search(r'[A-Z]{3}FUT$', symbol):
                    monthly_futs.append(i)
            logging.info(f"[MACRO_TOKENS] USDINR monthly FUT count: {len(monthly_futs)}")
            
            if monthly_futs:
                monthly_futs.sort(key=lambda x: x['expiry'])
                
                # User rule: "Always use next month expiry" for USDINR
                # Index 0 = Current Month (e.g., Nov), Index 1 = Next Month (e.g., Dec)
                if len(monthly_futs) > 1:
                    target = monthly_futs[1]  # Next Month
                    desc = "Next Month"
                else:
                    target = monthly_futs[0]  # Fallback to Current
                    desc = "Current Month"
                
                macro_tokens['USDINR'] = target['instrument_token']
                logging.info(f"✓ Found USDINR Future ({desc}): {target['tradingsymbol']} ({target['instrument_token']})")
            else:
                logging.warning("[MACRO_TOKENS] No monthly USDINR futures found in CDS instruments.")
                
        except Exception as e:
            logging.warning(f"[MACRO_TOKENS] USDINR search failed: {e}")

        # 2. CRUDEOIL (MCX)
        try:
            instruments_mcx = kite_client.instruments('MCX')
            logging.info(f"[MACRO_TOKENS] MCX instruments fetched: {len(instruments_mcx)}")
            crude_futs = [
                i for i in instruments_mcx 
                if i['name'] == 'CRUDEOIL' and i['instrument_type'] == 'FUT'
            ]
            logging.info(f"[MACRO_TOKENS] CRUDEOIL FUT count: {len(crude_futs)}")
            if crude_futs:
                crude_futs.sort(key=lambda x: x['expiry'])
                
                # User rule: "Always use next Monthly expiry" 
                # In commodities, "Next Monthly" typically means the Front Month (most active)
                # unless it's very close to expiry. Assuming Front Month (Index 0) here.
                target = crude_futs[0]
                
                macro_tokens['CRUDEOIL'] = target['instrument_token']
                logging.info(f"✓ Found CRUDEOIL Future (Front): {target['tradingsymbol']} ({target['instrument_token']})")
            else:
                logging.warning("[MACRO_TOKENS] No CRUDEOIL futures found in MCX instruments.")
        except Exception as e:
            logging.warning(f"[MACRO_TOKENS] CRUDEOIL search failed: {e}")

    except Exception as e:
        logging.error(f"Error finding macro tokens: {e}")
    
    return macro_tokens


def fetch_nifty_sentiment() -> Dict[str, float]:
    """
    Fetch NIFTY50 and NIFTY100 sentiment data using adsentiment.py logic.
    
    Returns:
        Dictionary with:
        - sentiment_score_50: NIFTY50 sentiment score (0-100)
        - sentiment_confidence_50: NIFTY50 confidence (0-100)
        - trin_50: NIFTY50 TRIN value
        - sentiment_score_100: NIFTY100 sentiment score (0-100)
        - sentiment_confidence_100: NIFTY100 confidence (0-100)
        - trin_100: NIFTY100 TRIN value
    """
    try:
        from adsentiment import get_market_sentiment
        
        # Fetch sentiment for both indices
        logging.info("[NIFTY_SENTIMENT] Calling get_market_sentiment for NIFTY50...")
        nifty50_data = get_market_sentiment("50")
        logging.info(f"[NIFTY_SENTIMENT] NIFTY50 data received: {type(nifty50_data)}, keys: {list(nifty50_data.keys()) if isinstance(nifty50_data, dict) else 'Not a dict'}")
        
        logging.info("[NIFTY_SENTIMENT] Calling get_market_sentiment for NIFTY100...")
        nifty100_data = get_market_sentiment("100")
        logging.info(f"[NIFTY_SENTIMENT] NIFTY100 data received: {type(nifty100_data)}, keys: {list(nifty100_data.keys()) if isinstance(nifty100_data, dict) else 'Not a dict'}")
        
        # Extract numeric values from formatted strings
        def extract_sentiment_score(score_str: str) -> float | None:
            """Extract numeric value from '77.6/100 → STRONG BULLISH' format."""
            try:
                if not score_str:
                    return None
                # Extract number before "/100"
                match = re.search(r'([\d.]+)/100', str(score_str))
                if match:
                    return float(match.group(1))
            except (ValueError, AttributeError, TypeError) as e:
                logging.debug(f"Failed to extract sentiment score from '{score_str}': {e}")
            return None
        
        def extract_confidence(conf_str: str) -> float | None:
            """Extract numeric value from '90%' format."""
            try:
                if not conf_str:
                    return None
                # Extract number before "%"
                match = re.search(r'([\d.]+)%', str(conf_str))
                if match:
                    return float(match.group(1))
            except (ValueError, AttributeError, TypeError) as e:
                logging.debug(f"Failed to extract confidence from '{conf_str}': {e}")
            return None
        
        # Debug: Log raw data received
        logging.debug(f"[NIFTY_SENTIMENT] N50 raw data: {nifty50_data}")
        logging.debug(f"[NIFTY_SENTIMENT] N100 raw data: {nifty100_data}")
        
        # Extract TRIN values (handle both float and string)
        def safe_float_trin(value) -> float | None:
            try:
                if value is None:
                    return None
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    # Try to extract numeric value
                    match = re.search(r'([\d.]+)', str(value))
                    if match:
                        return float(match.group(1))
            except (ValueError, TypeError):
                pass
            return None
        
        result = {
            'sentiment_score_50': extract_sentiment_score(nifty50_data.get('Sentiment Score')),
            'sentiment_confidence_50': extract_confidence(nifty50_data.get('Confidence')),
            'trin_50': safe_float_trin(nifty50_data.get('TRIN')),
            'sentiment_score_100': extract_sentiment_score(nifty100_data.get('Sentiment Score')),
            'sentiment_confidence_100': extract_confidence(nifty100_data.get('Confidence')),
            'trin_100': safe_float_trin(nifty100_data.get('TRIN')),
        }
        
        # Log extracted values for debugging
        logging.debug(f"[NIFTY_SENTIMENT] Extracted values: {result}")
        
        # Log extracted values (handle None values safely)
        score50_str = f"{result['sentiment_score_50']:.1f}" if result['sentiment_score_50'] is not None else "None"
        conf50_str = f"{result['sentiment_confidence_50']:.0f}%" if result['sentiment_confidence_50'] is not None else "None"
        trin50_str = f"{result['trin_50']:.2f}" if result['trin_50'] is not None else "None"
        score100_str = f"{result['sentiment_score_100']:.1f}" if result['sentiment_score_100'] is not None else "None"
        conf100_str = f"{result['sentiment_confidence_100']:.0f}%" if result['sentiment_confidence_100'] is not None else "None"
        trin100_str = f"{result['trin_100']:.2f}" if result['trin_100'] is not None else "None"
        
        logging.info(
            f"✓ NIFTY Sentiment: N50 Score={score50_str} "
            f"(Conf={conf50_str}, TRIN={trin50_str}), "
            f"N100 Score={score100_str} "
            f"(Conf={conf100_str}, TRIN={trin100_str})"
        )
        
        return result
        
    except ImportError:
        logging.warning("adsentiment module not found. NIFTY sentiment data unavailable.")
        return {
            'sentiment_score_50': None,
            'sentiment_confidence_50': None,
            'trin_50': None,
            'sentiment_score_100': None,
            'sentiment_confidence_100': None,
            'trin_100': None,
        }
    except Exception as e:
        logging.error(f"Failed to fetch NIFTY sentiment: {e}", exc_info=True)
        return {
            'sentiment_score_50': None,
            'sentiment_confidence_50': None,
            'trin_50': None,
            'sentiment_score_100': None,
            'sentiment_confidence_100': None,
            'trin_100': None,
        }
