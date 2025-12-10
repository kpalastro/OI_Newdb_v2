"""
Feature Worker Process for Distributed Processing.

Offloads heavy feature engineering and ML inference to a separate process
to keep the main WebSocket thread responsive.
"""
import logging
import multiprocessing
import time
import traceback
from typing import Dict, Any, Optional
from datetime import datetime

# Import core modules - note: these imports happen inside the process 
# or must be pickle-safe if top-level. 
# For safety in MP, we often import inside run() or ensure module state isn't shared.
# However, standard imports usually work fine on fork (Linux) but spawn (Windows) requires care.
# Since user is on Windows (implied by paths), we must ensure imports don't trigger side effects.

from feature_engineering import engineer_live_feature_set, FeatureEngineeringError
from ml_core import MLSignalGenerator
from execution.strategy_router import AdvancedStrategyRouter

LOGGER = logging.getLogger(__name__)

class FeatureWorkerProcess(multiprocessing.Process):
    """
    Dedicated worker process for feature calculation and signal generation.
    """
    def __init__(
        self,
        exchange: str,
        input_queue: multiprocessing.Queue,
        output_queue: multiprocessing.Queue,
        shutdown_event: multiprocessing.Event
    ):
        super().__init__(name=f"FeatureWorker-{exchange}")
        self.exchange = exchange
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.shutdown_event = shutdown_event
        self.daemon = True  # Ensure it dies if parent dies hard
        
    def run(self):
        """
        Main loop for the worker process.
        """
        # Initialize resources that shouldn't be shared across processes
        # or need to be recreated in the new process space (especially on Windows)
        LOGGER.info(f"[{self.exchange}] FeatureWorkerProcess started. PID: {multiprocessing.current_process().pid}")
        
        try:
            # Load ML Models here to keep them in this process's memory
            self.ml_generator = MLSignalGenerator(self.exchange)
            self.strategy_router = AdvancedStrategyRouter(self.exchange)
            
            LOGGER.info(f"[{self.exchange}] Models loaded in worker process.")
            
            while not self.shutdown_event.is_set():
                try:
                    # Blocking get with timeout to allow checking shutdown flag
                    task = self.input_queue.get(timeout=1.0)
                    
                    if task is None:
                        # Sentinel for shutdown
                        break
                        
                    self._process_task(task)
                    
                except multiprocessing.queues.Empty:
                    continue
                except Exception as e:
                    LOGGER.error(f"[{self.exchange}] Error in feature worker loop: {e}")
                    # Don't crash the worker, just log and continue
                    time.sleep(0.1)
                    
        except Exception as e:
            LOGGER.critical(f"[{self.exchange}] Critical failure in FeatureWorkerProcess: {e}\n{traceback.format_exc()}")
        finally:
            LOGGER.info(f"[{self.exchange}] FeatureWorkerProcess shutting down.")

    def _process_task(self, task: Dict[str, Any]):
        """
        Process a single tick/snapshot task.
        
        Expected task keys:
        - type: 'tick'
        - payload: { ... data needed for engineering ... }
        """
        task_type = task.get('type')
        
        if task_type == 'tick':
            self._handle_tick_update(task.get('payload', {}))
        else:
            LOGGER.warning(f"[{self.exchange}] Unknown task type: {task_type}")

    def _handle_tick_update(self, payload: Dict[str, Any]):
        """
        Execute feature engineering and ML prediction.
        """
        start_time = time.time()
        
        # Unpack payload - this needs to match what AppManager sends
        # We need Handler state proxies or just the raw data.
        # Passing complex objects (Handler) via Queue is bad (pickling issues).
        # Better to pass the raw data structures: reels, option_chain, etc.
        
        # For Phase 2, we assume the payload contains snapshots of required data.
        # This might be heavy, but it ensures isolation.
        
        try:
            # extract data
            call_options = payload.get('call_options', [])
            put_options = payload.get('put_options', [])
            spot_price = payload.get('spot_price')
            atm_strike = payload.get('atm_strike')
            timestamp = payload.get('timestamp') # datetime object
            latest_vix = payload.get('latest_vix')
            
            # Since engineer_live_feature_set expects a 'handler' object for reels and history,
            # we need a way to provide that context without passing the full handler.
            # We can create a lightweight proxy or dataclass.
            
            handler_proxy = HandlerProxy(payload.get('handler_state', {}))
            
            # 1. Feature Engineering
            features = engineer_live_feature_set(
                handler=handler_proxy,
                call_options=call_options,
                put_options=put_options,
                spot_price=spot_price,
                atm_strike=atm_strike,
                now=timestamp,
                latest_vix=latest_vix
            )
            
            # 2. ML Inference
            # The signal generator uses features dict
            raw_signal = self.strategy_router.generate_signal(features)
            
            # 3. Strategy Routing
            trade_recommendation = self.strategy_router.route_strategy(raw_signal, features)
            
            # 4. Prepare Result
            processing_time = time.time() - start_time
            
            result = {
                'type': 'signal',
                'exchange': self.exchange,
                'timestamp': timestamp,
                'processing_time': processing_time,
                'features': features, # Optional: might be too big to send back every time?
                'raw_signal': raw_signal, # StrategySignal dataclass (needs to be pickleable)
                'recommendation': trade_recommendation # TradeRecommendation dataclass
            }
            
            self.output_queue.put(result)
            
        except FeatureEngineeringError as fe:
            # Expected error (e.g. not enough history), just log debug
            # LOGGER.debug(f"[{self.exchange}] Feature engineering skipped: {fe}")
            pass 
        except Exception as e:
            LOGGER.error(f"[{self.exchange}] Failed to process tick: {e}")
            # traceback.print_exc()


class HandlerProxy:
    """
    Lightweight proxy for ExchangeDataHandler to pass state to feature engineering.
    """
    def __init__(self, state: Dict[str, Any]):
        self.underlying_token = state.get('underlying_token')
        self.expiry_date = state.get('expiry_date')
        self.data_reels = state.get('data_reels', {})
        self.futures_oi_reels = state.get('futures_oi_reels', [])
        self.microstructure_cache = state.get('microstructure_cache', {})
        self.flow_cache = state.get('flow_cache', {})
        self.oi_history = state.get('oi_history', [])
        self.latest_future_price = state.get('latest_future_price')
        self.macro_feature_cache = state.get('macro_feature_cache', {})
        
        # Add config proxy if needed
        self.config = ConfigProxy(state.get('config', {}))

    def calculate_atm_shift_intensity_ewma(self):
        # We might need to pre-calculate this in main thread or pass enough data
        # For now, return 0.0 or value passed in state
        return 0.0 

class ConfigProxy:
    def __init__(self, config_dict):
        self.risk_free_rate = config_dict.get('risk_free_rate', 0.10)

