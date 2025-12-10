"""
Black-Scholes Greeks Calculator for Options Trading.

Provides accurate Greeks calculations using scipy.stats and numpy.
"""

import numpy as np
from scipy.stats import norm
from math import log, sqrt, exp
from typing import Dict, List, Optional, Tuple


class GreeksCalculator:
    """
    Calculate Black-Scholes Greeks for options.
    """
    
    def __init__(self, risk_free_rate: float = 0.10):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 0.10 = 10% for India)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_greeks(
        self,
        spot_price: float,
        strike: float,
        time_to_expiry_years: float,
        implied_vol: float,
        option_type: str  # 'CE' or 'PE'
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for an option.
        
        Returns:
            Dict with keys: delta, gamma, vega, theta, vanna, charm, speed, zomma
        """
        if time_to_expiry_years <= 0:
            # Expired option - all Greeks are 0
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'vanna': 0.0,
                'charm': 0.0,
                'speed': 0.0,
                'zomma': 0.0
            }
        
        if implied_vol <= 0 or spot_price <= 0:
            return self._zero_greeks()
        
        is_call = option_type.upper() in ['CE', 'C', 'CALL']
        
        # Calculate d1 and d2
        d1, d2 = self._calculate_d1_d2(
            spot_price, strike, time_to_expiry_years, implied_vol
        )
        
        # PDF of standard normal at d1
        nd1_prime = norm.pdf(d1)
        
        # CDF of standard normal
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2) if is_call else norm.cdf(-d2)
        n_neg_d1 = norm.cdf(-d1)
        n_neg_d2 = norm.cdf(-d2)
        
        sqrt_t = sqrt(time_to_expiry_years)
        vol_sqrt_t = implied_vol * sqrt_t
        
        # Basic Greeks
        delta = nd1 if is_call else (nd1 - 1.0)
        
        gamma = nd1_prime / (spot_price * vol_sqrt_t) if vol_sqrt_t > 0 else 0.0
        
        vega = spot_price * nd1_prime * sqrt_t / 100.0  # Divided by 100 for % vol change
        
        # Theta (time decay)
        discount_factor = exp(-self.risk_free_rate * time_to_expiry_years)
        if is_call:
            theta = (
                -(spot_price * nd1_prime * implied_vol) / (2 * sqrt_t)
                - self.risk_free_rate * strike * discount_factor * nd2
            )
        else:
            theta = (
                -(spot_price * nd1_prime * implied_vol) / (2 * sqrt_t)
                + self.risk_free_rate * strike * discount_factor * n_neg_d2
            )
        
        # Higher-order Greeks
        if vol_sqrt_t > 0:
            vanna = -nd1_prime * d2 / implied_vol
            
            # Charm (delta decay)
            if is_call:
                charm = -nd1_prime * (2 * self.risk_free_rate * time_to_expiry_years - d2 * vol_sqrt_t) / (2 * time_to_expiry_years * vol_sqrt_t)
            else:
                charm = -nd1_prime * (2 * self.risk_free_rate * time_to_expiry_years + d2 * vol_sqrt_t) / (2 * time_to_expiry_years * vol_sqrt_t)
            
            # Speed (third-order price sensitivity)
            speed = -gamma * (d1 / vol_sqrt_t + 1) / spot_price if spot_price > 0 else 0.0
            
            # Zomma (gamma-vega)
            zomma = gamma * (d1 * d2 - 1) / implied_vol if implied_vol > 0 else 0.0
        else:
            vanna = 0.0
            charm = 0.0
            speed = 0.0
            zomma = 0.0
        
        return {
            'delta': float(delta),
            'gamma': float(gamma),
            'vega': float(vega),
            'theta': float(theta),
            'vanna': float(vanna),
            'charm': float(charm),
            'speed': float(speed),
            'zomma': float(zomma),
            'd1': float(d1),
            'd2': float(d2)
        }
    
    def _calculate_d1_d2(
        self,
        spot_price: float,
        strike: float,
        time_to_expiry_years: float,
        implied_vol: float
    ) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes."""
        if time_to_expiry_years <= 0 or implied_vol <= 0:
            return 0.0, 0.0
        
        sqrt_t = sqrt(time_to_expiry_years)
        vol_sqrt_t = implied_vol * sqrt_t
        
        if strike <= 0:
            return 0.0, 0.0
        
        d1 = (
            log(spot_price / strike) + 
            (self.risk_free_rate + 0.5 * implied_vol ** 2) * time_to_expiry_years
        ) / vol_sqrt_t
        
        d2 = d1 - vol_sqrt_t
        
        return d1, d2
    
    def _zero_greeks(self) -> Dict[str, float]:
        """Return zero Greeks."""
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'vanna': 0.0,
            'charm': 0.0,
            'speed': 0.0,
            'zomma': 0.0
        }


def calculate_gamma_exposure(
    call_options: List[Dict],
    put_options: List[Dict],
    spot_price: float,
    time_to_expiry_years: float,
    risk_free_rate: float = 0.10,
    contract_multiplier: int = 50  # NIFTY lot size
) -> Dict[str, float]:
    """
    Calculate market-wide Gamma Exposure (GEX) and other aggregate Greeks.
    
    Args:
        call_options: List of call option dicts with keys: strike, latest_oi, iv
        put_options: List of put option dicts with keys: strike, latest_oi, iv
        spot_price: Current underlying spot price
        time_to_expiry_years: Time to expiry in years
        risk_free_rate: Risk-free rate (default 0.10)
        contract_multiplier: Contract multiplier (lot size)
    
    Returns:
        Dict with aggregate Greeks metrics
    """
    calculator = GreeksCalculator(risk_free_rate)
    
    total_gex = 0.0
    total_vanna_exposure = 0.0
    total_charm_exposure = 0.0
    total_speed_exposure = 0.0
    total_zomma_exposure = 0.0
    
    # Process call options (positive gamma exposure for market makers if they're short)
    for opt in call_options:
        strike = opt.get('strike')
        oi = opt.get('latest_oi', 0) or 0
        iv = opt.get('iv', 0.0) or 0.0
        
        if strike is None or oi <= 0 or iv <= 0:
            continue
        
        greeks = calculator.calculate_greeks(
            spot_price, strike, time_to_expiry_years, iv, 'CE'
        )
        
        # Gamma Exposure: OI × Gamma × Spot² × Multiplier
        # Market makers are typically short options, so we use negative sign
        # But convention is to report as if market makers are short (negative GEX = positive for price)
        gex_contribution = contract_multiplier * oi * greeks['gamma'] * (spot_price ** 2)
        total_gex += gex_contribution
        
        # Aggregate other exposures
        total_vanna_exposure += contract_multiplier * oi * greeks['vanna'] * spot_price
        total_charm_exposure += contract_multiplier * oi * greeks['charm'] * spot_price
        total_speed_exposure += contract_multiplier * oi * greeks['speed'] * (spot_price ** 3)
        total_zomma_exposure += contract_multiplier * oi * greeks['zomma'] * (spot_price ** 2)
    
    # Process put options (negative gamma exposure for market makers if they're short)
    for opt in put_options:
        strike = opt.get('strike')
        oi = opt.get('latest_oi', 0) or 0
        iv = opt.get('iv', 0.0) or 0.0
        
        if strike is None or oi <= 0 or iv <= 0:
            continue
        
        greeks = calculator.calculate_greeks(
            spot_price, strike, time_to_expiry_years, iv, 'PE'
        )
        
        # Puts contribute negatively to GEX (market makers short puts = negative gamma)
        gex_contribution = -contract_multiplier * oi * greeks['gamma'] * (spot_price ** 2)
        total_gex += gex_contribution
        
        # Aggregate other exposures (puts contribute with opposite sign)
        total_vanna_exposure -= contract_multiplier * oi * greeks['vanna'] * spot_price
        total_charm_exposure -= contract_multiplier * oi * greeks['charm'] * spot_price
        total_speed_exposure -= contract_multiplier * oi * greeks['speed'] * (spot_price ** 3)
        total_zomma_exposure -= contract_multiplier * oi * greeks['zomma'] * (spot_price ** 2)
    
    # Calculate gamma flip zones (strikes where gamma changes sign)
    # This is simplified - full implementation would find zero crossings
    gamma_flip_zones = _calculate_gamma_flip_zones(
        call_options, put_options, spot_price, time_to_expiry_years, risk_free_rate
    )
    
    return {
        'gamma_exposure': total_gex,
        'vanna_exposure': total_vanna_exposure,
        'charm_exposure': total_charm_exposure,
        'speed_exposure': total_speed_exposure,
        'zomma_exposure': total_zomma_exposure,
        'gamma_flip_zones': gamma_flip_zones
    }


def _calculate_gamma_flip_zones(
    call_options: List[Dict],
    put_options: List[Dict],
    spot_price: float,
    time_to_expiry_years: float,
    risk_free_rate: float = 0.10
) -> List[float]:
    """
    Identify strike prices where net gamma exposure changes sign (flip zones).
    
    Returns:
        List of strike prices where gamma flips
    """
    # Simplified: Find strikes where net gamma is near zero
    # Full implementation would solve for zero crossings
    calculator = GreeksCalculator(risk_free_rate)
    
    flip_zones = []
    strikes = sorted(set([
        opt.get('strike') for opt in call_options + put_options 
        if opt.get('strike') is not None
    ]))
    
    if len(strikes) < 2:
        return flip_zones
    
    # Calculate net gamma at each strike
    net_gammas = []
    for strike in strikes:
        # Find options at this strike
        call_oi = sum(
            opt.get('latest_oi', 0) or 0 
            for opt in call_options 
            if opt.get('strike') == strike
        )
        put_oi = sum(
            opt.get('latest_oi', 0) or 0 
            for opt in put_options 
            if opt.get('strike') == strike
        )
        
        if call_oi == 0 and put_oi == 0:
            net_gammas.append(0.0)
            continue
        
        # Use average IV for the strike
        call_ivs = [opt.get('iv', 0.0) for opt in call_options if opt.get('strike') == strike and opt.get('iv')]
        put_ivs = [opt.get('iv', 0.0) for opt in put_options if opt.get('strike') == strike and opt.get('iv')]
        
        avg_iv = 0.0
        if call_ivs or put_ivs:
            avg_iv = np.mean([iv for iv in call_ivs + put_ivs if iv > 0]) if (call_ivs or put_ivs) else 0.0
        
        if avg_iv <= 0:
            net_gammas.append(0.0)
            continue
        
        # Calculate gamma contributions
        call_greeks = calculator.calculate_greeks(spot_price, strike, time_to_expiry_years, avg_iv, 'CE')
        put_greeks = calculator.calculate_greeks(spot_price, strike, time_to_expiry_years, avg_iv, 'PE')
        
        net_gamma = (call_oi * call_greeks['gamma']) - (put_oi * put_greeks['gamma'])
        net_gammas.append(net_gamma)
    
    # Find sign changes (simplified - look for near-zero crossings)
    for i in range(len(net_gammas) - 1):
        if net_gammas[i] * net_gammas[i + 1] < 0:  # Sign change
            # Interpolate flip zone
            flip_zone = (strikes[i] + strikes[i + 1]) / 2.0
            flip_zones.append(flip_zone)
    
    return flip_zones

