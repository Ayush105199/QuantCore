"""
Black-Scholes Options Pricing Engine
--------------------------------------
Implements the full Black-Scholes model from scratch — no scipy, no shortcuts.
Every formula is derived from first principles.

Model assumptions:
  1. Underlying follows Geometric Brownian Motion (GBM)
  2. No dividends
  3. Constant volatility (sigma) — the most-criticized assumption
  4. Constant risk-free rate (r)
  5. European options only (can exercise only at expiry)
  6. No transaction costs, no bid-ask spread

Parameters:
  S     : current spot price of the underlying
  K     : strike price
  T     : time to expiry in years (e.g. 30 days = 30/365 = 0.0822)
  r     : risk-free rate (annualized, e.g. 0.065 for 6.5%)
  sigma : implied/realized volatility (annualized, e.g. 0.20 for 20%)
"""

import math
from dataclasses import dataclass


# ── Normal Distribution (implemented from scratch) ────────────────────────────

def _norm_pdf(x: float) -> float:
    """
    Standard normal probability density function.
    f(x) = (1 / sqrt(2*pi)) * exp(-x^2 / 2)
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function.

    Uses the Abramowitz & Stegun (1964) rational approximation 26.2.17.
    Maximum absolute error: 7.5e-8 — sufficient for all finance applications.

    This is the same approximation used in many production trading systems
    because it avoids the error function and is fast to compute.

    Coefficients:
        p  = 0.2316419
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
    """
    if x >= 0:
        t    = 1.0 / (1.0 + 0.2316419 * x)
        poly = t * (0.319381530
               + t * (-0.356563782
               + t * (1.781477937
               + t * (-1.821255978
               + t * 1.330274429))))
        return 1.0 - _norm_pdf(x) * poly
    else:
        return 1.0 - _norm_cdf(-x)


# ── d1 and d2 ─────────────────────────────────────────────────────────────────

def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))

    Interpretation: distance-to-strike adjusted for growth and vol.
    N(d1) = delta of a call option.
    """
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    d2 = d1 - sigma * sqrt(T)

    Interpretation: N(d2) = risk-neutral probability that S_T > K at expiry.
    """
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


# ── Option Prices ─────────────────────────────────────────────────────────────

def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes call price.
    C = S * N(d1) - K * exp(-r*T) * N(d2)

    Interpretation:
      S * N(d1)           = expected value of receiving the stock if S_T > K
      K * exp(-r*T)*N(d2) = PV of paying the strike if S_T > K
    """
    if T <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes put price.
    P = K * exp(-r*T) * N(-d2) - S * N(-d1)

    Also derivable from put-call parity:
    P = C - S + K * exp(-r*T)
    """
    if T <= 0:
        return max(K - S, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


# ── The Greeks ────────────────────────────────────────────────────────────────

def delta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call") -> float:
    """
    Delta = dC/dS (or dP/dS for puts).

    Call delta = N(d1)      Range: [0, 1]
    Put  delta = N(d1) - 1  Range: [-1, 0]

    Intuition: If delta = 0.6, the option gains Rs.0.60 for every Rs.1
    rise in the underlying. Delta = 0.5 means the option is "at-the-money".

    Use: Delta hedging — to be market-neutral, short delta shares per long option.
    """
    if T <= 0:
        return (1.0 if S > K else 0.0) if option_type == "call" else (-1.0 if S < K else 0.0)
    d1_val = _d1(S, K, T, r, sigma)
    if option_type == "call":
        return _norm_cdf(d1_val)
    return _norm_cdf(d1_val) - 1.0


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Gamma = d^2C/dS^2 = d(Delta)/dS  [same for calls and puts]

    Gamma = N'(d1) / (S * sigma * sqrt(T))

    Intuition: Rate of change of delta. High gamma = delta changes rapidly.
    ATM options near expiry have the highest gamma (most sensitive).
    Gamma scalping: buy gamma, delta-hedge frequently → profit from volatility.

    Use: Risk management — gamma risk is the primary concern for options desks.
    """
    if T <= 0:
        return 0.0
    d1_val = _d1(S, K, T, r, sigma)
    return _norm_pdf(d1_val) / (S * sigma * math.sqrt(T))


def theta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call") -> float:
    """
    Theta = dC/dT (time decay per day).

    Call theta = -[S*N'(d1)*sigma/(2*sqrt(T))] - r*K*exp(-rT)*N(d2)
    Put  theta = -[S*N'(d1)*sigma/(2*sqrt(T))] + r*K*exp(-rT)*N(-d2)

    Result is divided by 365 to give theta per calendar day.
    Theta is almost always negative — options lose value as time passes.

    Intuition: Selling options (short theta) earns the time decay.
    Buying options (long theta) pays time decay daily.

    Use: Option sellers target high-theta strategies (iron condors, covered calls).
    """
    if T <= 0:
        return 0.0
    d1_val = _d1(S, K, T, r, sigma)
    d2_val = d1_val - sigma * math.sqrt(T)
    disc   = K * math.exp(-r * T)

    term1 = -(S * _norm_pdf(d1_val) * sigma) / (2.0 * math.sqrt(T))
    if option_type == "call":
        term2 = -r * disc * _norm_cdf(d2_val)
    else:
        term2 = r * disc * _norm_cdf(-d2_val)

    return (term1 + term2) / 365.0


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Vega = dC/d(sigma) = dP/d(sigma)  [same for calls and puts]

    Vega = S * N'(d1) * sqrt(T)
    Divided by 100 to express per 1% change in volatility.

    Intuition: A vega of 0.25 means the option gains Rs.0.25 for every
    1% increase in implied volatility.

    Use: Volatility trading. HFT vol desks trade vega constantly.
    """
    if T <= 0:
        return 0.0
    d1_val = _d1(S, K, T, r, sigma)
    return S * _norm_pdf(d1_val) * math.sqrt(T) / 100.0


def rho(S: float, K: float, T: float, r: float, sigma: float,
        option_type: str = "call") -> float:
    """
    Rho = dC/dr (sensitivity to interest rate changes).

    Call rho = K * T * exp(-rT) * N(d2)   (positive — calls benefit from higher r)
    Put  rho = -K * T * exp(-rT) * N(-d2) (negative — puts hurt by higher r)

    Divided by 100 to express per 1% change in rates.

    Note: Rho is the smallest Greek in magnitude for short-dated options.
    Matters more for long-dated options (LEAPS).
    """
    if T <= 0:
        return 0.0
    d1_val = _d1(S, K, T, r, sigma)
    d2_val = d1_val - sigma * math.sqrt(T)
    disc   = K * T * math.exp(-r * T)

    if option_type == "call":
        return disc * _norm_cdf(d2_val) / 100.0
    return -disc * _norm_cdf(-d2_val) / 100.0


# ── Convenience Bundle ────────────────────────────────────────────────────────

@dataclass
class OptionResult:
    """Full pricing result for one option."""
    price      : float
    delta      : float
    gamma      : float
    theta      : float   # per calendar day
    vega       : float   # per 1% vol move
    rho        : float   # per 1% rate move
    d1         : float
    d2         : float
    option_type: str
    S          : float
    K          : float
    T          : float
    r          : float
    sigma      : float

    def moneyness(self) -> str:
        ratio = self.S / self.K
        if ratio > 1.02:
            return "ITM" if self.option_type == "call" else "OTM"
        if ratio < 0.98:
            return "OTM" if self.option_type == "call" else "ITM"
        return "ATM"

    def intrinsic_value(self) -> float:
        if self.option_type == "call":
            return max(self.S - self.K, 0.0)
        return max(self.K - self.S, 0.0)

    def time_value(self) -> float:
        return self.price - self.intrinsic_value()

    def summary(self) -> str:
        lines = [
            f"\n  {'='*48}",
            f"  Option: {self.option_type.upper()}  |  {self.moneyness()}",
            f"  S={self.S:.2f}  K={self.K:.2f}  T={self.T:.4f}yr  "
            f"r={self.r*100:.1f}%  sigma={self.sigma*100:.1f}%",
            f"  {'─'*48}",
            f"  Price         : {self.price:.6f}",
            f"  Intrinsic     : {self.intrinsic_value():.6f}",
            f"  Time Value    : {self.time_value():.6f}",
            f"  {'─'*48}",
            f"  Delta         : {self.delta:+.6f}   (Rs change per Rs1 spot move)",
            f"  Gamma         : {self.gamma:.8f}  (delta change per Rs1 spot move)",
            f"  Theta         : {self.theta:+.6f}   (Rs change per day — time decay)",
            f"  Vega          : {self.vega:.6f}   (Rs change per 1% vol move)",
            f"  Rho           : {self.rho:+.6f}   (Rs change per 1% rate move)",
            f"  d1            : {self.d1:.6f}",
            f"  d2            : {self.d2:.6f}",
            f"  {'='*48}",
        ]
        return "\n".join(lines)


def price_option(S: float, K: float, T: float, r: float, sigma: float,
                 option_type: str = "call") -> OptionResult:
    """
    Main entry point. Price one option and return all Greeks.

    Args:
        S           : spot price
        K           : strike price
        T           : time to expiry (years)
        r           : risk-free rate (decimal)
        sigma       : volatility (decimal)
        option_type : "call" or "put"

    Returns:
        OptionResult with price + all 5 Greeks
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    d1_val = _d1(S, K, T, r, sigma) if T > 0 else 0.0
    d2_val = d1_val - sigma * math.sqrt(T) if T > 0 else 0.0

    price_fn = bs_call if option_type == "call" else bs_put

    return OptionResult(
        price       = price_fn(S, K, T, r, sigma),
        delta       = delta(S, K, T, r, sigma, option_type),
        gamma       = gamma(S, K, T, r, sigma),
        theta       = theta(S, K, T, r, sigma, option_type),
        vega        = vega(S, K, T, r, sigma),
        rho         = rho(S, K, T, r, sigma, option_type),
        d1          = d1_val,
        d2          = d2_val,
        option_type = option_type,
        S=S, K=K, T=T, r=r, sigma=sigma,
    )


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("  BLACK-SCHOLES OPTIONS PRICING ENGINE")
    print("="*55)

    # NIFTY-style example: Nifty at 22000, strike 22500, 1 month to expiry
    result = price_option(S=22000, K=22500, T=30/365, r=0.065, sigma=0.18, option_type="call")
    print(result.summary())

    # ATM option — highest gamma and vega
    print("\n  ATM example (S == K):")
    atm = price_option(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call")
    print(atm.summary())

    # Put-call parity check
    call_p = bs_call(100, 100, 0.25, 0.05, 0.20)
    put_p  = bs_put (100, 100, 0.25, 0.05, 0.20)
    parity = call_p - put_p - 100 + 100 * math.exp(-0.05 * 0.25)
    print(f"\n  Put-call parity check: C - P - S + K*exp(-rT) = {parity:.10f}")
    print(f"  (Should be ~0.0 — confirms formula correctness)")

    print("\nPhase 2 Black-Scholes complete.")
