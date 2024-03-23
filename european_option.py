from scipy import stats
import numpy as np

# Assume continuous dividend with flat term-structure and flat dividend structure.
class EuropeanOption:
    def __init__(self):
        pass
        
    def get_BS_price(self, S=None, sigma=None, risk_free=None, \
                     dividend=None, K=None, exercise_date=None, calculation_date=None, \
                     day_count=None, dt=None, opt_type=None):
        
        # For our purpose, assume s0 is a NumPy array, other inputs are scalar.
        
#         T = np.arange(0, (exercise_date - calculation_date + 1)) * dt
        T = np.linspace(0, dt * (25 - 1), 25)
        
        T = np.repeat(np.flip(T[None, :]), S.shape[0], 0)

        # Ignore division by 0 warning (expected behaviors as the limits of CDF is defined).
        with np.errstate(divide='ignore'):
            d1 = np.divide(np.log(S / K) + (risk_free - dividend + 0.5 * sigma ** 2) * T, sigma * np.sqrt(T))
            d2 = np.divide(np.log(S / K) + (risk_free - dividend - 0.5 * sigma ** 2) * T, sigma * np.sqrt(T))

        if opt_type.lower() == 'call':
            return (S * stats.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-risk_free * T) * stats.norm.cdf(d2, 0.0, 1.0))
        elif opt_type.lower() == 'put':
            return (K * np.exp(-risk_free * T) * stats.norm.cdf(-d2, 0.0, 1.0) - S * stats.norm.cdf(-d1, 0.0, 1.0))
        else:
            raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    def get_BS_delta(self, S=None, sigma=None, risk_free=None, \
                     dividend=None, K=None, exercise_date=None, calculation_date=None, \
                     day_count=None, dt=None, opt_type=None):
        

        # For our purpose, assume s0 is a NumPy array, other inputs are scalar.
#         T = np.arange(0, (exercise_date - calculation_date + 1)) * dt
        T = np.linspace(0, dt * (25 - 1), 25)
        T = np.repeat(np.flip(T[None, :]), S.shape[0], 0)

        # Ignore division by 0 warning (expected behaviors as the limits of CDF is defined).
        with np.errstate(divide='ignore'):
            d1 = np.divide(np.log(S / K) + (risk_free - dividend + 0.5 * sigma ** 2) * T, sigma * np.sqrt(T))

        if opt_type.lower() == 'call':
            return stats.norm.cdf(d1, 0.0, 1.0)
        elif opt_type.lower() == 'put':
            return stats.norm.cdf(d1, 0.0, 1.0) - 1
        else:
            raise ValueError("Invalid option type. Must be 'call' or 'put'.")
        return stats.norm.cdf(d1, 0.0, 1.0)
            
            
    def get_BS_PnL(self, S=None, payoff=None, delta=None, dt=None, risk_free=None, \
                    final_period_cost=None, epsilon=None, cost_structure="proportional"):
        # Compute Black-Scholes PnL (for a short position, i.e. the Bank sells
        N = S.shape[1] - 1
        
        PnL_BS = np.multiply(S[:, 0], -delta[:, 0]) 
        
        if cost_structure == "proportional":
            PnL_BS -= np.abs(delta[:, 0]) * S[:, 0] * epsilon
        elif cost_structure == "constant":
            PnL_BS -= epsilon
                
        PnL_BS = PnL_BS * np.exp(risk_free * dt)
        
        for t in range(1, N):
            PnL_BS += np.multiply(S[:, t], -delta[:, t] + delta[:, t - 1])
            
            if cost_structure == "proportional":
                PnL_BS -= np.abs(delta[:, t] - delta[:, t - 1]) * S[:, t] * epsilon
            elif cost_structure == "constant":
                PnL_BS -= epsilon
                
            PnL_BS = PnL_BS * np.exp(risk_free * dt)

        PnL_BS += np.multiply(S[:, N], delta[:, N - 1]) + payoff 
        
        if final_period_cost:
            if cost_structure == "proportional":
                PnL_BS -= np.abs(delta[:, N - 1]) * S[:, N] * epsilon
            elif cost_structure == "constant":
                PnL_BS -= epsilon
                
        return PnL_BS


