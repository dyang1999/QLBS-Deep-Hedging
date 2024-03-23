# RL for options Hedging and Pricing

The library uses **Python 3** with the following modules:
- numpy 
- scipy 
- matplotlib 
- pandas 
- bspline (QLBS)
- tensorflow (DH)

# Q-Learning Black Scholes (QLBS)
Python library for QLBS learner 
- Incoporates **transaction costs** into the Q-learner 
- Alternative Q-learner reward function with sqaure root of the variance terms (standard deviation) + learning factor

## Notebooks 

```Run QLBS_master.ipynb``` -> QLBS with transaction costs 

```Run historical_crypto_data_qlbs.ipynb``` -> running QLBS on historical Crypto options data from Deribit 

```Run QLBS_std.ipynb``` -> QLBS with std reward function and smoothing

# Deep Hedging (DH) with simple NN and RNN
Python library for Deep Hedging
- Incoporates **transaction costs** into the Deep learner

```Run NN_options_master.ipynb.ipynb``` -> Simple NN and RNN with transaction costs 

```NN_model.py``` -> NN and RNN model architecture

## References

Igor Halperin, (2017). “QLBS: Q-Learner in the black-scholes (-merton) worlds." (QLBS)
Hans Buhler et al, (2019). “Deep Hedging.”  (Deep Hedging)


