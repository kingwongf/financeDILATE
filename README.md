## DILATE Loss Model for Financial Market Indices Forecasting
Using the purposed loss objective, DILATE here [paper](https://papers.nips.cc/paper/8672-shape-and-time-distortion-loss-for-training-deep-time-series-forecasting-models), a seq2seq model has been trained. The out-of-sample loss in forecasting US Equity index below.


|                  |   MSE |   DTW |    TDI |
|------------------|-------|-------|--------|
| net_gru_mse      | **2.219** | 1.992 | 0.0364 |
| net_gru_soft_dtw | 2.239 | **1.921** | 0.0321 |
| net_gru_dilate   | 2.381 | 2.051 | **0.0236** |


Simply run `mainEncoderDEcoder2dProcess.py`, `mainEncoderDEcoder2dLogProcess.py` or `mainEncoderDEcoder2dZScoreProcess.py`. Replace with your own time series by changing self.df, self.target_col read in `__init__`. 
