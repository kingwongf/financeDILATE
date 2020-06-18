## DILATE Loss Model for Financial Market Indices Forecasting
Using the purposed loss objective, DILATE here [paper](https://papers.nips.cc/paper/8672-shape-and-time-distortion-loss-for-training-deep-time-series-forecasting-models), a seq2seq model has been trained. The out-of-sample loss in forecasting US Equity index below.


| Net_MSE                   |      MSE |     DTW |    TDI |
|---------------------------|----------|---------|--------|
| dynamic within train      | 2.219    | 1.992   | 0.0364 |
| log                       | 0.989196 | 1.53756 | 0      |
| expanding standard scalar | 3.31774  | 2.4167  | 0      |

| Net_Soft_DTW              |      MSE |     DTW |    TDI |
|---------------------------|----------|---------|--------|
| dynamic within train      | 2.239    | 1.921   | 0.0321 |
| log                       | 0.982469 | 1.53119 | 0      |
| expanding standard scalar | 3.18047  | 2.39887 | 0      |

| Net_DILATE                |      MSE |      DTW |      TDI |
|---------------------------|----------|----------|----------|
| dynamic within train      | 2.381    | 2.051    | 0.0236   |
| log                       | 0.985712 | 1.53426  | 0        |
| expanding standard scalar | 0.275519 | 0.693559 | 0.693559 |


Simply run `mainEncoderDEcoder2dProcess.py`, `mainEncoderDEcoder2dLogProcess.py` or `mainEncoderDEcoder2dZScoreProcess.py`. Replace with your own time series by changing `self.df`, `self.target_col` read in `__init__`. 
