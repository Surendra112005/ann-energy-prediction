# Household Appliance Energy Prediction using MLP

## Dataset

The official UCI Appliances Energy Prediction dataset contains 19,735 cleaned rows and 29 columns. Timestamp column: `date`. Target: `Appliances` (Wh). Dataset source: [UCI repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction).

## Chronological partitions

| Partition | Rows | Start | End |
|---|---:|---|---|
| Training | 13,814 | 2016-01-11 17:00:00 | 2016-04-16 15:10:00 |
| Validation | 2,960 | 2016-04-16 15:20:00 | 2016-05-07 04:30:00 |
| Test | 2,961 | 2016-05-07 04:40:00 | 2016-05-27 18:00:00 |

Imputation and feature/target scaling were fitted using training rows only. Time was encoded with cyclical hour, weekday, and month features. Duplicate complete rows were removed; rows without valid timestamps or target values were excluded.

## Model and training

The network has 31 inputs, hidden layers [128, 64, 32] with ReLU activations, and one linear output. It uses MSE and explicit full-batch gradient descent (one gradient update per epoch; batch size 13,814), learning rate 0.001, seed 48. Early stopping monitored validation MSE with patience 15 and restored the best validation checkpoint (epoch 1). The target is standardized using training statistics, and output weights are zero-initialized so initial predictions equal the training-target mean after inverse scaling. The dataset's random `rv1` and `rv2` fields are not used as predictors.

## Held-out test results

| Model | MAE (Wh) | RMSE (Wh) | R² |
|---|---:|---:|---:|
| Training-mean baseline | 52.8285 | 90.8921 | -0.0004 |
| MLP | 52.9614 | 90.8869 | -0.0003 |

All reported values are computed from the actual held-out test predictions. This single-building historical-data model estimates associations and should not be interpreted as causal or as a guarantee of future energy use.

## Citation

Candanedo, L. (2017). *Appliances Energy Prediction* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VC8G.
