# HKU CS Final Year Project (fyp 18061)
Check out our project webpage [here](http://i.cs.hku.hk/~fyp18061)!
Some necessary information is provided below.

## A-Hybrid-Approach-for-AID
This repo started from replication of paper by Wang et. al, A Hybrid Approach for Automatic Incident Detection.

## Analysis of Data Source
- `data_analysis/regression.ipynb`: It tries to figure out the relationship between speed and other features.  Linear regression turns out to be a simple method to work out the explanatory power of `weekday` on speed.  It remains to be checked whether incidents are correlated with whether it is weekday or weekend.

## Progress/Improvements (on features)
The two notebooks below illustrates the pipeline from data preprocessing to model evaluation.

- `hybrid_sev_weekday.ipynb`: historical speed/flow/occupancy, severity parameters, and weekday/weekend indicator.
- `hybrid_with_severity_params_and_weekdays.ipynb`: historical speed/flow/occupancy and severity parameters.  This model incorporates weekday/weekend information using an ensemble approach, namely there are different models are trained on weekday and weekend data.

## TODO:
- Bootstrap evaluation: write scripts to run models repeatedly.
- Data normalization based on stations: preprocess data separately for different stations.  We observed that each station has different speed, flow and occupancy distribution from one another.
