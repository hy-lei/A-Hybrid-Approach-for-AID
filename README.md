# A-Hybrid-Approach-for-AID
This is a repo for replicating paper by Wang et. al, A Hybrid Approach for Automatic Incident Detection.

## Progress/Improvements (on features)
The two notebooks below illustrates the pipeline from data preprocessing to model evaluation.

- `hybrid_sev_weekday.ipynb`: historical speed/flow/occupancy, severity parameters, and weekday/weekend indicator.
- `hybrid_with_severity_params_and_weekdays.ipynb`: historical speed/flow/occupancy and severity parameters.  This model incorporates weekday/weekend information using an ensemble approach, namely there are different models are trained on weekday and weekend data.

## TODO:
- Bootstrap evaluation: write scripts to run models repeatedly.
- Data normalization based on stations: preprocess data separately for different stations.  We observed that each station has different speed, flow and occupancy distribution from one another.
