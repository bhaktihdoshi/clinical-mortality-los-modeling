# Final Model Artifact Manifest

This file is the clean coded record of final inputs and outputs used by each model family. It supports reproducibility and keeps official result files separate from exploratory or utility reruns.

## Official Runbook And Corrected Splits

- Merged runbook notebook: `/Users/bhaktidoshi/Downloads/Official_All_Models_Runbook.ipynb`
- Corrected split builder: `/Users/bhaktidoshi/Downloads/m05 - data set 3/make_corrected_tabular_splits.py`
- Corrected tabular splits: `final_train_tabular_corrected.csv`, `final_val_corrected.csv`, `final_test_corrected.csv`
- Correction summary: `corrected_tabular_split_summary.csv`
- Purpose: restores the 13 admissions present in the corrected transformer split but missing from the earlier tabular split files.

## Transformer

- Notebook: `/Users/bhaktidoshi/Downloads/dataset3_mortality_prediction.ipynb`
- Inputs: `transformer_train_master.csv`, `transformer_val_master.csv`, `transformer_test_master.csv`, `transformer_train_sequence_events.csv`, `transformer_val_sequence_events.csv`, `transformer_test_sequence_events.csv`
- Official outputs: `transformer_targeted_run.log`, `transformer_official_mortality_metrics.csv`, `transformer_official_los_metrics.csv`, `transformer_team_comparison_metrics.csv`
- Prediction outputs: `transformer_test_predictions.csv`, `transformer_los_test_predictions.csv`
- Plots: `transformer_mortality_curves.png`, `transformer_sgkf_mortality.png`, `transformer_sgkf_los.png`, `transformer_fairness_auprc.png`
- Robustness outputs: `transformer_sgkf_mortality.csv`, `transformer_sgkf_los.csv`
- Fairness output: `transformer_fairness_results.csv`

## XGBoost

- Notebook: `/Users/bhaktidoshi/Downloads/XGB 2.0.ipynb`
- Inputs: `final_train_tabular_corrected.csv`, `final_val_corrected.csv`, `final_test_corrected.csv` with fallback to `final_train_tabular.csv`, `final_val.csv`, `final_test.csv`
- Official outputs to save from notebook: `xgb_official_mortality_metrics.csv`, `xgb_official_los_metrics.csv`, `xgb_team_comparison_metrics.csv`
- Prediction outputs to save from notebook: `xgb_test_predictions.csv`, `xgb_los_test_predictions.csv`
- Fairness output to save from notebook: `xgb_fairness_results.csv`

## LSTM

- Notebook: `/Users/bhaktidoshi/Downloads/LSTM 2.0.ipynb`
- Inputs: `final_train_tabular_corrected.csv`, `final_val_corrected.csv`, `final_test_corrected.csv`, `new_data.csv` with fallback to original tabular split files
- Official outputs to save from notebook: `lstm_official_mortality_metrics.csv`, `lstm_official_los_metrics.csv`, `lstm_team_comparison_metrics.csv`
- Prediction outputs to save from notebook: `lstm_test_predictions.csv`, `lstm_los_test_predictions.csv`
- Robustness scope file: `lstm_robustness_scope.csv`
- Fairness output to save from notebook: `lstm_fairness_results.csv`

## BioGPT LLM

- Notebook: `/Users/bhaktidoshi/Downloads/notebook481dd59c7e_LLM.ipynb`
- Inputs: `final_train_tabular_corrected.csv`, `final_val_corrected.csv`, `final_test_corrected.csv`, raw diagnosis/CCS lookup files with fallback to original tabular split files
- Official mortality output: `biogpt_official_mortality_metrics.csv` plus backward-compatible `biogpt_official_test_metrics.csv`
- Official LOS output: `biogpt_los_official_test_metrics.csv`
- Team comparison output: `biogpt_team_comparison_metrics.csv`
- Prediction outputs: `biogpt_test_predictions.csv`, `biogpt_los_test_predictions.csv`
- Fairness outputs: `biogpt_fairness_results.csv`, `biogpt_fairness_auprc.png`
- Robustness scope file: `biogpt_robustness_scope.csv`

## EDA and Unsupervised Analysis

- Notebooks: `/Users/bhaktidoshi/Downloads/Section 1 (EDA & DC).ipynb`, `/Users/bhaktidoshi/Downloads/Section 2 (Pre-Processing & Split).ipynb`
- Outputs to save/reference: `outlier_summary.csv`, `unsupervised_cluster_summary.csv`, `clinical_cluster_pca.png`, `los_winsorization.png`
- Note: reference only unsupervised methods that were actually run, currently KMeans/PCA-style clinical clustering.

## Team Comparison

- Reusable builder script: `/Users/bhaktidoshi/Downloads/m05 - data set 3/build_team_model_comparison.py`
- Combined outputs: `team_model_comparison_metrics.csv`, `team_mortality_comparison_metrics.csv`, `team_los_comparison_metrics.csv`
- Standard mortality columns: `AUROC`, `AUPRC`, `F1`, `Precision`, `Recall`, `Sensitivity`, `Specificity`, `threshold_used`
- Standard LOS columns: `RMSE`, `R2`
