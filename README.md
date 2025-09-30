# 🌞 Solar Power Prediction

A research-ready pipeline for **solar power generation forecasting** using deep learning models (LSTM, Transformer, CNN-Transformer Hybrid, and LLaMA-inspired Time Series models).  
The pipeline integrates **weather + site metadata + temporal encodings** into grouped time-series datasets to predict normalized solar generation.

---



## ⚙️ Dataset Preparation
first download the data from https://www.kaggle.com/datasets/cdaclab/unisolar 
this need to be 4 csv 
Monthly_Summary_Solar.csv
Solar_Energy_Generation.csv
3042 Solar_Site_Details.csv
Weather_Data_reordered_all.csv
This project uses the **UniSolar Dataset** (Austrian PV plants).  
To build the master dataset:

```bash
python build_master_dataset.py --data_folder /path/to/unisolar
```
This generates: 
Solar_Power_Weather_Clean_MASTER.csv



## training


```bash
python run.py --config configs/example.yaml
```



## Evaluation

To compare models + ensemble:
```bash
python test.py --config configs/example.yaml
```

