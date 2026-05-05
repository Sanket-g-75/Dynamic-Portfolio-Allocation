- The idea is to create a Dynamic Portfolio Allocation that can use data related to indexes, commodities, vix, to adjust portfolio allocation in a set of stocks to maximize the reward 
- Will use YAHOO Finance to download the data of all the scripts.

stocks = ['NVDA','TSLA','RTX','GOOG','XOM','JPM','UNH']
w_indices = ['GSPC','SS','N225','DAX','NSEI','FTSE','FCHI']
forex = ['EURUSD','JPYUSD','GBPUSD','CHFUSD','AUDUSD']
commodities = ['GC','SI','CL','HG']
other = ['VIX']


- Infrastructure and Versioning > Github
- Data Versioning > DVC
- Orchestration > Apache Airflow
- Data Processing > Python, (Pydantic for Data Validation)
- Experiment Tracking > MLflow
- Model Serving > FastAPI + Docker
- CI/CD > Github Actions
- Observability > Promethus + Grafana


## Project Steps
* Ingestion
    * Ingest new stock either as a feature or as a stock for allocation
    * Update data to the latest data for all the stocks and features
* Preprocess
    * Combine all the prices of all the features 
    * Preprocess all the data to bring them in a certain range
* Model
    * Train the model when either new feature is added or new data is added
    * Make Prediction instantinously
    * Give portfolio changes based on 3 day time interval
    * Give portfolio metrics EOD daily
    * Get the last data of data till the model is trained