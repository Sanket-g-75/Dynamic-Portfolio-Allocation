from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    dag_id='dynamic_portfolio_pipeline',
    default_args=default_args,
    description='Orchestrates ingestion, preprocessing, and training for dynamic portfolio allocation.',
    schedule_interval='@daily',
    catchup=False,
    tags=['portfolio'],
) as dag:

    # Task 1: Ingest Data
    ingest_data = BashOperator(
        task_id='ingest_data',
        bash_command='cd D:/1.Work/WorkStation/WorkSpace/Projects/Dynamic-Portfolio-Allocation && python -m src.ingest'
    )

    # Task 2: Preprocess Data
    preprocess_data = BashOperator(
        task_id='preprocess_data',
        bash_command='cd D:/1.Work/WorkStation/WorkSpace/Projects/Dynamic-Portfolio-Allocation && python -m src.preprocess'
    )

    # Task 3: Train Model
    train_model = BashOperator(
        task_id='train_model',
        bash_command='cd D:/1.Work/WorkStation/WorkSpace/Projects/Dynamic-Portfolio-Allocation && python -m src.train'
    )

    # Define task dependencies
    ingest_data >> preprocess_data >> train_model
