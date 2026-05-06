from prometheus_client import Counter, Histogram, Gauge
import psutil

# Total prediction requests
prediction_requests = Counter(
    'prediction_requests_total',
    'Total prediction requests'
)

# Failed predictions
prediction_failures = Counter(
    'prediction_failures_total',
    'Total failed prediction requests'
)

# Prediction latency
prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds'
)

# CPU Usage
cpu_usage = Gauge(
    'cpu_usage_percent',
    'Current CPU usage'
)

# Memory Usage
memory_usage = Gauge(
    'memory_usage_percent',
    'Current memory usage'
)

def update_system_metrics():
    cpu_usage.set(psutil.cpu_percent())
    memory_usage.set(psutil.virtual_memory().percent)