from .lstm_forecast import LSTMForecast
from .transformer_forecast import TimeSeriesTransformer
from .cnn_transformer_hybrid import CNNTransformerHybrid
from .llama_timeseries import LLaMATimeSeries
from .cnn_timeseries import CNNTimeseries
from .cnn_lstm_timeseries import CNNLSTMTimeSeries
from .bidirectional_lstm import BiLSTMTimeSeries

__all__ = [
    "LSTMForecast",
    "TimeSeriesTransformer",
    "CNNTransformerHybrid",
    "LLaMATimeSeries",
    "CNNTimeseries",
    "CNNLSTMTimeSeries",
    "BiLSTMTimeSeries",
]

# Registry for easy model initialization
MODEL_REGISTRY = {
    "lstm": LSTMForecast,
    "transformer": TimeSeriesTransformer,
    "cnn_transformer": CNNTransformerHybrid,
    "llama_ts": LLaMATimeSeries,
    "cnn": CNNTimeseries,
    "cnn_lstm": CNNLSTMTimeSeries,
    "bilstm": BiLSTMTimeSeries,
}
