"""AI models package"""

from ._base_ai_models import (
    AIDoseModelBase,
    get_model,
    get_available_models,
    register_model,
)
from ._ConvLSTM import predict_ConvLSTM

register_model(predict_ConvLSTM)


__all__ = [
    "AIDoseModelBase",
    "predict_ConvLSTM",
    "get_model",
    "get_available_models",
    "register_model",
]
