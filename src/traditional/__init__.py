'''
传统音频啸叫抑制方法模块

本模块包含三种经典的音频啸叫抑制算法，用于与深度学习方法进行性能对比：

1. FrequencyShiftMethod - 移频移向法
2. GainSuppressionMethod - 增益抑制法  
3. AdaptiveFeedbackMethod - 自适应反馈抵消法

所有方法都继承自nn.Module，保持与现有深度学习框架的兼容性，
可以直接替换深度学习模型进行训练和评估。
'''

from .frequency_shift import FrequencyShiftMethod
from .gain_suppression import GainSuppressionMethod
from .adaptive_feedback import AdaptiveFeedbackMethod

__all__ = [
    'FrequencyShiftMethod',
    'GainSuppressionMethod', 
    'AdaptiveFeedbackMethod'
]
