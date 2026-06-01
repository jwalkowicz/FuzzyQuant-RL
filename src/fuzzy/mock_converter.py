import random


# Mock
# TBD
class FuzzyStateConverter:
    def get_state(self, rsi: float, macd: float) -> int:
        return random.randint(0, 8)
