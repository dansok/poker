from enum import Enum
from operator import countOf
import random


class ACTION(Enum):
    RAISE = 0
    CALL = 1
    CHECK = 2
    FOLD = 3
    # BLIND = 4   it will be emulated with RAISE

    @staticmethod
    def select_random(actions):
        result = random.choice([ACTION.RAISE, ACTION.CALL, ACTION.CHECK, ACTION.FOLD])
        if result == ACTION.RAISE and countOf(actions, ACTION.RAISE) >= 2:
            return ACTION.CALL
        return result

    @staticmethod
    def select_raise_or_call(actions):
        if countOf(actions, ACTION.RAISE) >= 2:
            return ACTION.CALL
        return random.choice([ACTION.RAISE, ACTION.CALL])


class ActionResult:
    def __init__(self, action, contribution):
        self.action = action
        self.contribution = contribution
