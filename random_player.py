import random
from enum import Enum

from hand import Hand


class ACTION(Enum):
    RAISE = 1
    CALL = 2
    CHECK = 3
    FOLD = 4

    @staticmethod
    def select_random():
        return random.choice(list(ACTION))


class RandomPlayer:
    def __init__(self, player_id, hand):
        self.player_id = player_id  # Unique identifier for the player
        self.hand = Hand(hand)  # Initial hand dealt to the player (two cards)
        self.money = 1_000


    def act(self, max_raise=0):
        if self.money == 0:
            return ACTION.FOLD, 0, 0

        action = ACTION.select_random()

        if action == ACTION.RAISE:
            amount = min(25, max_raise)
            self.money -= amount

            return ACTION.RAISE, amount, self.money

        return action, 0, self.money
