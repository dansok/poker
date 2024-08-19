import random
from enum import Enum

from hand import Hand


MIN_RAISE = 25


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
        self.last_action = None

    def act(self, round_raise_amount = 0):
        action = self.select_action(round_raise_amount)
        add_to_pot = 0
        if action == ACTION.RAISE:
            add_to_pot = round_raise_amount + random.randint(1, self.money - round_raise_amount)
        elif action == ACTION.CALL:
            add_to_pot = round_raise_amount
        self.money -= add_to_pot
        self.last_action = action
        return action, add_to_pot

    def select_action(self, current_raise_amount):
        if self.last_action == ACTION.FOLD:
            return ACTION.FOLD

        if self.money == 0 or self.money < current_raise_amount:
            action = ACTION.FOLD
        else:
            action = ACTION.select_random()

        if action == ACTION.CALL and current_raise_amount == 0:
            action = ACTION.CHECK
        if action == ACTION.RAISE and current_raise_amount == self.money:
            action = ACTION.CALL

        return action
