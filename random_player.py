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
    @staticmethod
    def select_raise_or_call():
        return random.choice([ACTION.CALL, ACTION.CHECK])


class ActionResult:
    def __init__(self, action, contribution):
        self.action = action
        self.contribution = contribution


class RandomPlayer:
    def __init__(self, player_id, hand):
        self.player_id = player_id  # Unique identifier for the player
        self.hand = Hand(hand)  # Initial hand dealt to the player (two cards)
        self.money = 1_000
        self.total_contribution = 0  # Total amount contributed to pot for the current round
        self.last_action = None

    # current_bet is the current bet for the round
    # max_bet is maximal allowed bet for the round
    def act(self, current_bet, max_bet):
        contribution = 0
        if self.last_action == ACTION.FOLD or (self.money == 0 and self.total_contribution == 0):
            action = ACTION.FOLD
        elif self.total_contribution == max_bet:
            action = ACTION.CHECK
        else:
            if self.total_contribution < current_bet:
                action = ACTION.select_raise_or_call()
            elif current_bet == max_bet:
                action = ACTION.CALL
            else:
                action = ACTION.select_random()

            if action == ACTION.RAISE:
                target_bet = random.randint(current_bet + 1, max_bet)
                contribution = target_bet - self.total_contribution
            elif action == ACTION.CALL:
                contribution = current_bet - self.total_contribution

        self.money -= contribution
        self.total_contribution += contribution
        self.last_action = action
        return ActionResult(action, contribution)

    def prepare_for_round(self):
        self.total_contribution = 0
        self.last_action = None
