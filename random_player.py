import math
import random

from action import ACTION, ActionResult
from hand import Hand

MIN_RAISE = 25


class RandomPlayer:
    def __init__(self, player_id, hand):
        self.player_id = player_id  # Unique identifier for the player
        self.hand = Hand(hand)  # Initial hand dealt to the player (two cards)
        self.money = 1_000
        self.total_bet = 0  # Total amount contributed to pot for the current round
        self.last_action = None
        self.bets = []
        self.actions = []
        self.bets = []

    # current_bet is the current bet for the round
    # max_bet is maximal allowed bet for the round
    def act(self, current_bet, max_bet):
        bet = 0
        if self.last_action == ACTION.FOLD or (self.money == 0 and self.total_bet == 0):
            action = ACTION.FOLD
        elif self.total_bet == max_bet:
            action = ACTION.CHECK
        elif current_bet == 0:
            action = ACTION.BLIND
        elif current_bet == max_bet:
            action = ACTION.CALL
        else:
            if self.total_bet < current_bet:
                action = ACTION.select_raise_or_call(self.actions)
            else:
                action = ACTION.select_random(self.actions)

        if current_bet == max_bet:
            action = ACTION.CALL

        if action == ACTION.BLIND:
            bet = 1

        if action == ACTION.RAISE:
            # target_bet = random.randint(current_bet + 1, max_bet)
            # bet = target_bet - self.total_bet
            remainder_for_current_bet = current_bet - self.total_bet
            bet = remainder_for_current_bet + RandomPlayer.get_random_bet(max_bet - remainder_for_current_bet)
        elif action == ACTION.CALL:
            bet = current_bet - self.total_bet

        self.money -= bet
        self.total_bet += bet
        self.last_action = action
        self.bets.append(bet)
        self.actions.append(action)
        self.bets.append(bet)
        return ActionResult(action, bet)

    def prepare_for_round(self):
        self.total_bet = 0
        self.last_action = None
        self.bets = []
        self.actions = []
        self.bets = []

    @staticmethod
    def get_random_bet(upper_limit):
        if upper_limit < 2:
            return 1
        weights = [1 / pow(i, 2) for i in range(1, math.floor(upper_limit + 1))]
        lst = random.choices(range(1, math.floor(upper_limit + 1)), weights)
        return lst[0]
