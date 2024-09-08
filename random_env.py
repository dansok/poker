import sys

from deck import Deck
from random_player import RandomPlayer, ACTION


class RandomEnv:
    def __init__(self):
        self.deck = None
        self.players = None
        self.community_cards = None
        self.pot = None
        self.reset()
        self.play_count = 0

    # def play_round(self):
    #     max_raise = self.calc_max_raise();
    #
    #     for player in self.players:
    #         action = player.act(max_raise=max_raise)
    #         self.pot += action[1]
    #         max_raise = min(action[2], self.max_raise)
    #
    #     self.max_raise = min(self.max_raise, max_raise)

    def play_round(self):
        for player in self.players:
            player.prepare_for_round()
        max_bet_for_round = self.calc_max_bet_for_round()
        pot = 0
        current_bet = 0
        while not self.round_finished(current_bet):
            for player in self.players:
                action_result = player.act(current_bet, max_bet_for_round)
                pot += action_result.contribution
                if action_result.action == ACTION.RAISE:
                    current_bet += action_result.contribution
                if action_result.action == ACTION.FOLD and self.all_but_one_folded():
                    break

    # We consider round finished when all players that didn't fold contributed amount equal to current_bet
    def round_finished(self, current_bet):
        if self.all_but_one_folded():
            return True
        for player in self.players:
            if player.last_action != ACTION.FOLD and player.total_contribution != current_bet:
                return False
        return True

    def all_but_one_folded(self):
        number_of_folds = 0
        for player in self.players:
            if player.last_action == ACTION.FOLD:
                number_of_folds += 1
        return number_of_folds == len(self.players) - 1



    def calc_max_bet_for_round(self):
        result = None
        for player in self.players:
            if player.money > 0:
                if result is None:
                    result = player.money
                else:
                    result = min(result, player.money)
        return result

    def play(self):
        self.play_count += 1
        self.reset()
        for _ in range(3):
            self.community_cards.append(self.deck.draw())
            self.play_round()

        winners = []
        for player in self.players:
            if player.last_action != ACTION.FOLD:
                rank = player.hand.rank_hand(self.community_cards)
                if len(winners) == 0 or winners[0][1][0] < rank[0]:
                    winners = [[player, rank]]
                elif winners[0][1][0] == rank[0]:
                    winners.append([player, rank])
        for winner in winners:
            winner[0].money += self.pot / len(winners)

    def reset(self):
        self.deck = Deck()
        self.players = [RandomPlayer(i, [self.deck.draw(), self.deck.draw()]) for i in range(5)]
        self.community_cards = [self.deck.draw(), self.deck.draw()] # third card will be appended at the beginning of the first round
        self.pot = 0
