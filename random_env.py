from deck import Deck
from random_player import RandomPlayer


class RandomEnv:
    def __init__(self):
        self.deck = Deck()
        self.players = [RandomPlayer(i, [self.deck.draw(), self.deck.draw()]) for i in range(5)]
        self.community_cards = [self.deck.draw(), self.deck.draw(), self.deck.draw()]
        self.pot = 0
        self.max_raise = self.players[0].money

    def play_round_1(self):
        max_raise = self.max_raise

        for player in self.players:
            action = player.act(max_raise=self.max_raise)
            self.pot += action[1]
            max_raise = min(action[2], self.max_raise)

        self.max_raise = min(self.max_raise, max_raise)

    def play_round_2(self):
        pass
