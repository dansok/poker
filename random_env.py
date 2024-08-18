from deck import Deck
from random_player import RandomPlayer


class RandomEnv:
    def __init__(self):
        self.deck = Deck()
        self.players = [RandomPlayer(i, [self.deck.draw(), self.deck.draw()]) for i in range(5)]
        self.community_cards = [self.deck.draw(), self.deck.draw(), self.deck.draw()]

    def play_round_1(self):
        actions = [player.act() for player in self.players]
        pot = sum(action.second for action in actions)
