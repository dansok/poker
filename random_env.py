from deck import Deck
from random_player import RandomPlayer


class RandomEnv:
    def __init__(self):
        self.num_folds = None
        self.max_raise = None
        self.pot = None
        self.community_cards = None
        self.players = None
        self.deck = None
        self.reset()

    def play_round(self):
        max_raise = self.max_raise

        for player in self.players:
            action = player.act(max_raise=self.max_raise)
            self.pot += action[1]
            max_raise = min(action[2], self.max_raise)

        self.max_raise = min(self.max_raise, max_raise)

    def play_round_1(self):
        self.play_round()

    def play_round_2(self):
        self.community_cards.append(self.deck.draw())

        self.play_round()

    def play_round_3(self):
        self.community_cards.append(self.deck.draw())

        self.play_round()

    def run(self):
        self.deck = None
        self.players = None
        self.community_cards = None
        self.reset()

    def reset(self):
        self.deck = Deck()
        self.players = [RandomPlayer(i, [self.deck.draw(), self.deck.draw()]) for i in range(5)]
        self.community_cards = [self.deck.draw(), self.deck.draw(), self.deck.draw()]
        self.pot = 0
        self.max_raise = self.players[0].money
        self.num_folds = len(self.players)
