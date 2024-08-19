from deck import Deck
from random_player import RandomPlayer, ACTION



class RandomEnv:
    def __init__(self):
        self.deck = None
        self.players = None
        self.community_cards = None
        self.pot = None
        self.current_raise = None
        self.num_folds = None
        self.reset()

    def play_round(self):
        for player in self.players:
            action = player.act(current_raise=self.current_raise)
            self.pot += action[1]
            if action[0] == ACTION.RAISE:
                self.current_raise = action[1]

    def play(self):
        self.play_round()
        self.community_cards.append(self.deck.draw())
        self.play_round()
        self.community_cards.append(self.deck.draw())
        self.play_round()

        winners = []
        for player in self.players:
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
        self.community_cards = [self.deck.draw(), self.deck.draw(), self.deck.draw()]
        self.pot = 0
        self.current_raise = 0
        self.num_folds = len(self.players)
