from deck import Deck
from random_player import RandomPlayer, ACTION

class RandomEnv:
    def __init__(self):
        self.deck = None
        self.players = None
        self.community_cards = None
        self.pot = None
        self.reset()

    def play_round(self):
        round_raise_amount = 0
        pot = 0
        for player in self.players:
            action = player.act(round_raise_amount=round_raise_amount)
            pot += action[1]
            if action[0] == ACTION.RAISE:
                round_raise_amount += action[1]

    def play(self):
        for _ in range(3):
            self.community_cards.append(self.deck.draw())
            self.play_round()
            if self.check_game_finished():
                break

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

    def check_game_finished(self):
        for player in self.players:
            if player.last_action != ACTION.FOLD:
                return False
        return True

    def reset(self):
        self.deck = Deck()
        self.players = [RandomPlayer(i, [self.deck.draw(), self.deck.draw()]) for i in range(5)]
        self.community_cards = [self.deck.draw(), self.deck.draw()] # third card will be appended at the beginning of the first round
        self.pot = 0
