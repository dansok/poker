from random_player import ACTION


class Round:
    def __init__(
        self,
        deck,
        players,
        community_cards,
        pot,
    ):
        self.deck = deck
        self.players = players
        self.community_cards = community_cards
        self.pot = pot
        self.max_raise = self.players[0].money
        self.num_folds = len(self.players)
        self.folds = {}

    def play_round(self):
        max_raise = self.max_raise

        for player in self.players:
            if player in self.folds:
                continue

            action = player.act(max_raise=self.max_raise)

            if action[0] == ACTION.FOLD:
                self.folds |= {player}

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

    def is_round_over(self):
        return len(self.folds) == len(self.players)

    def run(self):
        self.play_round_1()

        if self.is_round_over():
            pass

