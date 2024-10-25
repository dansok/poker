from card import Card
from deck import Deck
from action import ACTION
from player import Player


class Env:
    def __init__(self):
        self.deck = None
        self.players = None
        self.community_cards = None
        self.pot = 0
        self.reset()
        self.play_count = 0

    def play_round(self):
        for player in self.players:
            player.prepare_for_round()
        max_bet_for_round = self.calc_max_bet_for_round()
        # print('max_bet_for_round', max_bet_for_round)
        self.pot = 0
        current_bet = 0
        while max_bet_for_round > 0 and not self.round_finished(current_bet):
            for player in self.players:
                if player.last_action != ACTION.FOLD:
                    action, _, contribution = player.act(current_bet, max_bet_for_round, self.community_cards, self.pot)
                    self.pot += contribution
                    current_bet = max(current_bet, player.total_contribution)
                    if action == ACTION.FOLD and self.all_but_one_folded():
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

        for player in self.players:
            print('bbbbb', self.pot, player.player_id, player.money, player.actions, player.contributions, sum(player.contributions))

    # We consider round finished when all players that didn't fold contributed amount equal to current_bet
    def round_finished(self, current_bet):
        if self.all_but_one_folded():
            return True
        for player in self.players:
            if player.last_action != ACTION.FOLD and player.total_contribution < current_bet:
                return False
        return current_bet > 0

    def all_but_one_folded(self):
        number_of_folds = 0
        for player in self.players:
            if player.last_action == ACTION.FOLD:
                number_of_folds += 1
        return number_of_folds == len(self.players) - 1

    def calc_max_bet_for_round(self):
        result = 0
        for player in self.players:
            if player.money > 0:
                if result == 0:
                    result = player.money
                else:
                    result = min(result, player.money)
        return result

    def play(self):
        self.play_count += 1
        self.reset()
        player0 = self.players[0]
        player0_money_before_round = player0.money
        for _ in range(3):
            # print(f'round {_}')
            self.community_cards.append(self.deck.draw())
            money_before_round = [player.money for player in self.players]
            self.play_round()
            for i, player in enumerate(self.players):
                profit = player.money - money_before_round[i]
                player.train(profit)

            # print(f'state as of the end of round {_}')
            # self.render()

        out_list = player0.hand.cards + self.community_cards
        play_result = ''
        for card in out_list:
            if play_result > '':
                play_result += f',{Card.card_to_index(card)}'
            else:
                play_result = f'{Card.card_to_index(card)}'
        proceeds = player0.money - player0_money_before_round
        play_result += f',{proceeds}'

        return play_result

    def reset(self):
        self.deck = Deck()
        self.players = [Player(i, [self.deck.draw(), self.deck.draw()]) for i in range(5)]
        self.community_cards = [self.deck.draw(),
                                self.deck.draw()]  # third card will be appended at the beginning of the first round
        self.pot = 0

    def render(self):
        print(f'pot: {self.pot}')
        print('=================================')
        for player in self.players:
            print(
                f"""player {player.player_id}
cards: {player.hand.cards}
money: {player.money}
bets: {player.bets}
actions: {player.actions}
================================="""
            )
