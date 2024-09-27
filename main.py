from random_env import RandomEnv
#
# def main():
#     env = RandomEnv()
#     out_file = open('poker_data.csv', 'w')
#     out_file.write('my_card_1, my_card_2, community_card_1, community_card_2,community_card_3,community_card_4,community_card_5,weighted_output')
#     for _ in range(1000):
#         print(f'play {_}')
#         play_result = env.play()
#         out_file.write('\n' + play_result)
#     out_file.close()
#
# if __name__ == "__main__":
#     main()
#

from collections import OrderedDict

from deck import Deck
from hand import Hand
from card import Card


def main():
    with open('poker_data.csv', 'w') as out_file:
        out_file.write(
            'my_card_1,my_card_2,community_card_1,community_card_2,community_card_3,community_card_4,'
            'community_card_5,weighted_output')
        for _ in range(1_000_000):
            deck = Deck()

            hands = [Hand(cards=[deck.draw(), deck.draw()]) for _ in range(6)]
            community_cards = [deck.draw(), deck.draw(), deck.draw(), deck.draw(), deck.draw()]

            ranks = OrderedDict({hand: hand.rank_hand(community_cards=community_cards) for hand in hands})
            ranks = sorted(ranks.items(), key=lambda item: item[1])

            print(f"ranks == {ranks}")
            my_hand = hands[0]
            best_rank = ranks[len(ranks) - 1]
            weighted_score = 0
            if best_rank[0] == my_hand:
                score = best_rank[1][0]
                number_of_winners = 0
                for rank in ranks:
                    if score == rank[1][0]:
                        number_of_winners += 1
                weighted_score = 1 / number_of_winners
            all_cards = my_hand.cards + community_cards
            out_file.write('\n' + ','.join(str(Card.card_to_index(card)) for card in all_cards) + f',{weighted_score}')


if __name__ == "__main__":
    main()
