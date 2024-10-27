from env import Env


def main():
    # env = RandomEnv()
    env = Env()
    with open("play_results.csv", "w") as output_file:
        output_file.write("my_card_1,my_card_2,community_card_1,community_card_2,community_card_3,community_card_4,"
                          "community_card_5,weighted_output")
        for _ in range(100_000):
            print(f"play {_}")
            play_result = env.play()
            output_file.write(f"\n{play_result}")
    env.render()


# def main():
#     with open("poker_data.csv", "w") as out_file:
#         out_file.write(
#             "my_card_1,my_card_2,community_card_1,community_card_2,community_card_3,community_card_4,"
#             "community_card_5,weighted_output")
#         for cnt in range(1_000_000):
#             deck = Deck()
#
#             hands = [Hand(cards=[deck.draw(), deck.draw()]) for _ in range(6)]
#             community_cards = [deck.draw() for _ in range(5)]
#
#             ranks = OrderedDict({hand: hand.rank_hand(community_cards=community_cards) for hand in hands})
#             ranks = sorted(ranks.items(), key=lambda item: item[1])
#
#             if cnt % 1000 == 0:
#                 print(f"run# {cnt} ranks == {ranks}")
#             my_hand = hands[0]
#             best_rank = ranks[len(ranks) - 1]
#             weighted_score = 0
#             if best_rank[0] == my_hand:
#                 score = best_rank[1][0]
#                 number_of_winners = 0
#                 for rank in ranks:
#                     if score == rank[1][0]:
#                         number_of_winners += 1
#                 weighted_score = 1 / number_of_winners
#             all_cards = my_hand.cards + community_cards
#             out_file.write(f"\n{','.join(str(Card.card_to_index(card)) for card in all_cards)},{weighted_score}")


if __name__ == "__main__":
    main()
