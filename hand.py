from itertools import combinations
from card import Card

class Hand:
    def __init__(self, cards):
        self.cards = cards  # List of Card objects

    def rank_hand(self, community_cards):
        """Evaluate the best possible hand using the player's hand and the community cards."""
        all_cards = self.cards + community_cards
        best_hand = max(combinations(all_cards, 5), key=self._hand_rank)
        return self._hand_rank(best_hand)

    def _hand_rank(self, hand):
        """Return a value indicating the rank of the hand."""
        ranks = sorted((card.order() for card in hand), reverse=True)
        rank_count = [0] * 13
        all_cards_same_suit = True
        current_suit = None
        for card in hand:
            rank_index = Card.RANK_ORDER[card.rank]
            rank_count[rank_index] += 1
            if current_suit is None:
                current_suit = card.suit
            elif card.suit != current_suit:
                all_cards_same_suit = False
                break

        straight = self.check_straight(rank_count)
        if straight is not None:
            if all_cards_same_suit:
                # it is straight flush
                return straight + 40_000_000_000, ranks
            else:
                return straight, ranks

        four_of_a_kind = self.check_four(rank_count)
        if four_of_a_kind is not None:
            return four_of_a_kind, ranks

        full_house = self.check_full_house(rank_count)
        if full_house is not None:
            return full_house

        flush = self.check_flush(rank_count, all_cards_same_suit)
        if flush is not None:
            return flush, ranks

        three_of_a_kind = self.check_three(rank_count)
        if three_of_a_kind is not None:
            return three_of_a_kind, ranks

        two_pairs = self.check_two_pairs(rank_count)
        if two_pairs is not None:
            return two_pairs

        pair = self.check_pair(rank_count)
        if pair is not None:
            return pair

        return self.check_highest_card(rank_count)

    @staticmethod
    def check_straight(rank_count):
        base = 50_000_000_000
        # special case A2345
        if rank_count[12] == 1 and rank_count[0] == 1 and rank_count[1] == 1 and rank_count[2] == 1 and rank_count[3] == 1:
            return base + 3
        for i in range(len(rank_count)):
            if rank_count[i] == 1:
                if rank_count[i+1] == 1 and rank_count[i+2] == 1 and rank_count[i+3] == 1 and rank_count[i+4] == 1:
                    return base + i + 4
                else:
                    return None
            elif rank_count[i] > 1:
                return None

    @staticmethod
    def check_four(rank_count):
        kicker = None
        quad_rank = None
        for i in range(len(rank_count)):
            if kicker is not None and quad_rank is not None:
                break
            count = rank_count[i]
            if count > 0:
                if count == 1:
                    kicker = i
                elif count == 4:
                    quad_rank = i
                else:
                    return None
        if kicker is None or quad_rank is None:
            return None
        return 80_000_000_000 + quad_rank * 100 + kicker

    @staticmethod
    def check_full_house(rank_count):
        triple_rank = None
        pair_rank = None
        for i in range(len(rank_count)):
            if triple_rank is not None and pair_rank is not None:
                break
            count = rank_count[i]
            if count > 0:
                if count == 2:
                    pair_rank = i
                elif count == 3:
                    triple_rank = i
                else:
                    return None
        if triple_rank is None or pair_rank is None:
            return None
        return 70_000_000_000 + triple_rank * 100 + pair_rank

    @staticmethod
    def check_flush(rank_count, all_cards_same_suit):
        if not all_cards_same_suit:
            return None
        result = 60_000_000_000
        multiplier = 1
        for i in range(len(rank_count)):
            count = rank_count[i]
            if count > 0:
                for _ in range(count):
                    result += i * multiplier
                    multiplier *= 100
        return result

    @staticmethod
    def check_three(rank_count):
        triple_rank = None
        kicker1 = None
        kicker2 = None
        for i in range(len(rank_count)):
            count = rank_count[i]
            if count > 0:
                if count == 1:
                    if kicker1 is None:
                        kicker1 = i
                    elif kicker2 is None:
                        kicker2 = i
                    else:
                        return None
                elif count == 3:
                    triple_rank = i
                else:
                    return None
        if triple_rank is None or kicker1 is None or kicker2 is None:
            return None
        return 40_000_000_000 + triple_rank * 10_000 + kicker2 * 100 + kicker1

    @staticmethod
    def check_two_pairs(rank_count):
        pair1 = None
        pair2 = None
        kicker = None
        for i in range(len(rank_count)):
            count = rank_count[i]
            if count > 0:
                if count == 2:
                    if pair1 is None:
                        pair1 = i
                    elif pair2 is None:
                        pair2 = i
                elif count == 1:
                    if kicker is None:
                        kicker = 1
                    else:
                        return None
                else:
                    return None
        if pair1 is None or pair2 is None or kicker is None:
            return None
        return 30_000_000_000 + pair2 * 10_000 + pair1 * 100 + kicker

    @staticmethod
    def check_pair(rank_count):
        pair = None
        kicker1 = None
        kicker2 = None
        kicker3 = None
        for i in range(len(rank_count)):
            count = rank_count[i]
            if count > 0:
                if count == 2:
                    if pair is None:
                        pair = i
                    else:
                        return None
                elif count == 1:
                    if kicker1 is None:
                        kicker1 = 1
                    if kicker2 is None:
                        kicker2 = 1
                    if kicker3 is None:
                        kicker3 = 1
                    else:
                        return None
                else:
                    return None
        if pair is None or kicker1 is None or kicker2 is None or kicker3 is None:
            return None
        return 20_000_000_000 + pair * 1_000_000 + kicker3 * 10_000 + kicker2 * 100 + kicker1

    @staticmethod
    def check_highest_card(rank_count):
        result = 10_000_000_000
        multiplier = 1
        for i in range(len(rank_count)):
            count = rank_count[i]
            if count > 0:
                for _ in range(count):
                    result += i * multiplier
                    multiplier *= 100
        return result

