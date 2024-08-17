from itertools import combinations


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
        suits = [card.suit for card in hand]
        unique_ranks = sorted(set(ranks), reverse=True)

        # Check for straight flush
        if self._is_straight(ranks) and self._is_flush(suits):
            return 8, ranks

        # Check for four of a kind
        if self._has_n_of_a_kind(ranks, 4):
            return 7, self._get_n_of_a_kind(ranks, 4), unique_ranks

        # Check for full house
        if self._has_n_of_a_kind(ranks, 3) and self._has_n_of_a_kind(ranks, 2):
            return 6, self._get_n_of_a_kind(ranks, 3), self._get_n_of_a_kind(ranks, 2)

        # Check for flush
        if self._is_flush(suits):
            return 5, ranks

        # Check for straight
        if self._is_straight(ranks):
            return 4, ranks

        # Check for three of a kind
        if self._has_n_of_a_kind(ranks, 3):
            return 3, self._get_n_of_a_kind(ranks, 3), unique_ranks

        # Check for two pair
        if len(self._get_pairs(ranks)) == 2:
            pairs = self._get_pairs(ranks)
            return 2, pairs, unique_ranks

        # Check for one pair
        if self._has_n_of_a_kind(ranks, 2):
            return 1, self._get_n_of_a_kind(ranks, 2), unique_ranks

        # High card
        return 0, ranks

    @staticmethod
    def _is_straight(ranks):
        """Check if the ranks form a straight."""
        rank_set = set(ranks)
        if len(rank_set) < 5:
            return False
        if max(rank_set) - min(rank_set) == 4:
            return True
        if {12, 0, 1, 2, 3}.issubset(rank_set):
            return True
        return False

    @staticmethod
    def _is_flush(suits):
        """Check if the suits form a flush."""
        return len(set(suits)) == 1

    @staticmethod
    def _has_n_of_a_kind(ranks, n):
        """Check if there are n cards of the same rank."""
        return any(ranks.count(rank) == n for rank in ranks)

    @staticmethod
    def _get_n_of_a_kind(ranks, n):
        """Get the rank of the n cards of the same rank."""
        for rank in ranks:
            if ranks.count(rank) == n:
                return rank
        return None

    @staticmethod
    def _get_pairs(ranks):
        """Get all pairs in the hand."""
        pairs = [rank for rank in set(ranks) if ranks.count(rank) == 2]
        return pairs
