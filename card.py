import numpy as np


class Card:
    RANK_ORDER = {
        '2': 0,
        '3': 1,
        '4': 2,
        '5': 3,
        '6': 4,
        '7': 5,
        '8': 6,
        '9': 7,
        'T': 8,
        'J': 9,
        'Q': 10,
        'K': 11,
        'A': 12,
    }

    SUITS = "♠♥♣♦"

    def __init__(self, rank, suit):
        if rank not in Card.RANK_ORDER:
            raise ValueError(f"Invalid rank: {rank}")
        if suit not in Card.SUITS:
            raise ValueError(f"Invalid suit: {suit}")

        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank}{self.suit}"

    def order(self):
        """Return the rank value for comparison."""
        return Card.RANK_ORDER[self.rank]

    def __lt__(self, other):
        """Compare two cards based on rank."""
        return self.order() < other.order()

    def __le__(self, other):
        """Compare two cards based on rank."""
        return self.order() <= other.order()

    def __eq__(self, other):
        """Check if two cards are equal based on rank."""
        return self.order() == other.order()

    def __ne__(self, other):
        """Check if two cards are not equal based on rank."""
        return self.order() != other.order()

    def __gt__(self, other):
        """Compare two cards based on rank."""
        return self.order() > other.order()

    def __ge__(self, other):
        """Compare two cards based on rank."""
        return self.order() >= other.order()

    @staticmethod
    def card_to_index(card):
        """Convert a card to a unique index."""
        rank_index = Card.RANK_ORDER[card.rank]
        suit_index = Card.SUITS.index(card.suit)
        return rank_index * 4 + suit_index

    @staticmethod
    def card_to_one_hot(card):
        """Convert a card to a one-hot encoded vector."""
        one_hot = np.zeros(52, dtype=np.float64)
        index = Card.card_to_index(card)
        one_hot[index] = 1
        return one_hot
