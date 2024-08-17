import random

from card import Card


class Deck:
    SUITS = "♠♥♣♦"
    RANKS = "23456789TJQKA"

    def __init__(self):
        """Initialize the deck with all 52 cards."""
        self.cards = [Card(rank, suit) for suit in Deck.SUITS for rank in Deck.RANKS]
        self.shuffle()

    def __repr__(self):
        """Return a string representation of the deck."""
        return f"Deck({self.cards})"

    def shuffle(self):
        """Shuffle the deck of cards in place."""
        random.shuffle(self.cards)

    def draw(self):
        """Draw the top card from the deck. Returns None if the deck is empty."""
        return self.cards.pop() if self.cards else None
