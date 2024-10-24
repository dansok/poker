import numpy as np
import gymnasium as gym
from gymnasium import spaces

from card import Card
from deck import Deck
from hand import Hand

# Define action constants
FOLD = 0
CALL = 1
RAISE = 2

INITIAL_BALANCE = 1000
SMALL_BLIND = 10
BIG_BLIND = 20


class TexasHoldemEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(TexasHoldemEnv, self).__init__()

        # Define the action space: 0 = Fold, 1 = Check/Call, 2 = Raise
        self.action_space = spaces.Discrete(3)

        # Define the observation space (one-hot encoded hand + community cards + pot value)
        self.observation_space = spaces.Dict({
            "hand": spaces.Box(0, 1, (52 * 2,), dtype=np.int64),  # Two cards, each one-hot encoded (52)
            "community_cards": spaces.Box(0, 1, (52 * 5,), dtype=np.int64),  # Five community cards, each one-hot encoded (52)
            "pot": spaces.Box(0, float('inf'), (1,), dtype=np.float64),  # Pot value as a single float
        })

        self.deck = Deck()
        self.community_cards = []
        self.players_hands = [Hand([]) for _ in range(6)]  # Initialize hands with Hand objects
        self.players_balances = [INITIAL_BALANCE for _ in range(6)]
        self.current_player = 0
        self.pot = 0  # Initialize pot value
        self.small_blind_position = 0
        self.big_blind_position = 1

    def reset(self):
        """Reset the environment to an initial state."""
        self.deck = Deck()
        self.deck.shuffle()
        self.community_cards = []

        # Deal two cards to each player and wrap in Hand objects
        self.players_hands = [Hand([self.deck.draw(), self.deck.draw()]) for _ in range(6)]
        self.current_player = 0

        # Reset the community cards and pot
        self.community_cards = [None] * 5
        self.pot = 0

        # Blinds
        self.small_blind_position = (self.small_blind_position + 1) % 6
        self.big_blind_position = (self.big_blind_position + 1) % 6
        self.players_balances[self.small_blind_position] -= SMALL_BLIND
        self.players_balances[self.big_blind_position] -= BIG_BLIND
        self.pot += SMALL_BLIND + BIG_BLIND

        return self._get_observation()

    def step(self, action):
        """Execute one time step within the environment."""
        reward = 0
        done = False

        # Process the action
        if action == FOLD:
            reward = -1
            done = True
        elif action == CALL:
            reward = 0
        elif action == RAISE:
            reward = 0.1
            self.pot += 10  # Example increment for RAISE action

        # Move to the next player
        self.current_player = (self.current_player + 1) % 6

        # Add community cards (simulate dealing)
        if self.community_cards[0] is None:
            # Deal the flop
            self.community_cards[0] = self.deck.draw()
            self.community_cards[1] = self.deck.draw()
            self.community_cards[2] = self.deck.draw()
        elif self.community_cards[3] is None:
            # Deal the turn
            self.community_cards[3] = self.deck.draw()
        elif self.community_cards[4] is None:
            # Deal the river
            self.community_cards[4] = self.deck.draw()
            done = True  # Game ends after the river

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """Return the current observation with one-hot encoding."""
        player_hand = self.players_hands[self.current_player].cards
        hand_encoding = np.concatenate([Card.card_to_one_hot(card) for card in player_hand])

        community_cards_encoding = np.concatenate([
            Card.card_to_one_hot(card) if card is not None else np.zeros(52, dtype=np.int64)
            for card in self.community_cards
        ])

        pot_encoding = np.array([self.pot], dtype=np.float64)

        # Concatenate all parts to form the complete observation vector
        vector = np.concatenate([hand_encoding, community_cards_encoding, pot_encoding])

        return {
            "hand": hand_encoding,
            "community_cards": community_cards_encoding,
            "pot": pot_encoding,
            "vector": vector
        }

    def render(self, mode="human"):
        """Render the environment."""
        if mode == "human":
            output = ["+-----------------------------+", "|       Community Cards       |"]
            community = " ".join(str(card) if card else "  " for card in self.community_cards)
            output.append(f"|      {community:<21}  |")
            output.append("+-----------------------------+")
            for i, hand in enumerate(self.players_hands, start=1):
                hand_str = " ".join(str(card) for card in hand.cards)
                output.append(f"| Player {i}: {hand_str:<16} Balance: {self.players_balances[i - 1]}  |")
            output.append("+-----------------------------+")
            output.append(f"| Pot: {self.pot:<26}  |")
            output.append("+-----------------------------+")
            print("\n".join(output))
