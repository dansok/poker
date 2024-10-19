import numpy as np
import torch
import torch.nn as nn
from torch import optim
from approximators import PolicyNetwork, ValueNetwork  # Assuming these are custom modules
from card import Card  # Assuming Card has card_to_one_hot method
from hand import Hand  # Assuming Hand is a custom class for storing player hand


class Player:
    def __init__(self, player_id, hand):
        self.player_id = player_id  # Unique identifier for the player
        self.hand = Hand(hand)  # Initial hand dealt to the player (two cards)

        # Networks initialized with dynamic input dimension
        self.policy_network = PolicyNetwork(
            input_dim=52 * (2 + 5) + 1,  # 5 community cards (max) + 2 hand cards + pot value
            hidden_dim_1=256,
            hidden_dim_2=128,
            hidden_dim_3=64,
            output_dim=3,
        )
        self.value_network = ValueNetwork(
            input_dim=52 * (2 + 5) + 1,  # 5 community cards (max) + 2 hand cards + pot value
            hidden_dim_1=256,
            hidden_dim_2=128,
            hidden_dim_3=64,
        )

        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.001)

    def get_observation(self, community_cards, pot):
        """Generate the observation vector for the player."""
        # Encode the player's hand (always two cards)
        hand_encoding = np.concatenate([Card.card_to_one_hot(card) for card in self.hand.cards])

        # Encode the community cards, pad with zeros if less than 5 community cards
        community_cards_encoding = np.concatenate([
            Card.card_to_one_hot(card) if card is not None else np.zeros(52, dtype=np.float64)
            for card in community_cards
        ])

        # Pad with zero vectors to ensure exactly 5 community card encodings
        missing_cards = 5 - len(community_cards)
        if missing_cards > 0:
            community_cards_encoding = np.concatenate([
                community_cards_encoding,
                np.zeros(52 * missing_cards, dtype=np.float64)
            ])

        # Encode the pot value
        pot_encoding = np.array([pot], dtype=np.float64)

        # Concatenate hand, community cards, and pot encoding to form the observation
        return np.concatenate([hand_encoding, community_cards_encoding, pot_encoding])

    def get_action_and_value(self, community_cards, pot):
        """Get the action and value for the current observation."""
        observation = self.get_observation(community_cards, pot)
        observation_tensor = torch.tensor(observation, dtype=torch.float64).unsqueeze(0)

        # Forward pass through policy and value networks
        action_probabilities = self.policy_network(observation_tensor)
        value = self.value_network(observation_tensor)

        # Sample action based on the probabilities
        action = torch.multinomial(action_probabilities[0], num_samples=1).item()

        return action, value.item()

    def train(self, observation, action, reward):
        """Train the policy and value networks."""
        observation_tensor = torch.tensor(observation, dtype=torch.float64).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.long).unsqueeze(0)
        reward_tensor = torch.tensor(reward, dtype=torch.float64).unsqueeze(0)

        # Forward pass
        action_probabilities = self.policy_network(observation_tensor)
        value = self.value_network(observation_tensor)

        # Compute policy loss
        action_probability = action_probabilities[0, action_tensor]
        policy_loss = -torch.log(action_probability) * (reward_tensor - value)

        # Compute value loss
        value_loss = nn.MSELoss()(value, reward_tensor)

        # Optimize policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Optimize value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def act(self, community_cards, pot, epsilon=0.0):
        """
        Selects an action for the player based on the current game state.

        Args:
            community_cards (list): A list of community cards (can be fewer than 5).
            pot (float): The current size of the pot.
            epsilon (float): Probability of taking a random action for exploration (default=0.0).

        Returns:
            int: The selected action (0: Fold, 1: Check/Call, 2: Raise).
            float: The predicted value of the current state.
        """
        # Step 1: Get the current observation (hand, community cards, and pot)
        observation = self.get_observation(community_cards, pot)
        observation_tensor = torch.tensor(observation, dtype=torch.float64).unsqueeze(0)

        # Step 2: Forward pass through the policy network to get action probabilities
        action_probabilities = self.policy_network(observation_tensor).detach().numpy()[0]

        # Step 3: Epsilon-greedy exploration: With epsilon probability, take a random action
        if np.random.rand() < epsilon:
            action = np.random.choice(len(action_probabilities))  # Random action (0, 1, or 2)
        else:
            # Step 4: Select action based on the highest probability (greedy policy)
            action = np.argmax(action_probabilities)

        # Step 5: Get the value prediction for the current observation
        value = self.value_network(observation_tensor).item()

        return action, value

    @staticmethod
    def _card_to_index(card):
        """Convert a card to a unique index."""
        rank_index = Card.RANK_ORDER[card.rank]
        suit_index = "♠♥♣♦".index(card.suit)
        return rank_index * 4 + suit_index

    @staticmethod
    def _card_to_one_hot(card):
        """Convert a card to a one-hot encoded vector."""
        one_hot = np.zeros(52, dtype=np.float64)
        index = Player._card_to_index(card)
        one_hot[index] = 1
        return one_hot
