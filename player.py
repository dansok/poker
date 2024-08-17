import numpy as np
import torch
import torch.nn as nn
from torch import optim
from approximators import PolicyNetwork, ValueNetwork
from card import Card
from hand import Hand


class Player:
    def __init__(self, player_id, hand):
        self.player_id = player_id  # Unique identifier for the player
        self.hand = Hand(hand)  # Initial hand dealt to the player (two cards)

        self.policy_network = PolicyNetwork(
            input_dim=52 * (2 + 5) + 1,  # 5 community cards + 2 hand cards + pot value
            hidden_dim_1=256,
            hidden_dim_2=128,
            hidden_dim_3=64,
            output_dim=3,
        )
        self.value_network = ValueNetwork(
            input_dim=52 * (2 + 5) + 1,  # 5 community cards + 2 hand cards + pot value
            hidden_dim_1=256,
            hidden_dim_2=128,
            hidden_dim_3=64,
        )

        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.001)

    def get_observation(self, community_cards, pot):
        """Generate the observation vector for the player."""
        hand_encoding = np.concatenate([Card.card_to_one_hot(card) for card in self.hand.cards])
        community_cards_encoding = np.concatenate([
            Card.card_to_one_hot(card) if card is not None else np.zeros(52, dtype=np.float64)
            for card in community_cards
        ])
        pot_encoding = np.array([pot], dtype=np.float64)

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
