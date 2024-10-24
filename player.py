import numpy as np
import torch
import torch.nn as nn
from torch import optim
import math
import random
from action import ACTION
from approximators import PolicyNetwork, ValueNetwork
from card import Card
from hand import Hand


class Player:
    def __init__(self, player_id, hand):
        self.player_id = player_id  # Unique identifier for the player
        self.hand = Hand(hand)  # Initial hand dealt to the player (two cards)
        self.money = 1_000
        self.total_contribution = 0
        self.last_action = None
        self.bets = []
        self.actions = []
        self.contributions = []
        self.experience_buffer = []  # Store experiences for training

        # Networks initialized with dynamic input dimension
        self.policy_network = PolicyNetwork(
            input_dim=52 * (2 + 5) + 1,  # 5 community cards (max) + 2 hand cards + pot value
            hidden_dim_1=256,
            hidden_dim_2=128,
            hidden_dim_3=64,
            output_dim=len(ACTION),  # Output dimension matches number of actions
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
        hand_encoding = np.concatenate([Player._card_to_one_hot(card) for card in self.hand.cards])

        community_cards_encoding = np.concatenate([
            Player._card_to_one_hot(card) if card is not None else np.zeros(52, dtype=np.float64)
            for card in community_cards
        ])

        missing_cards = 5 - len(community_cards)
        if missing_cards > 0:
            community_cards_encoding = np.concatenate([
                community_cards_encoding,
                np.zeros(52 * missing_cards, dtype=np.float64)
            ])

        pot_encoding = np.array([pot], dtype=np.float64)
        return np.concatenate([hand_encoding, community_cards_encoding, pot_encoding])

    def get_action_and_value(self, community_cards, pot):
        """Get the action and value for the current observation."""
        observation = self.get_observation(community_cards, pot)
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        action_probabilities = self.policy_network(observation_tensor)
        value = self.value_network(observation_tensor)

        action_index = torch.multinomial(action_probabilities[0], num_samples=1).item()
        action = ACTION(action_index)

        return action, value.item()

    def act(self, current_bet, max_bet, community_cards, pot, epsilon=0.0):
        """
        Selects an action for the player based on the current game state, respecting the game's rules.

        Args:
            current_bet (float): Current bet for the round.
            max_bet (float): Max possible bet for the round.
            community_cards (list): A list of community cards (can be fewer than 5).
            pot (float): The current size of the pot.
            epsilon (float): Probability of taking a random action for exploration (default=0.0).

        Returns:
            action: The selected action.
            float: The predicted value of the current state.
            contribution: Contribution made by the player in the selected action.
        """
        contribution = 0
        observation = self.get_observation(community_cards, pot)
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        action_probabilities = self.policy_network(observation_tensor)
        value = self.value_network(observation_tensor)

        if np.random.rand() < epsilon:
            action_index = np.random.choice(len(ACTION))  # Random action index
        else:
            valid_actions = self.get_valid_actions(current_bet, max_bet)
            masked_action_probabilities = action_probabilities.clone()
            masked_action_probabilities[0, ~valid_actions] = -float('inf')  # Mask invalid actions

            action_index = torch.multinomial(torch.softmax(masked_action_probabilities, dim=-1)[0], 1).item()

        action = ACTION(action_index)

        # Handle forced actions based on game rules
        if action == ACTION.FOLD or (self.money == 0 and self.total_contribution == 0):
            action = ACTION.FOLD
        elif self.total_contribution == max_bet:
            action = ACTION.CHECK
        elif current_bet == 0:
            action = ACTION.BLIND
        elif current_bet == max_bet:
            action = ACTION.CALL
        else:
            if action == ACTION.RAISE:
                remainder_for_current_bet = current_bet - self.total_contribution
                contribution = remainder_for_current_bet + self.get_random_bet(max_bet - remainder_for_current_bet)
            elif action == ACTION.CALL:
                contribution = current_bet - self.total_contribution

        self.money -= contribution
        self.total_contribution += contribution
        self.last_action = action
        self.bets.append(contribution)
        self.actions.append(action)
        self.contributions.append(contribution)

        if action == ACTION(action_index):  # Action matches the network's suggestion
            self.experience_buffer.append((observation, action, value))

        return action, value.item(), contribution

    def train(self, final_reward, gamma=0.99):
        """
        Train the policy and value networks using experiences stored in the buffer.

        Args:
            final_reward (float): The reward obtained at the end of the round.
            gamma (float): Discount factor for future rewards (default=0.99).
        """
        if len(self.experience_buffer) == 0:
            return

        G = final_reward  # Initialize return with the final reward
        for observation, action, _ in reversed(self.experience_buffer):
            observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor(action.value, dtype=torch.long).unsqueeze(0)
            reward_tensor = torch.tensor(G, dtype=torch.float32).unsqueeze(0)

            # Forward pass for both policy and value networks
            action_probabilities = self.policy_network(observation_tensor)
            value = self.value_network(observation_tensor)

            # Compute policy loss (REINFORCE rule)
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

            # Update the return for the next timestep (G = r + gamma * G)
            G = gamma * G

        self.experience_buffer = []  # Clear the experience buffer after training

    def prepare_for_round(self):
        self.total_contribution = 0
        self.last_action = None
        self.bets = []
        self.actions = []
        self.contributions = []
        self.experience_buffer = []  # Reset experiences for the new round

    def get_valid_actions(self, current_bet, max_bet):
        """
        Returns a boolean mask indicating which actions are valid in the current game state.
        """
        valid_actions = np.ones(len(ACTION), dtype=bool)  # All actions valid by default

        if self.money == 0 or self.last_action == ACTION.FOLD:
            valid_actions[:] = False
            valid_actions[ACTION.FOLD.value] = True
            return valid_actions

        if self.total_contribution == max_bet:
            valid_actions[:] = False
            valid_actions[ACTION.CHECK.value] = True
            valid_actions[ACTION.FOLD.value] = True
            return valid_actions

        if current_bet == 0:
            valid_actions[ACTION.BLIND.value] = True

        if current_bet > 0 and current_bet > self.total_contribution:
            valid_actions[ACTION.CALL.value] = True

        if self.money > current_bet:
            valid_actions[ACTION.RAISE.value] = True

        return valid_actions

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

    @staticmethod
    def get_random_bet(upper_limit):
        if upper_limit < 2:
            return 1
        weights = [1 / pow(i, 2) for i in range(1, math.floor(upper_limit + 1))]
        lst = random.choices(range(1, math.floor(upper_limit + 1)), weights)
        return lst[0]
