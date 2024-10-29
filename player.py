import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from action import ACTION
from approximators import PolicyNetwork, ValueNetwork, RaiseAmountNetwork
from card import Card
from hand import Hand


class Player:
    def __init__(self, player_id, hand):
        self.player_id = player_id
        self.hand = Hand(hand)
        self.money = 1_000
        self.total_bet = 0
        self.last_action = None
        self.bets = []
        self.actions = []
        self.experience_buffer = []

        input_dim = 52 * (2 + 5) + 1
        self.policy_network = PolicyNetwork(input_dim, 512, 256, 128, 64, len(ACTION))
        self.value_network = ValueNetwork(input_dim, 512, 256, 128, 64)
        self.raise_amount_network = RaiseAmountNetwork(input_dim, 256, 128)

        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.001)
        self.raise_amount_optimizer = optim.Adam(self.raise_amount_network.parameters(), lr=0.001)

    def get_observation(self, community_cards, pot):
        hand_encoding = np.concatenate([Player._card_to_one_hot(card) for card in self.hand.cards])
        community_cards_encoding = np.concatenate([
            Player._card_to_one_hot(card) if card is not None else np.zeros(52, dtype=np.float64)
            for card in community_cards
        ])
        if len(community_cards) < 5:
            community_cards_encoding = np.concatenate(
                [community_cards_encoding, np.zeros(52 * (5 - len(community_cards)), dtype=np.float64)]
            )
        pot_encoding = np.array([pot], dtype=np.float64)
        return np.concatenate([hand_encoding, community_cards_encoding, pot_encoding])

    def get_action_and_value(self, community_cards, pot):
        observation = self.get_observation(community_cards, pot)
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        action_probabilities = self.policy_network(observation_tensor)
        value = self.value_network(observation_tensor)
        action_index = torch.multinomial(action_probabilities[0], num_samples=1).item()
        action = ACTION(action_index)
        return action, value.item()

    def act(self, current_bet, max_bet, community_cards, pot, epsilon=0.0):
        bet = 0
        observation = self.get_observation(community_cards, pot)
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        action_probabilities = self.policy_network(observation_tensor)
        value = self.value_network(observation_tensor)

        is_learned_experience: bool = False

        if np.random.rand() < epsilon:
            action_index = np.random.choice(len(ACTION))
        else:
            # Ensure valid_actions is a PyTorch boolean tensor
            valid_actions = torch.tensor(self.get_valid_actions(current_bet, max_bet)).to(torch.bool)

            # Check for NaN or inf in the original action probabilities
            if torch.isnan(action_probabilities).any() or torch.isinf(action_probabilities).any():
                raise ValueError("action_probabilities contains NaN or inf values before masking.")

            masked_action_probabilities = action_probabilities.clone()
            masked_action_probabilities[0, ~valid_actions] = -float('inf')

            # Check if all actions are masked
            if not valid_actions.any():
                raise ValueError("No valid actions available; all actions are masked.")

            # Check for NaN or inf after masking
            if torch.isnan(masked_action_probabilities).any() or torch.isinf(masked_action_probabilities).any():
                raise ValueError("masked_action_probabilities contains NaN or inf values after masking.")

            # Sample an action based on the softmax probabilities
            action_index = torch.multinomial(torch.softmax(masked_action_probabilities, dim=-1)[0], 1).item()

        action = ACTION(action_index)

        if action == ACTION.FOLD or self.last_action == ACTION.FOLD or (
                self.money == 0 and self.total_bet == 0):
            action = ACTION.FOLD
        elif self.total_bet >= max_bet:
            action = ACTION.CHECK
        elif current_bet == 0:
            action = ACTION.RAISE
            bet = 1
        elif current_bet >= max_bet:
            action = ACTION.CALL
            bet = current_bet - self.total_bet
        else:
            is_learned_experience = True

            if action == ACTION.RAISE:
                raise_amount = self.raise_amount_network(observation_tensor).item()
                bet = min(current_bet - self.total_bet + raise_amount, self.money, max_bet)
            elif action == ACTION.CALL:
                bet = current_bet - self.total_bet

        self.money -= bet
        self.total_bet += bet
        self.last_action = action
        self.bets.append(bet)
        self.actions.append(action)
        self.bets.append(bet)

        if is_learned_experience:
            self.experience_buffer.append((observation, action, value, bet))

        return action, value.item(), bet

    def train(self, win_outcome, gamma=0.99):
        """Train the policy, value, and raise amount networks based on win/loss outcome."""
        if not self.experience_buffer:
            return

        G = win_outcome  # Set G to 1 if win, 0 if lose
        for observation, action, _, bet in reversed(self.experience_buffer):
            observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor(action.value, dtype=torch.long).unsqueeze(0)
            reward_tensor = torch.tensor([G], dtype=torch.float32).unsqueeze(0)

            # Policy and value network forward pass
            action_probabilities = self.policy_network(observation_tensor)
            value = self.value_network(observation_tensor)

            # Policy loss
            action_probability = action_probabilities[0, action_tensor]
            policy_loss = -torch.log(action_probability) * (
                        reward_tensor - torch.sigmoid(value))  # Sigmoid for probability

            # Value loss for probability of winning
            value_loss = F.binary_cross_entropy_with_logits(value, reward_tensor)

            # Optimize policy network
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.policy_optimizer.step()

            # Optimize value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # Train raise_amount_network if action was RAISE
            if action == ACTION.RAISE:
                predicted_raise = self.raise_amount_network(observation_tensor)
                actual_raise_tensor = torch.tensor([bet], dtype=torch.float32)  # True raise amount

                # Ensure both tensors have matching shapes
                predicted_raise = predicted_raise.view(1)
                actual_raise_tensor = actual_raise_tensor.view(1)

                # Compute MSE loss
                raise_amount_loss = F.mse_loss(predicted_raise, actual_raise_tensor)

                self.raise_amount_optimizer.zero_grad()
                raise_amount_loss.backward()
                self.raise_amount_optimizer.step()

            # Update G for next step (still gamma-discounted, even though binary)
            G = gamma * G

        # Clear experience buffer
        self.experience_buffer = []

    def prepare_for_round(self):
        self.total_bet = 0
        self.last_action = None
        self.bets = []
        self.actions = []
        self.experience_buffer = []

    def get_valid_actions(self, current_bet, max_bet):
        valid_actions = np.ones(len(ACTION), dtype=bool)
        if self.money == 0 or self.last_action == ACTION.FOLD:
            valid_actions[:] = False
            valid_actions[ACTION.FOLD.value] = True
            return valid_actions
        if self.total_bet == max_bet:
            valid_actions[:] = False
            valid_actions[ACTION.CHECK.value] = True
            valid_actions[ACTION.FOLD.value] = True
            return valid_actions
        if current_bet == 0:
            valid_actions[ACTION.RAISE.value] = True
        if current_bet > 0 and current_bet > self.total_bet:
            valid_actions[ACTION.CALL.value] = True
        if self.money > current_bet:
            valid_actions[ACTION.RAISE.value] = True
        return valid_actions

    @staticmethod
    def _card_to_index(card):
        rank_index = Card.RANK_ORDER[card.rank]
        suit_index = "♠♥♣♦".index(card.suit)
        return rank_index * 4 + suit_index

    @staticmethod
    def _card_to_one_hot(card):
        one_hot = np.zeros(52, dtype=np.float64)
        index = Player._card_to_index(card)
        one_hot[index] = 1
        return one_hot

    @staticmethod
    def get_random_bet(upper_limit):
        if upper_limit < 2:
            return 1
        weights = [1 / pow(i, 2) for i in range(1, math.floor(upper_limit + 1))]
        choices = random.choices(range(1, math.floor(upper_limit + 1)), weights)
        return choices[0]
