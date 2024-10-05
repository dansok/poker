import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random

# Define the padding token and number of cards
PAD_TOKEN = 52  # We assume cards are indexed 0-51, 52 is for padding


# PokerHandPredictor model
class PokerHandPredictor(nn.Module):
    def __init__(self, num_cards=53, d_model=64, nhead=4, num_encoder_layers=3):
        super(PokerHandPredictor, self).__init__()

        # Embedding layer for cards (52 unique cards + PAD token)
        self.card_embedding = nn.Embedding(num_cards, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Output layer for predicting win, draw, and loss
        self.fc = nn.Linear(d_model, 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, card_sequence, src_key_padding_mask=None):
        # Embed card sequences
        embedded_cards = self.card_embedding(card_sequence)  # [batch_size, seq_length, d_model]

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(embedded_cards, src_key_padding_mask=src_key_padding_mask)

        # Pooling (mean pooling over the sequence length)
        pooled_output = transformer_output.mean(dim=1)

        # Output probabilities for win, draw, and loss
        logits = self.fc(pooled_output)
        probs = self.softmax(logits)

        return probs


# Create padding mask (shape: [batch_size, seq_len])
def create_padding_mask(sequence, pad_token=PAD_TOKEN):
    return sequence == pad_token


# Custom Dataset class to load poker data from CSV
class PokerDataset(Dataset):
    def __init__(self, csv_file, pad_token=52):
        # Load the data from CSV
        self.data = pd.read_csv(csv_file)
        self.pad_token = pad_token  # Set the padding token (52)

    @staticmethod
    def _convert_outcome(outcome):
        # Convert outcome to class index (0: Loss, 1: Draw, 2: Win)
        if outcome == 1.0:  # Win
            return 2
        elif outcome == 0.0:  # Loss
            return 0
        else:  # Draw (if needed)
            return 1

    def _random_mask(self, cards):
        # Randomly choose one of three scenarios:
        # 1. Full 5 community cards
        # 2. 4 community cards + 1 PAD_TOKEN
        # 3. 3 community cards + 2 PAD_TOKEN

        scenario = random.choice([5, 4, 3])  # Randomly choose the number of community cards
        if scenario == 4:
            cards[-1] = self.pad_token  # Mask community_card_5
        elif scenario == 3:
            cards[-1] = self.pad_token  # Mask community_card_5
            cards[-2] = self.pad_token  # Mask community_card_4

        return cards

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the row corresponding to the current index
        row = self.data.iloc[idx]

        # Extract the player's two cards and five community cards
        cards = row[['my_card_1', 'my_card_2', 'community_card_1', 'community_card_2',
                     'community_card_3', 'community_card_4', 'community_card_5']].values

        # Convert to list of integers
        cards = [int(c) for c in cards]

        # Randomly mask community cards to simulate 3, 4, or 5 community cards
        cards = self._random_mask(cards)

        # Convert to torch tensor
        cards = torch.tensor(cards, dtype=torch.long)

        # Convert outcome to target class
        outcome = self._convert_outcome(row['weighted_output'])  # Convert to 0 (Loss), 1 (Draw), or 2 (Win)

        return cards, outcome


# Create DataLoader
def create_dataloaders(csv_file, batch_size=32):
    dataset = PokerDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Training function
def train(model, dataloader, loss_fn, optimizer, num_epochs=10):
    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch, (cards, outcome) in enumerate(dataloader):
            optimizer.zero_grad()  # Clear previous gradients

            # Generate padding mask
            padding_mask = create_padding_mask(cards).T

            # Forward pass through the model
            predictions = model(cards, src_key_padding_mask=padding_mask)

            # Compute loss
            loss = loss_fn(predictions, outcome)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")


# Function to evaluate 20 random data points after training
def evaluate_random_data(model, dataloader, num_samples=20):
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # No gradient computation during evaluation
        samples_evaluated = 0
        for cards, outcome in dataloader:
            # Generate padding mask
            padding_mask = create_padding_mask(cards).T

            # Get predictions
            predictions = model(cards, src_key_padding_mask=padding_mask)

            # Print probabilities and actual labels
            for i in range(cards.size(0)):
                if samples_evaluated >= num_samples:
                    return

                print(f"Sample {samples_evaluated + 1}:")
                print("Card sequence:", cards[i].tolist())
                print("Predicted probabilities (Loss, Draw, Win):", predictions[i].tolist())
                print("Actual outcome:", outcome[i].item())
                print("Predicted class:", torch.argmax(predictions[i]).item())
                print("-" * 50)

                samples_evaluated += 1


# Main function to load data, train the model, and evaluate
def main():
    # Hyperparameters
    num_cards = 53  # 52 unique cards + 1 PAD token
    d_model = 64  # Embedding size
    nhead = 4  # Number of attention heads
    num_encoder_layers = 3  # Transformer layers
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.01

    # Create the model
    model = PokerHandPredictor(num_cards=num_cards, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers)

    # Loss function (CrossEntropy for classification)
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load dataset and create DataLoader
    train_dataloader = create_dataloaders('poker_data.csv', batch_size=batch_size)

    # Train the model
    train(model, train_dataloader, loss_fn, optimizer, num_epochs=num_epochs)

    # Evaluate 20 random data points after training
    print("\nEvaluating 20 random samples:")
    evaluate_random_data(model, train_dataloader, num_samples=20)


if __name__ == "__main__":
    main()
