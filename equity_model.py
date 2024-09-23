import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Assume 52 unique cards and a PAD_TOKEN to pad sequences
PAD_TOKEN = 52  # Using 52 as the padding token (cards are 0-51)


class PokerHandPredictor(nn.Module):
    def __init__(self, num_cards=53, d_model=64, nhead=4, num_encoder_layers=3):
        super(PokerHandPredictor, self).__init__()

        # Embedding layer for card tokens (52 cards + 1 PAD_TOKEN)
        self.card_embedding = nn.Embedding(num_cards, d_model)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Final output layers to predict probabilities for win, draw, and loss
        self.fc = nn.Linear(d_model, 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, card_sequence, src_key_padding_mask=None):
        # Embed the card sequence
        embedded_cards = self.card_embedding(card_sequence)  # [batch_size, seq_length, d_model]

        # Pass through transformer encoder, with padding mask
        transformer_output = self.transformer_encoder(embedded_cards, src_key_padding_mask=src_key_padding_mask)

        # Pooling over the sequence (e.g., using mean pooling)
        pooled_output = transformer_output.mean(dim=1)

        # Final linear layer to get 3 probabilities (win, draw, loss)
        logits = self.fc(pooled_output)
        probs = self.softmax(logits)

        return probs


# Helper function to create padding mask with shape (batch_size, seq_length)
def create_padding_mask(sequence, pad_token=PAD_TOKEN):
    # The mask will have True where there's padding, and False for valid tokens
    return sequence == pad_token


def main():
    # Example usage
    num_cards = 53  # 52 unique cards + 1 PAD token
    d_model = 64  # Embedding dimension
    nhead = 4  # Number of attention heads in transformer
    num_encoder_layers = 3  # Number of transformer layers

    # Create the model
    model = PokerHandPredictor(num_cards=num_cards, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers)

    # Example input (batch of hands with variable community cards, padded to 7 cards total)
    # Cards: [my_card_1, my_card_2, community_card_1, community_card_2, community_card_3, ..., PAD_TOKEN]
    example_input = torch.tensor([
        [0, 12, 25, 38, 10, PAD_TOKEN, PAD_TOKEN],  # Hand with 3 community cards (padded)
        [1, 13, 26, 39, 11, 12, PAD_TOKEN],  # Hand with 4 community cards (padded)
        [5, 7, 30, 45, 3, 19, 27]  # Hand with 5 community cards (no padding)
    ])

    # Generate padding mask (batch_size, seq_len)
    padding_mask = create_padding_mask(example_input).T

    # Forward pass
    output = model(example_input, src_key_padding_mask=padding_mask)
    print(output)  # Output: probabilities for win, draw, loss for each hand in the batch


if __name__ == "__main__":
    main()
