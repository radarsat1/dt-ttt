import argparse
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# 1. Configuration Dataclass
@dataclass
class Config:
    """Configuration for the Decision Transformer Tic-Tac-Toe experiment."""
    # Experiment settings
    exp_name: str = "dt_tictactoe"
    seed: int = 42
    num_rollouts: int = 5000
    val_rollouts: int = 500

    # Model parameters
    embedding_dim: int = 128
    nhead: int = 4
    num_decoder_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1

    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 128
    num_epochs: int = 50

    # Game settings
    board_size: int = 9
    max_seq_len: int = 9 # Maximum number of moves in a game

    # Filled by argparse
    args: dict = field(default_factory=dict)

# 2. Tic-Tac-Toe Game Board
class TicTacToe:
    """A simple Tic-Tac-Toe game environment."""
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1

    def get_state(self):
        """Returns the current board state."""
        return self.board.copy()

    def get_available_moves(self) -> List[int]:
        """Returns a list of available moves (indices of empty cells)."""
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def make_move(self, move: int) -> Tuple[np.ndarray, int, bool]:
        """
        Makes a move on the board.
        Returns: (new_state, reward, is_done)
        """
        if self.board[move] != 0:
            raise ValueError("Invalid move")

        self.board[move] = self.current_player
        winner = self.check_winner()

        if winner is not None:
            reward = winner if self.current_player == 1 else -winner
            return self.get_state(), reward, True

        self.current_player *= -1 # Switch player
        return self.get_state(), 0, False

    def check_winner(self) -> int | None:
        """Checks for a winner, a draw, or if the game is ongoing."""
        board_2d = self.board.reshape(3, 3)
        # Check rows, columns, and diagonals
        lines = np.concatenate((board_2d, board_2d.T,
                                [np.diag(board_2d)], [np.diag(np.fliplr(board_2d))]))

        for line in lines:
            if np.all(line == 1): return 1
            if np.all(line == -1): return -1

        if not self.get_available_moves():
            return 0 # Draw

        return None # Game is ongoing

# 3. Agents
class RandomAgent:
    """An agent that chooses moves randomly."""
    def get_move(self, game: TicTacToe, state_hist=None, action_hist=None, rtg_hist=None) -> int:
        return random.choice(game.get_available_moves())

class ModelAgent:
    """An agent that uses a PyTorch model to select moves."""
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def get_move(self, game: TicTacToe, state_hist, action_hist, rtg_hist) -> int:
        # Prepare inputs for the Decision Transformer model
        # The states and rtgs include the current timestep, actions do not yet.
        states_seq = state_hist + [game.get_state()]
        actions_seq = action_hist
        rtgs_seq = rtg_hist

        # Pad the actions sequence to match the states sequence length.
        # The value of the padding (e.g., 0) doesn't matter because of the causal mask.
        if len(actions_seq) < len(states_seq):
            actions_seq = actions_seq + [0] * (len(states_seq) - len(actions_seq))

        # Truncate all sequences to the model's max context length from the right
        max_len = self.model.config.max_seq_len
        states_seq = states_seq[-max_len:]
        actions_seq = actions_seq[-max_len:]
        rtgs_seq = rtgs_seq[-max_len:]

        states = torch.tensor(states_seq, dtype=torch.long).unsqueeze(0).to(self.device)
        actions = torch.tensor(actions_seq, dtype=torch.long).unsqueeze(0).to(self.device)
        rtgs = torch.tensor(rtgs_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            action_preds = self.model(states, actions, rtgs)

        # Get the logits for the last predicted action in the sequence
        logits = action_preds[0, -1]

        # Mask unavailable moves
        available_moves = game.get_available_moves()
        mask = torch.ones_like(logits) * -float('inf')
        mask[available_moves] = 0
        logits += mask

        # Sample from the resulting distribution
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().item()

# 4. Rollout Function
def run_rollout(agent1, agent2) -> List[Tuple[np.ndarray, int, int]]:
    """
    Simulates a single game between two agents and records the trajectory.
    Returns a list of (state, action, reward) tuples.
    """
    game = TicTacToe()
    trajectory = []
    is_done = False

    while not is_done:
        current_agent = agent1 if game.current_player == 1 else agent2

        state_before_move = game.get_state()
        action = current_agent.get_move(game)

        _, reward, is_done = game.make_move(action)

        trajectory.append((state_before_move, action, reward))

    # Distribute the final reward back through the trajectory
    final_reward = trajectory[-1][-1]
    full_rollout = []
    for i, (state, action, _) in enumerate(trajectory):
        # Player 1's reward is the final outcome
        # Player 2's reward is the inverted outcome
        player_at_turn = 1 if i % 2 == 0 else -1
        reward = final_reward * player_at_turn
        full_rollout.append((state, action, reward))

    return full_rollout

# 5. PyTorch DataLoader
class RolloutDataset(Dataset):
    """Dataset for storing and processing game rollouts."""
    def __init__(self, rollouts: List, config: Config):
        self.config = config
        self.states, self.actions, self.rtgs = self._process_rollouts(rollouts)

    def _process_rollouts(self, rollouts):
        all_states, all_actions, all_rtgs = [], [], []
        for rollout in rollouts:
            states = np.array([s for s, a, r in rollout])
            actions = np.array([a for s, a, r in rollout])
            rewards = np.array([r for s, a, r in rollout])

            # Calculate returns-to-go
            rtgs = np.cumsum(rewards[::-1])[::-1]

            # Pad sequences to max_seq_len
            padded_states = np.zeros((self.config.max_seq_len, self.config.board_size))
            padded_states[:len(states)] = states

            padded_actions = np.zeros(self.config.max_seq_len, dtype=int)
            padded_actions[:len(actions)] = actions

            padded_rtgs = np.zeros(self.config.max_seq_len)
            padded_rtgs[:len(rtgs)] = rtgs

            all_states.append(padded_states)
            all_actions.append(padded_actions)
            all_rtgs.append(padded_rtgs)

        return (torch.tensor(np.array(all_states), dtype=torch.long),
                torch.tensor(np.array(all_actions), dtype=torch.long),
                torch.tensor(np.array(all_rtgs), dtype=torch.float32).unsqueeze(-1))

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rtgs[idx]

def create_dataloader(config: Config, is_validation=False) -> DataLoader:
    """Generates rollouts and creates a DataLoader."""
    num_games = config.val_rollouts if is_validation else config.num_rollouts
    agent1 = RandomAgent()
    agent2 = RandomAgent()

    rollouts = [run_rollout(agent1, agent2) for _ in range(num_games)]
    dataset = RolloutDataset(rollouts, config)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=not is_validation)


# 6. Decision Transformer Model
class DecisionTransformer(nn.Module):
    """A basic Decision Transformer model for Tic-Tac-Toe."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Embeddings
        # State: 0 (empty), 1 (player 1), -1 (player 2) -> map to 0, 1, 2
        self.state_encoder = nn.Embedding(3, config.embedding_dim)
        self.rtg_encoder = nn.Linear(1, config.embedding_dim)
        self.action_encoder = nn.Embedding(config.board_size, config.embedding_dim)
        self.pos_encoder = nn.Embedding(config.max_seq_len, config.embedding_dim)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embedding_dim,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)

        # Output head
        self.action_head = nn.Linear(config.embedding_dim, config.board_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, states, actions, rtgs):
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Remap states from {-1, 0, 1} to {0, 1, 2} for embedding
        states_mapped = states + 1
        state_embeds = self.state_encoder(states_mapped).sum(dim=2) # Sum embeddings for each cell
        rtg_embeds = self.rtg_encoder(rtgs)
        action_embeds = self.action_encoder(actions)

        positions = torch.arange(0, seq_len, device=states.device).unsqueeze(0)
        pos_embeds = self.pos_encoder(positions)

        # Create the sequence for the transformer
        # Order: RTG, State, Action
        # We interleave the embeddings
        input_embeds = torch.stack((rtg_embeds, state_embeds, action_embeds), dim=2)
        input_embeds = input_embeds.reshape(batch_size, 3 * seq_len, self.config.embedding_dim)

        # Add positional embeddings. Each (rtg, s, a) triplet gets the same pos embedding
        pos_embeds_tripled = pos_embeds.repeat_interleave(3, dim=1)
        input_embeds += pos_embeds_tripled

        # Causal mask to prevent attending to future tokens
        causal_mask = nn.Transformer.generate_square_subsequent_mask(3 * seq_len).to(states.device)

        # The transformer decoder expects target and memory.
        # For a decoder-only setup like this, we pass the same sequence as both.
        transformer_output = self.transformer_decoder(tgt=input_embeds, memory=input_embeds, tgt_mask=causal_mask)

        # We only care about the outputs that correspond to states to predict the next action
        action_logits = self.action_head(transformer_output[:, 1::3, :])

        return action_logits


# 7. Training Function
def train(config: Config):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f"runs/{config.exp_name}")

    # Setup
    set_seed(config.seed)
    train_loader = create_dataloader(config)

    model = DecisionTransformer(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for states, actions, rtgs in train_loader:
            states, actions, rtgs = states.to(device), actions.to(device), rtgs.to(device)

            # We predict each action given the history up to that point
            # So, the input actions should be shifted
            # a_hat_t = model(s_0, a_0, rtg_0, ..., s_t, a_t, rtg_t)
            # The model predicts the action for the *next* state based on the current one
            action_preds = model(states, actions, rtgs)

            # The target is the action sequence
            # We flatten batch and sequence dimensions for the loss function
            action_preds_flat = action_preds.view(-1, config.board_size)
            actions_flat = actions.view(-1)

            loss = loss_fn(action_preds_flat, actions_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_loss:.4f}")

        # Validation
        if (epoch + 1) % 5 == 0:
            validate(model, config, device, writer, epoch)

    writer.close()
    torch.save(model.state_dict(), f"{config.exp_name}_model.pth")
    print("Training complete.")


# 8. Validation Function
def validate(model: nn.Module, config: Config, device: torch.device, writer: SummaryWriter, epoch: int):
    """Evaluates the model's performance against a random agent."""
    model.eval()
    model_agent = ModelAgent(model, device)
    random_agent = RandomAgent()

    wins_p1, wins_p2, losses, draws = 0, 0, 0, 0
    model_as_p1_count = 0
    model_as_p2_count = 0

    for _ in range(config.val_rollouts):
        game = TicTacToe()
        model_plays_as = random.choice([1, -1]) # Randomly assign model as player 1 or player 2
        if model_plays_as == 1:
            model_as_p1_count += 1
        else:
            model_as_p2_count += 1

        state_hist, action_hist = [], []
        # We start with a target return of 1.0 (a win)
        target_return = 1.0
        rtg_hist = [target_return]

        # Assign agents based on `model_plays_as`
        agents = {1: None, -1: None}
        if model_plays_as == 1:
            agents[1] = model_agent
            agents[-1] = random_agent
        else:
            agents[1] = random_agent
            agents[-1] = model_agent

        done = False
        while not done:
            state_before_move = game.get_state()

            current_agent = agents[game.current_player]
            move = current_agent.get_move(game, state_hist, action_hist, rtg_hist)

            # Record the state and action that led to the new state
            state_hist.append(state_before_move)
            action_hist.append(move)

            _, reward, done = game.make_move(move)

            # Update the return-to-go for the next step. In Tic-Tac-Toe, rewards are
            # zero until the end, so the RTG for the next step is the same as the current.
            # In games with intermediate rewards, this would be: rtg_hist[-1] - reward
            rtg_hist.append(rtg_hist[-1])

        winner = game.check_winner()
        if winner is None:
            # This case should ideally not be reached if `done` is True and no winner is found (implies draw)
            # but explicitly handling it for robustness.
            draws += 1
        elif winner == model_plays_as:
            if model_plays_as == 1:
                wins_p1 += 1
            else:
                wins_p2 += 1
        elif winner == -model_plays_as: losses += 1
        else: draws += 1

    win_rate_p1 = wins_p1 / max(1, model_as_p1_count)
    win_rate_p2 = wins_p2 / max(1, model_as_p2_count)
    overall_wins = wins_p1 + wins_p2
    overall_games = model_as_p1_count + model_as_p2_count
    overall_win_rate = overall_wins / max(1, overall_games)

    writer.add_scalar("WinRate/validation_P1", win_rate_p1, epoch)
    writer.add_scalar("WinRate/validation_P2", win_rate_p2, epoch)
    writer.add_scalar("WinRate/validation_Overall", overall_win_rate, epoch)
    print(f"Validation Win Rate (P1): {win_rate_p1:.2f} (Wins: {wins_p1}, Games: {model_as_p1_count})")
    print(f"Validation Win Rate (P2): {win_rate_p2:.2f} (Wins: {wins_p2}, Games: {model_as_p2_count})")
    print(f"Validation Overall Win Rate: {overall_win_rate:.2f} (Wins: {overall_wins}, Losses: {losses}, Draws: {draws})")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    config = Config()
    parser = argparse.ArgumentParser(description="Train a Decision Transformer for Tic-Tac-Toe.")

    # Add arguments for each field in the Config dataclass
    for field_name, field_type in Config.__annotations__.items():
        if field_name == 'args': continue
        parser.add_argument(f"--{field_name}", type=field_type, default=getattr(config, field_name))

    args = parser.parse_args()

    # Update config with parsed arguments
    for arg_name, arg_value in vars(args).items():
        if hasattr(config, arg_name):
            setattr(config, arg_name, arg_value)

    config.args = vars(args)
    print("Configuration:", config)

    train(config)

if __name__ == "__main__":
    main()
