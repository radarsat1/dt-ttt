import argparse
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from collections import defaultdict, deque

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

    # New flags to control behavior
    use_advantage: bool = True
    online_data_generation: bool = True

    # Parameters for advantage calculation
    num_offline_rollouts: int = 20000  # For offline V/Q table calculation
    replay_buffer_size: int = 20000    # For online advantage calculation buffer

    # Dataset sizes
    num_train_rollouts: int = 5000     # For offline RTG/Advantage training sets
    games_per_epoch: int = 500         # For online RTG/Advantage training sets
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

# --- NEW: Replay Buffer Class ---
class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def add(self, rollout: List):
        self.buffer.append(rollout)

    def sample(self, num_samples: int) -> List:
        return random.sample(list(self.buffer), min(len(self.buffer), num_samples))

    def get_all(self) -> List:
        return list(self.buffer)

    def __len__(self):
        return len(self.buffer)

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
    def get_move(self, game: TicTacToe, *args, **kwargs) -> int:
        return random.choice(game.get_available_moves())

class ModelAgent:
    """Base class for an agent that uses a PyTorch model to select moves."""
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def get_move(self, game: TicTacToe, state_hist, action_hist, return_hist) -> int:
        raise NotImplementedError

class CanonicalModelAgent(ModelAgent):
    """A model agent that uses a canonical board representation."""
    def get_move(self, game: TicTacToe, state_hist, action_hist, return_hist) -> int:
        # Convert the board state to the perspective of the current player
        canonical_state = game.get_state() * game.current_player

        # The history also needs to be converted to the canonical perspective for the model
        canonical_state_hist = []
        turn_player = 1
        for state in state_hist:
            canonical_state_hist.append(state * turn_player)
            turn_player *= -1

        states_seq = canonical_state_hist + [canonical_state]
        actions_seq = action_hist

        # Pad the actions sequence to match the states sequence length.
        # The value of the padding (e.g., 0) doesn't matter because of the causal mask.
        if len(actions_seq) < len(states_seq):
            actions_seq = actions_seq + [0] * (len(states_seq) - len(actions_seq))

        # Truncate all sequences to the model's max context length from the right
        max_len = self.model.config.max_seq_len
        states_seq, actions_seq, ret_hist = states_seq[-max_len:], actions_seq[-max_len:], return_hist[-max_len:]
        states = torch.tensor(states_seq, dtype=torch.long).unsqueeze(0).to(self.device)
        actions = torch.tensor(actions_seq, dtype=torch.long).unsqueeze(0).to(self.device)
        returns = torch.tensor(ret_hist, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        with torch.no_grad(): action_preds = self.model(states, actions, returns)
        logits = action_preds[0, -1]

        available_moves = game.get_available_moves()
        mask = torch.ones_like(logits) * -float('inf')
        mask[available_moves] = 0
        logits += mask

        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().item()

# 4. Rollout Function
def run_rollout(agent1, agent2) -> List[Tuple[np.ndarray, int, int]]:
    """
    Simulates a single game between two agents and records the trajectory
    from a canonical perspective (current player is always '1').
    """
    game = TicTacToe()
    trajectory = []
    is_done = False

    while not is_done:
        current_agent = agent1 if game.current_player == 1 else agent2

        # The state is stored from the perspective of the current player.
        canonical_state = game.get_state() * game.current_player
        action = current_agent.get_move(game)

        _, _, is_done = game.make_move(action)
        trajectory.append((canonical_state, action, 0))
    winner = game.check_winner()
    return [(s, a, (1 if (i%2==0 and winner==1) or (i%2!=0 and winner==-1) else -1) if winner!=0 else 0) for i, (s,a,_) in enumerate(trajectory)]

# 5. Advantage Calculation and DataLoader
def get_state_key(state: np.ndarray) -> bytes:
    """Converts a numpy array state to a hashable key."""
    return state.tobytes()

def calculate_advantage_tables(rollouts: List, config: Config) -> Tuple[Dict, Dict]:
    """Calculates V(s) and Q(s,a) tables via Monte Carlo estimation."""
    state_returns = defaultdict(lambda: {'sum': 0.0, 'count': 0})
    state_action_returns = defaultdict(lambda: {'sum': 0.0, 'count': 0})

    for rollout in rollouts:
        rewards = [r for s, a, r in rollout]
        returns = np.cumsum(rewards[::-1])[::-1]
        for i, (state, action, reward) in enumerate(rollout):
            state_key = get_state_key(state)
            state_action_key = (state_key, action)

            state_returns[state_key]['sum'] += returns[i]
            state_returns[state_key]['count'] += 1
            state_action_returns[state_action_key]['sum'] += returns[i]
            state_action_returns[state_action_key]['count'] += 1

    v_table = {k: v['sum'] / v['count'] for k, v in state_returns.items()}
    q_table = {k: v['sum'] / v['count'] for k, v in state_action_returns.items()}

    print(f"Calculated V-table for {len(v_table)} states and Q-table for {len(q_table)} state-action pairs.")
    return v_table, q_table

def _process_single_rollout_rtg(rollout: List, config: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Processes a single game rollout into padded tensors for RTG."""
    states = np.array([s for s, a, r in rollout])
    actions = np.array([a for s, a, r in rollout])
    rewards = np.array([r for s, a, r in rollout])
    rtgs = np.cumsum(rewards[::-1])[::-1].copy()

    padded_states = np.zeros((config.max_seq_len, config.board_size))
    padded_states[:len(states)] = states
    padded_actions = np.zeros(config.max_seq_len, dtype=int)
    padded_actions[:len(actions)] = actions
    padded_rtgs = np.zeros(config.max_seq_len)
    padded_rtgs[:len(rtgs)] = rtgs

    return (torch.tensor(padded_states, dtype=torch.long),
            torch.tensor(padded_actions, dtype=torch.long),
            torch.tensor(padded_rtgs, dtype=torch.float32).unsqueeze(-1))

class OnlineRTGRolloutDataset(Dataset):
    """Dataset that generates game rollouts on the fly for RTG training."""
    def __init__(self, num_games: int, config: Config):
        self.num_games = num_games
        self.config = config
        self.agent1 = RandomAgent()
        self.agent2 = RandomAgent()

    def __len__(self):
        return self.num_games

    def __getitem__(self, idx):
        rollout = run_rollout(self.agent1, self.agent2)
        return _process_single_rollout_rtg(rollout, self.config)

class RTGRolloutDataset(Dataset):
    """Dataset for storing and processing pre-generated game rollouts with RTGs."""
    def __init__(self, rollouts: List, config: Config):
        self.config = config
        self.rollouts = rollouts

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        rollout = self.rollouts[idx]
        return _process_single_rollout_rtg(rollout, self.config)


class AdvantageRolloutDataset(Dataset):
    """Dataset that processes rollouts to use advantages instead of RTGs."""
    def __init__(self, rollouts: List, v_table: Dict, q_table: Dict, config: Config):
        self.config = config
        self.rollouts = rollouts
        self.v_table = v_table
        self.q_table = q_table

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        rollout = self.rollouts[idx]
        states = [s for s, a, r in rollout]
        actions = [a for s, a, r in rollout]
        advantages = []
        for state, action in zip(states, actions):
            state_key = get_state_key(state)
            state_action_key = (state_key, action)
            q_value = self.q_table.get(state_action_key, 0.0)
            v_value = self.v_table.get(state_key, 0.0)
            advantages.append(q_value - v_value)

        padded_states = np.zeros((self.config.max_seq_len, self.config.board_size))
        padded_states[:len(states)] = np.array(states)
        padded_actions = np.zeros(self.config.max_seq_len, dtype=int)
        padded_actions[:len(actions)] = actions
        padded_advantages = np.zeros(self.config.max_seq_len)
        padded_advantages[:len(advantages)] = advantages

        return (torch.tensor(padded_states, dtype=torch.long),
                torch.tensor(padded_actions, dtype=torch.long),
                torch.tensor(padded_advantages, dtype=torch.float32).unsqueeze(-1))



# 6. Decision Transformer
class DecisionTransformer(nn.Module):
    """A basic Decision Transformer model for Tic-Tac-Toe."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Embeddings
        # State: 0 (empty), 1 (player 1), -1 (player 2) -> map to 0, 1, 2
        self.state_encoder = nn.Embedding(3, config.embedding_dim)
        self.return_encoder = nn.Linear(1, config.embedding_dim)
        self.action_encoder = nn.Embedding(config.board_size, config.embedding_dim); self.pos_encoder = nn.Embedding(config.max_seq_len, config.embedding_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.embedding_dim, nhead=config.nhead, dim_feedforward=config.dim_feedforward, dropout=config.dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)

        # Output head
        self.action_head = nn.Linear(config.embedding_dim, config.board_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, states, actions, returns):
        batch_size, seq_len = states.shape[0], states.shape[1]
        state_embeds = self.state_encoder(states + 1).sum(dim=2)
        return_embeds = self.return_encoder(returns)
        action_embeds = self.action_encoder(actions)

        positions = torch.arange(0, seq_len, device=states.device).unsqueeze(0)
        pos_embeds = self.pos_encoder(positions)

        # Create the sequence for the transformer
        # Order: Return, State, Action
        # We interleave the embeddings
        input_embeds = torch.stack((return_embeds, state_embeds, action_embeds), dim=2)
        input_embeds = input_embeds.reshape(batch_size, 3 * seq_len, self.config.embedding_dim)

        # Add positional embeddings. Each (return, s, a) triplet gets the same pos embedding
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
    agent1, agent2 = RandomAgent(), RandomAgent()
    model = DecisionTransformer(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    best_win_rate, epochs_no_improve, patience = -1.0, 0, 10

    # --- Data setup based on config ---
    train_loader = None
    v_table, q_table = None, None # For online advantage calculation
    replay_buffer = None

    if config.use_advantage:
        if config.online_data_generation:
            # Online Advantage
            replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)
            print("Pre-filling replay buffer for online advantage calculation...")
            for _ in range(config.batch_size):
                if len(replay_buffer) < config.replay_buffer_size:
                    replay_buffer.add(run_rollout(agent1, agent2))
            # DataLoader will be created inside the epoch loop
        else:
            # Offline Advantage
            print(f"Generating {config.num_offline_rollouts} rollouts for offline advantage calculation...")
            offline_rollouts = [run_rollout(agent1, agent2) for _ in range(config.num_offline_rollouts)]
            v_table, q_table = calculate_advantage_tables(offline_rollouts, config)

            print(f"Generating {config.num_train_rollouts} rollouts for training set...")
            train_rollouts = [run_rollout(agent1, agent2) for _ in range(config.num_train_rollouts)]
            train_dataset = AdvantageRolloutDataset(train_rollouts, v_table, q_table, config)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    else: # RTG
        if config.online_data_generation:
            # Online RTG
            print(f"Using online data generation for RTG. Generating {config.games_per_epoch} games per epoch.")
            train_dataset = OnlineRTGRolloutDataset(config.games_per_epoch, config)
            # Shuffling is not needed as every sample is new.
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
        else:
            # Offline RTG
            print(f"Using RTG. Generating {config.num_train_rollouts} rollouts for training set...")
            train_rollouts = [run_rollout(agent1, agent2) for _ in range(config.num_train_rollouts)]
            train_dataset = RTGRolloutDataset(train_rollouts, config)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)


    for epoch in range(config.num_epochs):
        model.train()

        # For online advantage, data is regenerated each epoch
        if config.use_advantage and config.online_data_generation:
            for _ in range(config.games_per_epoch):
                replay_buffer.add(run_rollout(agent1, agent2))

            print(f"Epoch {epoch+1}/{config.num_epochs}: Re-calculating advantages from buffer ({len(replay_buffer)} games)...")
            v_table, q_table = calculate_advantage_tables(replay_buffer.get_all(), config)

            training_rollouts = replay_buffer.sample(config.games_per_epoch)
            train_dataset = AdvantageRolloutDataset(training_rollouts, v_table, q_table, config)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        total_loss = 0
        for states, actions, returns in train_loader:
            states, actions, returns = states.to(device), actions.to(device), returns.to(device)

            # We predict each action given the history up to that point
            # So, the input actions should be shifted
            # a_hat_t = model(s_0, a_0, rtg_0, ..., s_t, a_t, rtg_t)
            # The model predicts the action for the *next* state based on the current one
            action_preds = model(states, actions, returns)

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
    # Use the new agent that understands canonical states
    model_agent = CanonicalModelAgent(model, device)
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
        # Start with a target return of 1.0 (a win), which could be RTG or Advantage
        target_return = 1.0
        return_hist = [target_return]

        agents = {1: model_agent if model_plays_as == 1 else random_agent, -1: model_agent if model_plays_as == -1 else random_agent}
        done = False
        while not done:
            state_before_move = game.get_state()

            current_agent = agents[game.current_player]
            move = current_agent.get_move(game, state_hist, action_hist, return_hist)

            # Record the state and action that led to the new state
            state_hist.append(state_before_move)
            action_hist.append(move)

            _, reward, done = game.make_move(move)

            # Update the return-to-go for the next step. In Tic-Tac-Toe, rewards are zero
            # until the end, so the RTG/Advantage for the next step is the same as the
            # current. In games with intermediate rewards, this would be:
            # return_hist[-1] - reward
            return_hist.append(return_hist[-1])

        winner = game.check_winner()
        if winner == 0:
            draws +=1
        elif winner == model_plays_as:
            if model_plays_as == 1: wins_p1 += 1
            else: wins_p2 += 1
        else: # The winner was the other player
            losses += 1

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
    for field_name, field_info in Config.__dataclass_fields__.items():
        if field_name == 'args': continue
        if field_info.type == bool:
            parser.add_argument(f"--{field_name}", action=argparse.BooleanOptionalAction, default=field_info.default)
        else:
            parser.add_argument(f"--{field_name}", type=field_info.type, default=field_info.default)

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
