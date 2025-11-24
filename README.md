# Decision Transformer for Tic-Tac-Toe

This project explores the application of Decision Transformers (DT) to the game of
Tic-Tac-Toe. It investigates different training methodologies, including Return-to-Go
(RTG) and Advantage-based conditioning, with both offline and online data generation
strategies.

This code accompanies [a blog
post](https://sinclairs.gitlab.io/blog/revsiting-the-decision-transformer-with-tic-tac-toe)
on this topic.

## Project Structure

- `train.py`: Contains the core logic for the Tic-Tac-Toe game, Decision Transformer
  model, agents, data loading, and training loop.
- `run_experiments.sh`: A bash script to run predefined training experiments with
  different configurations.
- `plots.py`: Generate plots of loss and metrics accompanying the blog post from the
  tensorboard traces of `run_experiments.sh`.

## Setup

This project uses `uv` for dependency management and running the Python scripts.

1.  **Install `uv`**: If you don't have `uv` installed, see [installation
    instructions](https://docs.astral.sh/uv/getting-started/installation/).

2.  **Running the Application**: Simply run your Python script with `uv`:
    ```bash
    uv run train.py --help
    ```

    This command will automatically create a virtual environment and install the dependencies listed in `pyproject.toml`.

### If You Don’t Use `uv`

If you prefer not to use **`uv`**, you can create a virtual environment manually and
install the dependencies with:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

You can run the pre-configured experiments using the `run_experiments.sh` script:

```bash
bash run_experiments.sh
```

This script will execute four different training configurations:

-   `offline_rtg`: Offline training with Return-to-Go.
-   `online_rtg`: Online training with Return-to-Go.
-   `offline_advantage`: Offline training with Advantage.
-   `online_advantage`: Online training with Advantage.

Each experiment will save its results and logs in the `runs/` directory. You can monitor
the training progress using Tensorboard.

## Monitoring with Tensorboard

After running experiments, you can visualize the training loss and validation win rates
using Tensorboard:

```bash
tensorboard --logdir runs
```

Then, open your web browser and navigate to the address provided by Tensorboard (usually
`http://localhost:6006/`).

## Generating plots

You can run `uv run plots.py` to generate the plots in the blog post.  This plot loads
data from the tensorboard logs produced by running `run_experiments.py` and uses
matplotlib to plot the loss and win rate traces.

©2025 Stephen Sinclair <radarsat1@gmail.com>, see LICENCE (MIT). Code was generated using
Gemini 2.5 Pro, with oversight and editing by the author.
