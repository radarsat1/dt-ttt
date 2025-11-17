from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import lfilter

# From: https://stackoverflow.com/questions/41074688/how-do-you-read-tensorboard-files-programmatically
def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        str(path),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}

# From: https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
def smooth(sig, weight):
    return lfilter([1. - weight], [1., -weight], sig, zi=[sig[0]])[0]

def plot(run, scalars):
    dfs = parse_tensorboard(*Path(f'runs/{run}').glob('events.out.tfevents.*'), scalars)
    for ax, scal in zip(axs, scalars):
        ax.set_title(scal)
        if scal.startswith('WinRate'):
            ax.plot(dfs[scal].step, dfs[scal].value * 100, label=run)
            ax.set_ylim(20, 90)
        else:
            ax.plot(dfs[scal].step, dfs[scal].value, label=run)
        ax.set_xlabel('epochs')
        ax.legend()

if __name__ == '__main__':
    scalars = ['Loss/train', 'WinRate/validation_Overall',
               'WinRate/validation_P1', 'WinRate/validation_P2']
    sns.set(style="darkgrid", palette="muted", color_codes=True)
    fig, axs = plt.subplots(1, len(scalars), figsize=(20,4))
    runs = ['offline_rtg', 'online_rtg', 'offline_advantage', 'online_advantage']
    for run in runs:
        plot(run, scalars)
    plt.tight_layout()
    plt.subplots_adjust()
    plt.savefig('figure.svg')
