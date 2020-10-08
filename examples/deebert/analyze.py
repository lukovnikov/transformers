import numpy as np
import fire
from IPython import embed
from scipy.special import softmax
from transformers import glue_compute_metrics as compute_metrics


def entropy(x):
    x = softmax(x, -1)
    logx = np.log(x)
    entr = x * logx
    entr = -entr.sum(-1)
    return entr


def main(
        p="",
        task="",
    ):
    d = np.load(p)
    times = d["times"]
    logits = d["exit_logits"]
    preds = d["preds"]
    labels = d["labels"]
    task = d["task"] if task == "" else task
    entropies = entropy(logits)
    numex, numlayer = times.shape
    # time_argsorted = np.argsort(times.reshape(-1))
    # time_sorted = times.reshape(-1)[time_argsorted]
    min_time = times.min(-1).max()
    max_time = times.max()

    entr_argsorted = np.argsort(entropies.reshape(-1))[::-1]
    entr_sorted = entropies.reshape(-1)[entr_argsorted]

    min_entr = entropies.min(-1).max()
    max_entr = entropies.max()

    # for every entropy threshold, compute the metrics

    embed()



if __name__ == '__main__':
    fire.Fire(main)