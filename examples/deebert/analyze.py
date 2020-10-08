import numpy as np
import fire
from IPython import embed


def main(
        p="",
    ):
    d = np.load(p)
    numex, numlayer = d["times"].shape
    time_argsorted = np.argsort(d["times"].reshape(-1)).reshape(numex, numlayer)
    time_sorted = d["times"].reshape(-1)[time_argsorted].reshape(numex, numlayer)
    min_time = d["times"].min(-1).max()
    max_time = d["times"].max()

    embed()



if __name__ == '__main__':
    fire.Fire(main)