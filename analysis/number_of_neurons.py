from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def unify_equivariant_files():
    parent = Path('/Users/guy/Desktop/gcnn/number-of-neurons')
    files = [pd.read_csv(p) for p in parent.glob("*equivariant*")]

    return pd.concat(files).drop(columns=["Unnamed: 0"])


def unify_invariant_files():
    parent = Path('/Users/guy/Desktop/gcnn/number-of-neurons')
    files = [pd.read_csv(p) for p in parent.glob("*invariant*")]

    return pd.concat(files).drop(columns=["Unnamed: 0"])


def unify_files():
    equivariant_file = unify_equivariant_files()
    invariant_file = unify_invariant_files()

    unified_file = pd.concat({"invariant": invariant_file, "equivariant": equivariant_file})
    unified_file.to_csv("unified.csv", index_label=["model", "row"])


if __name__ == '__main__':
    unify_files()
    df = pd.read_csv('unified.csv')
    fig = sns.lineplot(data=df, x='number_of_neurons', y='test_accuracy', hue='model')
    plt.xlabel("size of FC (last layer)")
    fig.get_figure().savefig('accuracy-vs-size-of-fc-layer.png')
