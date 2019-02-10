import pickle
import subprocess
import time

from sklearn import tree


def load_data_set(name):
    with open(f'datasets/{name}/{name}.dataset', 'rb') as f:
        d = pickle.load(f)

    return d


def export_decision_tree(clf, filename):
    tree.export_graphviz(clf, out_file='out/decision_tree_pruning/{}.dot'.format(filename))
    subprocess.run([
        "dot", "-Tpng",
        "out/decision_tree_pruning/{}.dot".format(filename),
        "-o", "out/decision_tree_pruning/{}-tree.png".format(filename)
    ])


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
