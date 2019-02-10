import pickle
import subprocess

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
        "-o", "out/decision_tree_pruning/{}.png".format(filename)
    ])
