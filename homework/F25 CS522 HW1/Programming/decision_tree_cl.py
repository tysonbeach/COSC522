"""
Decision tree learning for binary classification (COSC 522 HW1).

Usage:
    python decision_tree.py <train_input> <valid_input> <max_depth>

Prints the learned tree (pre-order DFS) followed by the training and
validation error.

This module is also written to be import-friendly, e.g.:

    import decision_tree as dt
    attr_names, train_data, train_labels = dt.load_tsv("heart_train.tsv")
    root = dt.train_node(train_data, train_labels, attr_names,
                          depth=0, max_depth=4, threshold=0.0, min_size=0)
    print(dt.print_tree(root))

so that the empirical-question experiments (depth sweep, min-split-size
sweep, info-gain-threshold sweep, reduced error pruning) can all reuse the
same core implementation instead of re-deriving it.
"""

import sys
import math
from collections import Counter


class Node:
    """
    A node in the decision tree.

    - Interior node: `attr` / `attr_name` identify the (binary) attribute
      this node splits on; `left` is the subtree for attribute value 0 and
      `right` is the subtree for attribute value 1.
    - Leaf node: `attr` is None and `vote` holds the predicted label.

    `counts` always holds (# label-0, # label-1) for the training examples
    that reached this node -- this is what gets printed in brackets, e.g.
    "[14 0/14 1]", regardless of whether the node ends up being a leaf.
    """

    def __init__(self, depth=0):
        self.left = None
        self.right = None
        self.attr = None
        self.attr_name = None
        self.vote = None
        self.depth = depth
        self.counts = (0, 0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tsv(path):
    """Load a tsv file where the first row is the header and the last
    column is the (binary) class label. Returns (attr_names, data, labels)
    where `data` is a list of rows (each row a list of 0/1 ints, label
    column excluded) and `labels` is a list of 0/1 ints."""
    with open(path) as f:
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip() != ""]

    header = rows[0]
    attr_names = header[:-1]

    data = []
    labels = []
    for row in rows[1:]:
        data.append([int(v) for v in row[:-1]])
        labels.append(int(row[-1]))

    return attr_names, data, labels


# ---------------------------------------------------------------------------
# Entropy / mutual information
# ---------------------------------------------------------------------------

def entropy(labels):
    """Binary (base-2) entropy of a list of 0/1 labels."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * math.log2(p)
    return h


def mutual_information(data, labels, attr_index):
    """I(Y; X) = H(Y) - P(X=0)H(Y|X=0) - P(X=1)H(Y|X=1) for the binary
    attribute at `attr_index`."""
    n = len(labels)
    if n == 0:
        return 0.0

    h_y = entropy(labels)

    labels_0 = [labels[i] for i in range(n) if data[i][attr_index] == 0]
    labels_1 = [labels[i] for i in range(n) if data[i][attr_index] == 1]

    p0 = len(labels_0) / n
    p1 = len(labels_1) / n

    h_cond = p0 * entropy(labels_0) + p1 * entropy(labels_1)
    return h_y - h_cond


def majority_vote(labels):
    """Majority label; ties broken in favor of the numerically larger
    label (1 before 0)."""
    count0 = labels.count(0)
    count1 = labels.count(1)
    return 1 if count1 >= count0 else 0


def best_attribute(data, labels):
    """Return (best_index, best_gain) over all attribute columns, breaking
    ties in favor of the first (lowest-index) column."""
    if not data:
        return None, 0.0

    n_attrs = len(data[0])
    best_idx = None
    best_gain = float("-inf")

    for idx in range(n_attrs):
        gain = mutual_information(data, labels, idx)
        if gain > best_gain:
            best_gain = gain
            best_idx = idx

    return best_idx, best_gain


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def train_node(data, labels, attr_names, depth, max_depth,
               threshold=0.0, min_size=0):
    """Recursively train a decision (sub)tree.

    - max_depth: None means unlimited depth.
    - threshold: only split if the best mutual information is > threshold.
    - min_size: only split if the node has >= min_size examples
      (i.e. splitting is prohibited when len(data) < min_size).
    """
    node = Node(depth=depth)
    node.counts = (labels.count(0), labels.count(1))
    node.vote = majority_vote(labels)

    if len(data) == 0:
        return node

    # Base conditions: stop growing this branch.
    if max_depth is not None and depth >= max_depth:
        return node
    if len(set(labels)) == 1:
        return node
    if len(data) < min_size:
        return node

    best_idx, best_gain = best_attribute(data, labels)
    if best_idx is None or best_gain <= threshold:
        return node

    node.attr = best_idx
    node.attr_name = attr_names[best_idx]

    left_data, left_labels = [], []
    right_data, right_labels = [], []
    for row, lab in zip(data, labels):
        if row[best_idx] == 0:
            left_data.append(row)
            left_labels.append(lab)
        else:
            right_data.append(row)
            right_labels.append(lab)

    node.left = train_node(left_data, left_labels, attr_names,
                            depth + 1, max_depth, threshold, min_size)
    node.right = train_node(right_data, right_labels, attr_names,
                             depth + 1, max_depth, threshold, min_size)
    return node


# ---------------------------------------------------------------------------
# Prediction / error
# ---------------------------------------------------------------------------

def predict_one(node, row):
    while node.attr is not None:
        node = node.left if row[node.attr] == 0 else node.right
    return node.vote


def predict(node, data):
    return [predict_one(node, row) for row in data]


def error_rate(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return 0.0
    wrong = sum(1 for a, b in zip(y_true, y_pred) if a != b)
    return wrong / n


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def _format_counts(node):
    return "[{} 0/{} 1]".format(node.counts[0], node.counts[1])


def print_tree(node):
    """Pre-order DFS pretty-print matching the assignment's format."""
    lines = [_format_counts(node)]

    def helper(n, depth):
        if n.attr is None:
            return
        prefix = "| " * (depth + 1)
        lines.append("{}{} = 0: {}".format(prefix, n.attr_name, _format_counts(n.left)))
        helper(n.left, depth + 1)
        lines.append("{}{} = 1: {}".format(prefix, n.attr_name, _format_counts(n.right)))
        helper(n.right, depth + 1)

    helper(node, 0)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reduced error pruning
# ---------------------------------------------------------------------------

def reduced_error_prune(node, val_data, val_labels):
    """Bottom-up (post-order) reduced error pruning using a validation set.

    At each interior node we compare the validation error of the current
    subtree to the validation error we'd get by collapsing the node into a
    leaf that predicts its majority vote. Ties are broken in favor of the
    shorter tree, i.e. we prune whenever leaf_error <= subtree_error.

    Mutates `node` (and its descendants) in place and returns the number
    of validation examples misclassified by the (possibly now-pruned)
    subtree rooted at `node`.
    """
    if node.attr is None:
        return sum(1 for lab in val_labels if lab != node.vote)

    left_data, left_labels = [], []
    right_data, right_labels = [], []
    for row, lab in zip(val_data, val_labels):
        if row[node.attr] == 0:
            left_data.append(row)
            left_labels.append(lab)
        else:
            right_data.append(row)
            right_labels.append(lab)

    left_wrong = reduced_error_prune(node.left, left_data, left_labels)
    right_wrong = reduced_error_prune(node.right, right_data, right_labels)
    subtree_wrong = left_wrong + right_wrong

    leaf_wrong = sum(1 for lab in val_labels if lab != node.vote)

    if leaf_wrong <= subtree_wrong:
        node.attr = None
        node.attr_name = None
        node.left = None
        node.right = None
        return leaf_wrong

    return subtree_wrong


# ---------------------------------------------------------------------------
# Convenience helpers for the empirical questions (depth / min-size /
# threshold sweeps). Not required by the CLI, but handy to `import
# decision_tree as dt` and call from a notebook/script.
# ---------------------------------------------------------------------------

def run(train_path, valid_path, max_depth, threshold=0.0, min_size=0):
    """Train on train_path and evaluate on valid_path. Returns
    (root, attr_names, train_err, valid_err)."""
    attr_names, train_data, train_labels = load_tsv(train_path)
    _, valid_data, valid_labels = load_tsv(valid_path)

    root = train_node(train_data, train_labels, attr_names,
                       depth=0, max_depth=max_depth,
                       threshold=threshold, min_size=min_size)

    train_err = error_rate(train_labels, predict(root, train_data))
    valid_err = error_rate(valid_labels, predict(root, valid_data))
    return root, attr_names, train_err, valid_err


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 4:
        print("Usage: python decision_tree.py <train_input> <valid_input> <max_depth>")
        sys.exit(1)

    train_path = sys.argv[1]
    valid_path = sys.argv[2]
    max_depth = int(sys.argv[3])

    attr_names, train_data, train_labels = load_tsv(train_path)
    _, valid_data, valid_labels = load_tsv(valid_path)

    root = train_node(train_data, train_labels, attr_names,
                       depth=0, max_depth=max_depth,
                       threshold=0.0, min_size=0)

    print(print_tree(root))

    train_pred = predict(root, train_data)
    valid_pred = predict(root, valid_data)
    train_err = error_rate(train_labels, train_pred)
    valid_err = error_rate(valid_labels, valid_pred)

    print("error(train): {:.6f}".format(train_err))
    print("error(test): {:.6f}".format(valid_err))


if __name__ == "__main__":
    main()
