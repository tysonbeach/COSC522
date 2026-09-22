import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import sys

class Node:
    def __init__(self, labels):
        self.count_0 = int((labels == 0).sum())
        self.count_1 = int((labels == 1).sum())
        self.prediction = majority_vote(labels)
        self.feature = None
        self.left = None
        self.right = None

def entropy(labels):
    if labels.empty:
        return 0.0
    label_probabilities = labels.value_counts(normalize=True)
    total_entropy = 0
    for probability in label_probabilities:
        label_contribution = -probability * np.log2(probability)
        total_entropy += label_contribution
    return total_entropy

def mutual_information(data, feature):
    all_labels = data.iloc[:, -1]
    entropy_before_split = entropy(all_labels)

    feature_values = data[feature]
    rows_with_zero = data[feature_values == 0]
    rows_with_one = data[feature_values == 1]

    zero_group_labels = rows_with_zero.iloc[:, -1]
    one_group_labels = rows_with_one.iloc[:, -1]
    zero_group_entropy = entropy(zero_group_labels)
    one_group_entropy = entropy(one_group_labels)

    total_rows = len(data)
    zero_group_weight = len(rows_with_zero) / total_rows
    one_group_weight = len(rows_with_one) / total_rows
    entropy_after_split = (zero_group_weight * zero_group_entropy) + (one_group_weight * one_group_entropy)
    information_gain = entropy_before_split - entropy_after_split
    return information_gain

def best_feature(data, threshold=0.0):
    chosen_feature = None
    best_gain = threshold
    feature_names = data.columns[:-1]

    for feature_name in feature_names:
        information_gain = mutual_information(data, feature_name)
        if information_gain > best_gain:
            chosen_feature = feature_name
            best_gain = information_gain

    return chosen_feature

def majority_vote(labels):
    is_label_zero = labels == 0
    is_label_one = labels == 1
    count_0 = is_label_zero.sum()
    count_1 = is_label_one.sum()

    if count_1 >= count_0:
        return 1
    return 0

def build_tree(data, max_depth=None, threshold=0.0, depth=0, min_size=0):
    labels = data.iloc[:, -1]
    node = Node(labels)

    if max_depth is not None:
        if depth >= max_depth:
            return node

    if len(data) < min_size:
        return node

    if labels.nunique() == 1:
        return node

    split_feature = best_feature(data, threshold)
    if split_feature is None:
        return node

    node.feature = split_feature
    feature_values = data[split_feature]
    left_data = data[feature_values == 0]
    right_data = data[feature_values == 1]

    left_data = left_data.drop(columns=[split_feature])
    right_data = right_data.drop(columns=[split_feature])

    child_depth = depth + 1
    node.left = build_tree(left_data, max_depth, threshold, child_depth, min_size)
    node.right = build_tree(right_data, max_depth, threshold, child_depth, min_size)

    return node

def print_tree(node, depth=0, branch=None):
    counts = f"[{node.count_0} 0/{node.count_1} 1]"

    if branch is None:
        print(counts)
    else:
        print("| " * depth + f"{branch}: {counts}")

    if node.feature is None:
        return

    print_tree(node.left, depth + 1, f"{node.feature} = 0")
    print_tree(node.right, depth + 1, f"{node.feature} = 1")

def predict_one(tree, row):
    current_node = tree

    while current_node.feature is not None:
        feature_name = current_node.feature
        feature_value = row[feature_name]

        if feature_value == 0:
            current_node = current_node.left
        else:
            current_node = current_node.right

    return current_node.prediction


def predict(tree, features):
    predicted_labels = []

    for row_index, row in features.iterrows():
        predicted_label = predict_one(tree, row)
        predicted_labels.append(predicted_label)

    predictions = pd.Series(predicted_labels, index=features.index, dtype=int)
    return predictions

def error_rate(labels, predictions):
    incorrect_predictions = labels != predictions
    number_of_mistakes = incorrect_predictions.sum()
    number_of_examples = len(labels)

    error = number_of_mistakes / number_of_examples
    return error

def split_xy(data):
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    return features, labels

def accuracy(tree, features, labels):
    predictions = predict(tree, features)
    return 1 - error_rate(labels, predictions)

def best_param(results, param_column, metric_column="Validation accuracy"):
    best_metric = results[metric_column].max()
    tied_params = results.loc[results[metric_column] == best_metric, param_column]
    return tied_params.min(), best_metric, tied_params.tolist()

def prune_tree(node, validation_data):
    if node.feature is None:
        return node

    if validation_data.empty:
        node.feature = None
        node.left = None
        node.right = None
        return node

    split_feature = node.feature
    feature_values = validation_data[split_feature]
    left_validation_data = validation_data[feature_values == 0]
    right_validation_data = validation_data[feature_values == 1]

    node.left = prune_tree(node.left, left_validation_data)
    node.right = prune_tree(node.right, right_validation_data)

    validation_features = validation_data.iloc[:, :-1]
    validation_labels = validation_data.iloc[:, -1]
    subtree_predictions = predict(node, validation_features)
    subtree_mistakes = validation_labels != subtree_predictions
    subtree_error_count = subtree_mistakes.sum()

    leaf_prediction = node.prediction
    leaf_mistakes = validation_labels != leaf_prediction
    leaf_error_count = leaf_mistakes.sum()

    if leaf_error_count <= subtree_error_count:
        node.feature = None
        node.left = None
        node.right = None

    return node



if __name__ == "__main__":
    small_train = pd.read_csv("small_train.tsv", sep="\t")
    small_val = pd.read_csv("small_val.tsv", sep="\t")

    heart_train = pd.read_csv("heart_train.tsv", sep="\t")
    heart_val = pd.read_csv("heart_val.tsv", sep="\t")

    education_train = pd.read_csv("education_train.tsv", sep="\t")
    education_val = pd.read_csv("education_val.tsv", sep="\t")

    t = int(input("Enter Test Number (1, 2, 3, 4, 5, or 6): "))

    ## section 4.3 Q1
    if t == 1:
        datasets = [
            ("small", small_train, small_val),
            ("heart", heart_train, heart_val),
            ("education", education_train, education_val),
        ]

        results = []
        for dataset_name, training_data, validation_data in datasets:
            tree = build_tree(training_data, max_depth=None, threshold=0.0)

            validation_features, validation_labels = split_xy(validation_data)
            validation_accuracy = accuracy(tree, validation_features, validation_labels)

            dataset_result = {
                "Dataset": dataset_name,
                "Validation accuracy (%)": 100 * validation_accuracy,
            }
            results.append(dataset_result)

        accuracy_results = pd.DataFrame(results)
        print(accuracy_results.round(2))

        sys.exit()

    ## section 4.3 Q2
    if t == 2:
        heart_tree_depth4 = build_tree(heart_train, max_depth=4, threshold=0.0)
        print_tree(heart_tree_depth4)
        sys.exit()

    ## section 4.3 Q3
    if t == 3:
        heart_tree_depth4 = build_tree(heart_train, max_depth=4, threshold=0.0)
        heart_tree_copy = deepcopy(heart_tree_depth4)
        pruned_heart_tree = prune_tree(heart_tree_copy, heart_val)
        print_tree(pruned_heart_tree)

        validation_features, validation_labels = split_xy(heart_val)

        before_accuracy = accuracy(heart_tree_depth4, validation_features, validation_labels)
        after_accuracy = accuracy(pruned_heart_tree, validation_features, validation_labels)

        print(f"Validation accuracy before pruning: {before_accuracy:.2%}")
        print(f"Validation accuracy after pruning: {after_accuracy:.2%}")
        sys.exit()

    ## section 4.3 Q4
    if t == 4:
        training_features, training_labels = split_xy(heart_train)
        validation_features, validation_labels = split_xy(heart_val)

        depth_records = []
        for maximum_depth in range(9):  # range(9) includes 0 through 8.
            tree = build_tree(heart_train, max_depth=maximum_depth, threshold=0.0)

            depth_record = {
                "Maximum depth": maximum_depth,
                "Training accuracy": accuracy(tree, training_features, training_labels),
                "Validation accuracy": accuracy(tree, validation_features, validation_labels),
            }
            depth_records.append(depth_record)

        depth_results = pd.DataFrame(depth_records)
        print(depth_results.round(4))

        maximum_depths = depth_results["Maximum depth"]
        training_accuracies = depth_results["Training accuracy"]
        validation_accuracies = depth_results["Validation accuracy"]

        figure, axes = plt.subplots(figsize=(8, 5))
        axes.plot(maximum_depths, training_accuracies, marker="o", label="Training accuracy")
        axes.plot(maximum_depths, validation_accuracies, marker="o", label="Validation accuracy")

        axes.set_xlabel("Maximum depth")
        axes.set_ylabel("Accuracy")
        axes.set_title("Heart dataset: accuracy versus maximum depth")
        axes.set_xticks(range(9))
        axes.set_ylim(0, 1)
        axes.grid(alpha=0.3)
        axes.legend()
        figure.tight_layout()
        plt.show()

        best_depth, best_accuracy, best_depths = best_param(depth_results, "Maximum depth")
        training_never_decreases = training_accuracies.is_monotonic_increasing
        validation_never_decreases = validation_accuracies.is_monotonic_increasing

        print("Training accuracy is nondecreasing:", training_never_decreases)
        print("Validation accuracy is nondecreasing:", validation_never_decreases)
        print(f"Best validation accuracy: {best_accuracy:.2%}")
        print("Depths achieving it:", best_depths)

        sys.exit()

    ## section 4.3 Q5
    if t == 5:
        validation_features, validation_labels = split_xy(heart_val)
        number_of_training_examples = len(heart_train)

        size_records = []
        for minimum_size in range(number_of_training_examples + 1):
            tree = build_tree(
                data=heart_train,
                max_depth=None,
                threshold=0.0,
                min_size=minimum_size,
            )
            size_record = {
                "Minimum splitting size": minimum_size,
                "Validation accuracy": accuracy(tree, validation_features, validation_labels),
            }
            size_records.append(size_record)

        size_results = pd.DataFrame(size_records)
        best_min_size, best_size_accuracy, best_sizes = best_param(size_results, "Minimum splitting size")
        best_min_size = int(best_min_size)

        print("Optimal minimum splitting size:", best_min_size)
        print(f"Validation accuracy: {best_size_accuracy:.2%}")
        print("All sizes tied for best accuracy:", best_sizes)

        sys.exit()

    ## section 4.3 Q6
    if t == 6:
        validation_features, validation_labels = split_xy(heart_val)

        threshold_records = []
        for step in range(101):
            information_gain_threshold = step / 100  # 0.00, 0.01, ..., 1.00.
            tree = build_tree(
                data=heart_train,
                max_depth=None,
                threshold=information_gain_threshold,
                min_size=0,
            )
            threshold_record = {
                "Information gain threshold": information_gain_threshold,
                "Validation accuracy": accuracy(tree, validation_features, validation_labels),
            }
            threshold_records.append(threshold_record)

        threshold_results = pd.DataFrame(threshold_records)
        best_threshold, best_threshold_accuracy, best_thresholds = best_param(
            threshold_results, "Information gain threshold"
        )

        print(f"Optimal information gain threshold: {best_threshold:.2f}")
        print(f"Validation accuracy: {best_threshold_accuracy:.2%}")
        print("All thresholds tied for best accuracy:", best_thresholds)

        sys.exit()


