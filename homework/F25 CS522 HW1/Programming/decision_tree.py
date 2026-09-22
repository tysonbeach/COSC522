import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

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
    """Print one line per node, indenting each child under its parent.

    `branch` describes the edge into this node, e.g. "chest_pain = 0".
    """
    counts = f"[{node.count_0} 0/{node.count_1} 1]"

    if branch is None:
        # The root has no incoming edge, so only its counts are shown.
        print(counts)
    else:
        print("| " * depth + f"{branch}: {counts}")

    # A leaf has no split feature, so there are no children to print.
    if node.feature is None:
        return

    print_tree(node.left, depth + 1, f"{node.feature} = 0")
    print_tree(node.right, depth + 1, f"{node.feature} = 1")

def predict_one(tree, row):
    """Follow one example's feature values from the root to a leaf."""
    current_node = tree

    # Continue until there is no split feature: that means we reached a leaf.
    while current_node.feature is not None:
        feature_name = current_node.feature
        feature_value = row[feature_name]

        if feature_value == 0:
            current_node = current_node.left
        else:
            current_node = current_node.right

    return current_node.prediction


def predict(tree, features):
    """Predict every row and keep the input DataFrame's row indexes."""
    predicted_labels = []

    # iterrows() gives the row index and the row's values as a Series.
    for row_index, row in features.iterrows():
        predicted_label = predict_one(tree, row)
        predicted_labels.append(predicted_label)

    # Matching indexes let pandas compare each prediction with its true label.
    predictions = pd.Series(predicted_labels, index=features.index, dtype=int)
    return predictions

def error_rate(labels, predictions):
    """Return the fraction of wrong predictions for matching row indexes."""
    incorrect_predictions = labels != predictions
    number_of_mistakes = incorrect_predictions.sum()
    number_of_examples = len(labels)

    error = number_of_mistakes / number_of_examples
    return error

def prune_tree(node, validation_data):
    """Modify this subtree using validation errors; prefer a leaf on ties."""
    # A leaf is already as small as it can be.
    if node.feature is None:
        return node

    # No validation examples reach here: both choices make zero mistakes.
    # Keep the training prediction, but remove the split and its children.
    if validation_data.empty:
        node.feature = None
        node.left = None
        node.right = None
        return node

    # 1. Send each validation row down the same branch used for prediction.
    split_feature = node.feature
    feature_values = validation_data[split_feature]
    left_validation_data = validation_data[feature_values == 0]
    right_validation_data = validation_data[feature_values == 1]

    # 2. Simplify the children before deciding whether to remove their parent.
    node.left = prune_tree(node.left, left_validation_data)
    node.right = prune_tree(node.right, right_validation_data)

    # 3. Count mistakes if we keep the subtree after pruning its children.
    validation_features = validation_data.iloc[:, :-1]
    validation_labels = validation_data.iloc[:, -1]
    subtree_predictions = predict(node, validation_features)
    subtree_mistakes = validation_labels != subtree_predictions
    subtree_error_count = subtree_mistakes.sum()

    # 4. Count mistakes if every row instead receives one constant prediction.
    # This prediction was learned from TRAINING labels, not validation labels.
    leaf_prediction = node.prediction
    leaf_mistakes = validation_labels != leaf_prediction
    leaf_error_count = leaf_mistakes.sum()

    # 5. A tie also favors the leaf because it is the smaller tree.
    if leaf_error_count <= subtree_error_count:
        node.feature = None
        node.left = None
        node.right = None

    return node



if __name__ == "__main__":
    # A TSV file uses tabs to separate columns. Pandas reads its first row as names.
    # Keep each training set separate from the validation set used to evaluate it.
    small_train = pd.read_csv("small_train.tsv", sep="\t")
    small_val = pd.read_csv("small_val.tsv", sep="\t")

    heart_train = pd.read_csv("heart_train.tsv", sep="\t")
    heart_val = pd.read_csv("heart_val.tsv", sep="\t")

    education_train = pd.read_csv("education_train.tsv", sep="\t")
    education_val = pd.read_csv("education_val.tsv", sep="\t")

    ## Testing
    
    identical_labels = pd.Series([1, 1, 1, 1])
    balanced_labels = pd.Series([0, 0, 1, 1])
    mostly_one_labels = pd.Series([0, 1, 1, 1])
    small_training_labels = small_train.iloc[:, -1]

    print("All labels the same:", entropy(identical_labels))
    print("Evenly split labels:", entropy(balanced_labels))
    print("Mostly one label:", entropy(mostly_one_labels))
    print("Small training labels:", entropy(small_training_labels))

    # Exclude the last column: it is the answer, not a candidate feature.
    feature_names = small_train.columns[:-1]

    for feature_name in feature_names:
        information_gain = mutual_information(small_train, feature_name)
        print(f"{feature_name}: {information_gain:.6f} bits")

    default_choice = best_feature(small_train)
    higher_threshold_choice = best_feature(small_train, threshold=0.30)

    print("Default threshold:", default_choice)
    print("Threshold of 0.30:", higher_threshold_choice)

    mostly_zero_labels = pd.Series([0, 0, 0, 1])
    mostly_one_labels = pd.Series([0, 1, 1, 1])
    tied_labels = pd.Series([0, 0, 1, 1])
    small_training_labels = small_train.iloc[:, -1]

    print("Mostly zeros:", majority_vote(mostly_zero_labels))
    print("Mostly ones:", majority_vote(mostly_one_labels))
    print("Tied labels:", majority_vote(tied_labels))
    print("Small training labels:", majority_vote(small_training_labels))
    

    ## Create Root node
    small_training_labels = small_train.iloc[:, -1]
    root = Node(small_training_labels)

    print(f"Label counts: [{root.count_0} 0/{root.count_1} 1]")
    print("Prediction:", root.prediction)
    print("Split feature:", root.feature)
    print("Children:", root.left, root.right)

    small_tree = build_tree(small_train, max_depth=2)

    print("Root split:", small_tree.feature)
    print("Split where chest_pain = 0:", small_tree.left.feature)
    print("Split where chest_pain = 1:", small_tree.right.feature)

    depth_0_tree = build_tree(small_train, max_depth=0)
    print("\nDepth-0 split:", depth_0_tree.feature)
    print("Depth-0 prediction:", depth_0_tree.prediction)

    print_tree(small_tree)

    print_tree(depth_0_tree)

    # Separate the inputs from the answers before making predictions.
    training_features = small_train.iloc[:, :-1]
    training_labels = small_train.iloc[:, -1]
    validation_features = small_val.iloc[:, :-1]
    validation_labels = small_val.iloc[:, -1]

    small_train_predictions = predict(small_tree, training_features)
    small_val_predictions = predict(small_tree, validation_features)

    train_error = error_rate(training_labels, small_train_predictions)
    val_error = error_rate(validation_labels, small_val_predictions)
    train_accuracy = 1 - train_error
    val_accuracy = 1 - val_error

    print(f"Training error: {train_error:.6f}")
    print(f"Validation error: {val_error:.6f}")
    print(f"Training accuracy: {train_accuracy:.2%}")
    print(f"Validation accuracy: {val_accuracy:.2%}")

    validation_labels = small_val.iloc[:, -1]
    prediction_comparison = pd.DataFrame({
        "Actual": validation_labels,
        "Predicted": small_val_predictions,
    })
    prediction_comparison.head()

    ## section 4.3 Q1

    # Each tuple contains a name, its training data, and its validation data.
    datasets = [
        ("small", small_train, small_val),
        ("heart", heart_train, heart_val),
        ("education", education_train, education_val),
    ]

    results = []
    for dataset_name, training_data, validation_data in datasets:
        tree = build_tree(training_data, max_depth=None, threshold=0.0)

        validation_features = validation_data.iloc[:, :-1]
        validation_labels = validation_data.iloc[:, -1]
        predictions = predict(tree, validation_features)
        validation_error = error_rate(validation_labels, predictions)
        validation_accuracy = 1 - validation_error

        dataset_result = {
            "Dataset": dataset_name,
            "Validation accuracy (%)": 100 * validation_accuracy,
        }
        results.append(dataset_result)

    # Keep the original numeric results; round only the displayed table.
    accuracy_results = pd.DataFrame(results)
    print(accuracy_results.round(2))

    ## section 4.3 Q2

    heart_tree_depth4 = build_tree(heart_train, max_depth=4, threshold=0.0)
    print_tree(heart_tree_depth4)

    ### section 4.3 Q3

    # Copy all descendants as well as the root before pruning changes the tree.
    heart_tree_copy = deepcopy(heart_tree_depth4)
    pruned_heart_tree = prune_tree(heart_tree_copy, heart_val)
    print_tree(pruned_heart_tree)

    validation_features = heart_val.iloc[:, :-1]
    validation_labels = heart_val.iloc[:, -1]

    before_predictions = predict(heart_tree_depth4, validation_features)
    after_predictions = predict(pruned_heart_tree, validation_features)

    before_error = error_rate(validation_labels, before_predictions)
    after_error = error_rate(validation_labels, after_predictions)
    before_accuracy = 1 - before_error
    after_accuracy = 1 - after_error

    print(f"Validation accuracy before pruning: {before_accuracy:.2%}")
    print(f"Validation accuracy after pruning: {after_accuracy:.2%}")

    ## section 4.3 Q4

    # These rows stay the same throughout the experiment; only depth changes.
    training_features = heart_train.iloc[:, :-1]
    training_labels = heart_train.iloc[:, -1]
    validation_features = heart_val.iloc[:, :-1]
    validation_labels = heart_val.iloc[:, -1]

    depth_records = []
    for maximum_depth in range(9):  # range(9) includes 0 through 8.
        tree = build_tree(heart_train, max_depth=maximum_depth, threshold=0.0)

        training_predictions = predict(tree, training_features)
        validation_predictions = predict(tree, validation_features)
        training_error = error_rate(training_labels, training_predictions)
        validation_error = error_rate(validation_labels, validation_predictions)

        depth_record = {
            "Maximum depth": maximum_depth,
            "Training accuracy": 1 - training_error,
            "Validation accuracy": 1 - validation_error,
        }
        depth_records.append(depth_record)

    depth_results = pd.DataFrame(depth_records)
    print(depth_results.round(4))

    # Give each column a name that describes its role in the plot.
    maximum_depths = depth_results["Maximum depth"]
    training_accuracies = depth_results["Training accuracy"]
    validation_accuracies = depth_results["Validation accuracy"]

    # The figure is the full image; the axes are the area containing the curves.
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

    validation_accuracies = depth_results["Validation accuracy"]
    best_accuracy = validation_accuracies.max()

    # A True/False mask selects every row tied for the best accuracy.
    is_best_accuracy = validation_accuracies == best_accuracy
    best_depth_rows = depth_results[is_best_accuracy]
    best_depths = best_depth_rows["Maximum depth"].tolist()

    training_accuracies = depth_results["Training accuracy"]
    training_never_decreases = training_accuracies.is_monotonic_increasing
    validation_never_decreases = validation_accuracies.is_monotonic_increasing

    print("Training accuracy is nondecreasing:", training_never_decreases)
    print("Validation accuracy is nondecreasing:", validation_never_decreases)
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print("Depths achieving it:", best_depths)

    ## section 4.3 Q5

    validation_features = heart_val.iloc[:, :-1]
    validation_labels = heart_val.iloc[:, -1]
    number_of_training_examples = len(heart_train)

    size_records = []
    # Add 1 because range excludes its upper endpoint: we want to include 200.
    for minimum_size in range(number_of_training_examples + 1):
        tree = build_tree(
            data=heart_train,
            max_depth=None,
            threshold=0.0,
            min_size=minimum_size,
        )
        predictions = predict(tree, validation_features)
        validation_error = error_rate(validation_labels, predictions)

        size_record = {
            "Minimum splitting size": minimum_size,
            "Validation accuracy": 1 - validation_error,
        }
        size_records.append(size_record)

    size_results = pd.DataFrame(size_records)

    # First find the best accuracy, then collect all sizes that achieved it.
    validation_accuracies = size_results["Validation accuracy"]
    best_size_accuracy = validation_accuracies.max()
    is_best_accuracy = validation_accuracies == best_size_accuracy
    best_size_rows = size_results[is_best_accuracy]
    best_sizes = best_size_rows["Minimum splitting size"]

    # The assignment asks for the smallest size among ties.
    best_min_size = int(best_sizes.min())

    print("Optimal minimum splitting size:", best_min_size)
    print(f"Validation accuracy: {best_size_accuracy:.2%}")
    print("All sizes tied for best accuracy:", best_sizes.tolist())

    ## section 4.3 Q6

    validation_features = heart_val.iloc[:, :-1]
    validation_labels = heart_val.iloc[:, -1]

    threshold_records = []
    for step in range(101):
        information_gain_threshold = step / 100  # 0.00, 0.01, ..., 1.00.
        tree = build_tree(
            data=heart_train,
            max_depth=None,
            threshold=information_gain_threshold,
            min_size=0,
        )
        predictions = predict(tree, validation_features)
        validation_error = error_rate(validation_labels, predictions)

        threshold_record = {
            "Information gain threshold": information_gain_threshold,
            "Validation accuracy": 1 - validation_error,
        }
        threshold_records.append(threshold_record)

    threshold_results = pd.DataFrame(threshold_records)

    # Use the same selection steps as the minimum-size experiment.
    validation_accuracies = threshold_results["Validation accuracy"]
    best_threshold_accuracy = validation_accuracies.max()
    is_best_accuracy = validation_accuracies == best_threshold_accuracy
    best_threshold_rows = threshold_results[is_best_accuracy]
    best_thresholds = best_threshold_rows["Information gain threshold"]
    best_threshold = best_thresholds.min()

    print(f"Optimal information gain threshold: {best_threshold:.2f}")
    print(f"Validation accuracy: {best_threshold_accuracy:.2%}")
    print("All thresholds tied for best accuracy:", best_thresholds.tolist())




