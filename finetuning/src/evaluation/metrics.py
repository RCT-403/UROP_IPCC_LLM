from typing import List, Dict

def compute_accuracy(predictions: List[int], labels: List[int]) -> float:
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels) if labels else 0.0

def compute_precision(predictions: List[int], labels: List[int]) -> float:
    true_positive = sum(p == 1 and l == 1 for p, l in zip(predictions, labels))
    false_positive = sum(p == 1 and l == 0 for p, l in zip(predictions, labels))
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0

def compute_recall(predictions: List[int], labels: List[int]) -> float:
    true_positive = sum(p == 1 and l == 1 for p, l in zip(predictions, labels))
    false_negative = sum(p == 0 and l == 1 for p, l in zip(predictions, labels))
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

def compute_f1_score(predictions: List[int], labels: List[int]) -> float:
    precision = compute_precision(predictions, labels)
    recall = compute_recall(predictions, labels)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def compute_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    return {
        "accuracy": compute_accuracy(predictions, labels),
        "precision": compute_precision(predictions, labels),
        "recall": compute_recall(predictions, labels),
        "f1_score": compute_f1_score(predictions, labels),
    }