# Created by jing at 26.02.25

import random
from itertools import combinations


def get_all_combs(given_list):
    # Generate all combinations of all lengths
    all_combinations = []
    for r in range(1, len(given_list) + 1):
        all_combinations.extend(combinations(given_list, r))

    # Convert to a list of lists (optional)
    all_combinations = [list(comb) for comb in all_combinations]
    return all_combinations


def not_all_true(n):
    if n < 1:
        return []

    # Generate a random boolean list
    bool_list = [random.choice([True, False]) for _ in range(n)]

    # Ensure at least one False exists
    if all(bool_list):
        bool_list[random.randint(0, n - 1)] = False

    return bool_list



def at_least_one_true(n):
    if n < 2:
        return [True] if n == 1 else []

    # Generate a random boolean list
    bool_list = [random.choice([True, False]) for _ in range(n)]

    # Ensure at least one True
    if not any(bool_list):
        bool_list[random.randint(0, n - 1)] = True

    # Ensure at least one False
    if all(bool_list):
        bool_list[random.randint(0, n - 1)] = False

    return bool_list


def neg_clu_num(clu_num, min_num, max_num):
    new_clu_num = clu_num
    while new_clu_num == clu_num:
        new_clu_num = random.randint(min_num, max_num)
    clu_num = new_clu_num
    return clu_num

def random_select_unique_mix(lst, n):
    while True:
        selection = random.choices(lst, k=n)
        if len(set(selection)) > 1:  # Ensure at least 2 unique elements
            return selection

def duplicate_maintain_order(lst, n=2):
    return [item for item in lst for _ in range(n)]


def get_proper_sublist(lst):
    if not lst:
        return []  # Return an empty list if the input list is empty
    if len(lst) == 1:
        return []
    sublist_size = random.randint(1, len(lst) - 1)  # Ensure it's a proper sublist
    return random.sample(lst, sublist_size)  # Randomly select elements

def confusion_matrix_elements(predictions, ground_truth):
    TN = sum(1 for p, gt in zip(predictions, ground_truth) if p == 0 and gt == 0)
    FP = sum(1 for p, gt in zip(predictions, ground_truth) if p == 1 and gt == 0)
    FN = sum(1 for p, gt in zip(predictions, ground_truth) if p == 0 and gt == 1)
    TP = sum(1 for p, gt in zip(predictions, ground_truth) if p == 1 and gt == 1)

    return TN, FP, FN, TP


def calculate_metrics(TN, FP, FN, TP):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score
