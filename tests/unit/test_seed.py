"""Tests for rinchen.utils.seed.set_all."""

import random

from rinchen.utils import seed


def test_set_all_makes_random_deterministic():
    seed.set_all(42)
    first = [random.random() for _ in range(5)]

    seed.set_all(42)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_set_all_different_seeds_differ():
    seed.set_all(1)
    first = random.random()

    seed.set_all(2)
    second = random.random()

    assert first != second
