"""Centralized randomness management for reproducibility.

This module provides a single source of randomness for the entire codebase,
ensuring reproducibility when the same seed is used.

Usage:
    from ocr_icelandic.randomness import rng, np_rng, set_seed

    # Set seed for reproducibility (typically done once at startup)
    set_seed(42)

    # Use the random number generators
    value = rng.random()
    choice = rng.choice(['a', 'b', 'c'])
    np_array = np_rng.uniform(0, 1, size=(10,))
"""

import random as _random
from typing import Any, TypeVar
from collections.abc import Sequence

import numpy as np

from ocr_icelandic.logging_config import get_logger

logger = get_logger(__name__)

# Type variable for generic sequence operations
T = TypeVar("T")

# Global random instances
_rng = _random.Random()
_np_rng = np.random.Generator(np.random.PCG64())

# Track whether seed has been set
_seed_value: int | None = None


def set_seed(seed: int) -> None:
    """Set the global random seed for reproducibility.

    This function seeds both the Python random instance and the NumPy
    random generator used throughout the codebase.

    Args:
        seed: The seed value to use for random number generation.
    """
    global _rng, _np_rng, _seed_value

    _seed_value = seed
    _rng = _random.Random(seed)
    _np_rng = np.random.Generator(np.random.PCG64(seed))

    logger.info("Random seed set to %d for reproducibility", seed)


def get_seed() -> int | None:
    """Get the current seed value, or None if not explicitly set.

    Returns:
        The seed value if set, None otherwise.
    """
    return _seed_value


def reset() -> None:
    """Reset random generators without a specific seed.

    Creates new random instances without explicit seeding,
    resulting in non-reproducible randomness.
    """
    global _rng, _np_rng, _seed_value

    _seed_value = None
    _rng = _random.Random()
    _np_rng = np.random.Generator(np.random.PCG64())

    logger.debug("Random generators reset (non-reproducible)")


# ============================================================================
# Python random module compatible interface
# ============================================================================


def random() -> float:
    """Return a random float in the range [0.0, 1.0)."""
    return _rng.random()


def randint(a: int, b: int) -> int:
    """Return a random integer N such that a <= N <= b."""
    return _rng.randint(a, b)


def uniform(a: float, b: float) -> float:
    """Return a random float N such that a <= N <= b."""
    return _rng.uniform(a, b)


def choice(seq: Sequence[T]) -> T:
    """Return a random element from the non-empty sequence."""
    return _rng.choice(seq)


def choices(
    population: Sequence[T],
    weights: Sequence[float] | None = None,
    *,
    cum_weights: Sequence[float] | None = None,
    k: int = 1,
) -> list[T]:
    """Return a k-sized list of elements chosen with replacement."""
    return _rng.choices(population, weights=weights, cum_weights=cum_weights, k=k)


def sample(population: Sequence[T], k: int) -> list[T]:
    """Return a k-length list of unique elements from population."""
    return _rng.sample(population, k)


def shuffle(x: list[Any]) -> None:
    """Shuffle list x in place."""
    _rng.shuffle(x)


def gauss(mu: float = 0.0, sigma: float = 1.0) -> float:
    """Return a random float from a Gaussian distribution."""
    return _rng.gauss(mu, sigma)


# ============================================================================
# NumPy random compatible interface
# ============================================================================


def np_uniform(
    low: float = 0.0,
    high: float = 1.0,
    size: int | tuple[int, ...] | None = None,
) -> float | np.ndarray:
    """Draw samples from a uniform distribution.

    Args:
        low: Lower boundary of output interval.
        high: Upper boundary of output interval.
        size: Output shape. If None, returns a single value.

    Returns:
        Drawn samples from the parameterized uniform distribution.
    """
    return _np_rng.uniform(low, high, size)


def np_randint(
    low: int,
    high: int | None = None,
    size: int | tuple[int, ...] | None = None,
) -> int | np.ndarray:
    """Return random integers from low (inclusive) to high (exclusive).

    Args:
        low: Lowest integer or upper bound if high is None.
        high: Upper bound (exclusive).
        size: Output shape. If None, returns a single value.

    Returns:
        Random integers from the discrete uniform distribution.
    """
    return _np_rng.integers(low, high, size=size)


def np_choice(
    a: int | Sequence[T] | np.ndarray,
    size: int | tuple[int, ...] | None = None,
    replace: bool = True,
    p: Sequence[float] | np.ndarray | None = None,
) -> T | np.ndarray:
    """Generate a random sample from a given array.

    Args:
        a: Array or int to draw from. If int, draw from np.arange(a).
        size: Output shape. If None, returns a single value.
        replace: Whether sampling is with replacement.
        p: Probabilities associated with each entry.

    Returns:
        Randomly sampled values.
    """
    return _np_rng.choice(a, size=size, replace=replace, p=p)


# ============================================================================
# Direct access to underlying generators (for advanced use cases)
# ============================================================================


@property
def rng() -> _random.Random:
    """Get the underlying Python Random instance."""
    return _rng


@property
def np_rng() -> np.random.Generator:
    """Get the underlying NumPy Generator instance."""
    return _np_rng


# Export the underlying instances directly for convenience
# These should be used when direct access to the Random/Generator is needed
def get_rng() -> _random.Random:
    """Get the underlying Python Random instance."""
    return _rng


def get_np_rng() -> np.random.Generator:
    """Get the underlying NumPy Generator instance."""
    return _np_rng
