import scipy.sparse as sp
import numpy as np
from sklearn.utils import murmurhash3_32


def _sliding_window(tensor, window_size, step_size=1):
    if len(tensor) - window_size >= 0:
        for i in range(len(tensor), 0, -step_size):
            if i - window_size >= 0:
                yield tensor[i - window_size : i]
            else:
                break
    else:
        num_paddings = window_size - len(tensor)
        # Pad sequence with 0s if it is shorter than windows size.
        yield np.pad(tensor, (num_paddings, 0), "constant")


def _generate_sequences(user_ids, item_ids, indices, max_sequence_length, step_size):

    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for seq in _sliding_window(
            item_ids[start_idx:stop_idx], max_sequence_length, step_size
        ):

            yield (user_ids[i], seq)


class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions, but can also be enriched with ratings, timestamps,
    and interaction weights.
    For *implicit feedback* scenarios, user ids and item ids should
    only be provided for user-item pairs where an interaction was
    observed. All pairs that are not provided are treated as missing
    observations, and often interpreted as (implicit) negative
    signals.
    For *explicit feedback* scenarios, user ids, item ids, and
    ratings should be provided for all user-item-rating triplets
    that were observed in the dataset.
    Parameters
    ----------
    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    timestamps: array of np.int32, optional
        array of timestamps
    weights: array of np.float32, optional
        array of weights
    num_users: int, optional
        Number of distinct users in the dataset.
        Must be larger than the maximum user id
        in user_ids.
    num_items: int, optional
        Number of distinct items in the dataset.
        Must be larger than the maximum item id
        in item_ids.
    Attributes
    ----------
    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    timestamps: array of np.int32, optional
        array of timestamps
    weights: array of np.float32, optional
        array of weights
    num_users: int, optional
        Number of distinct users in the dataset.
    num_items: int, optional
        Number of distinct items in the dataset.
    """

    def __init__(
        self,
        user_ids,
        item_ids,
        ratings=None,
        timestamps=None,
        weights=None,
        num_users=None,
        num_items=None,
    ):

        self.num_users = num_users or int(user_ids.max() + 1)
        self.num_items = num_items or int(item_ids.max() + 1)

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.timestamps = timestamps
        self.weights = weights

        self.sequences = None
        self.test_sequences = None

    def __repr__(self):

        return (
            "<Interactions dataset ({num_users} users x {num_items} items "
            "x {num_interactions} interactions)>".format(
                num_users=self.num_users,
                num_items=self.num_items,
                num_interactions=len(self),
            )
        )

    def __len__(self):

        return len(self.user_ids)

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        data = self.ratings if self.ratings is not None else np.ones(len(self))

        return sp.coo_matrix((data, (row, col)), shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()

    def to_sequence(self, sequence_length=5, target_length=3, step_size=None):
        """
        Transform to sequence form.

        Valid subsequences of users' interactions are returned. For
        example, if a user interacted with items [1, 2, 3, 4, 5, 6, 7, 8, 9], the
        returned interactions matrix at sequence length 5 and target length 3
        will be be given by:

        sequences:

           [[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]]

        targets:

           [[6, 7],
            [7, 8],
            [8, 9]]

        sequence for test (the last 'sequence_length' items of each user's sequence):

        [[5, 6, 7, 8, 9]]

        Parameters
        ----------

        sequence_length: int
            Sequence length. Subsequences shorter than this
            will be left-padded with zeros.
        target_length: int
            Sequence target length.
        """

        max_sequence_length = sequence_length + target_length

        # Sort first by user id
        sort_indices = np.lexsort((self.user_ids,))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]

        user_ids, indices, counts = np.unique(
            user_ids, return_index=True, return_counts=True
        )

        num_subsequences = sum(
            [
                c - max_sequence_length + 1 if c >= max_sequence_length else 1
                for c in counts
            ]
        )

        sequences = np.zeros((num_subsequences, sequence_length), dtype=np.int64)
        sequences_targets = np.zeros((num_subsequences, target_length), dtype=np.int64)
        sequence_users = np.empty(num_subsequences, dtype=np.int64)

        test_sequences = np.zeros((self.num_users, sequence_length), dtype=np.int64)
        test_users = np.empty(self.num_users, dtype=np.int64)

        _uid = None
        for i, (uid, item_seq) in enumerate(
            _generate_sequences(
                user_ids, item_ids, indices, max_sequence_length, step_size
            )
        ):
            if uid != _uid:
                test_sequences[uid][:] = item_seq[-sequence_length:]
                test_users[uid] = uid
                _uid = uid

            sequences_targets[i][:] = item_seq[-target_length:]
            sequences[i][:] = item_seq[:sequence_length]
            sequence_users[i] = uid

        self.sequences = SequenceInteractions(
            sequence_users, sequences, sequences_targets
        )
        self.test_sequences = SequenceInteractions(test_users, test_sequences)


class SequenceInteractions(object):
    """
    Interactions encoded as a sequence matrix.
    Parameters
    ----------
    sequences: array of np.int32 of shape (num_sequences x max_sequence_length)
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    num_items: int, optional
        The number of distinct items in the data
    Attributes
    ----------
    sequences: array of np.int32 of shape (num_sequences x max_sequence_length)
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    """

    def __init__(self, user_ids, sequences, targets=None):
        self.user_ids = user_ids
        self.sequences = sequences
        self.targets = targets

        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]


def _index_or_none(array, shuffle_index):

    if array is None:
        return None
    else:
        return array[shuffle_index]


def shuffle_interactions(interactions, random_state=None):
    """
    Shuffle interactions.
    Parameters
    ----------
    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.
    Returns
    -------
    interactions: :class:`spotlight.interactions.Interactions`
        The shuffled interactions.
    """

    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(interactions.user_ids))
    random_state.shuffle(shuffle_indices)

    return Interactions(
        interactions.user_ids[shuffle_indices],
        interactions.item_ids[shuffle_indices],
        ratings=_index_or_none(interactions.ratings, shuffle_indices),
        timestamps=_index_or_none(interactions.timestamps, shuffle_indices),
        weights=_index_or_none(interactions.weights, shuffle_indices),
        num_users=interactions.num_users,
        num_items=interactions.num_items,
    )


def random_train_test_split(interactions, test_percentage=0.2, random_state=None):
    """
    Randomly split interactions between training and testing.
    Parameters
    ----------
    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    test_percentage: float, optional
        The fraction of interactions to place in the test set.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.
    Returns
    -------
    (train, test): (:class:`spotlight.interactions.Interactions`,
                    :class:`spotlight.interactions.Interactions`)
         A tuple of (train data, test data)
    """

    interactions = shuffle_interactions(interactions, random_state=random_state)

    cutoff = int((1.0 - test_percentage) * len(interactions))

    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)

    train = Interactions(
        interactions.user_ids[train_idx],
        interactions.item_ids[train_idx],
        ratings=_index_or_none(interactions.ratings, train_idx),
        timestamps=_index_or_none(interactions.timestamps, train_idx),
        weights=_index_or_none(interactions.weights, train_idx),
        num_users=interactions.num_users,
        num_items=interactions.num_items,
    )
    test = Interactions(
        interactions.user_ids[test_idx],
        interactions.item_ids[test_idx],
        ratings=_index_or_none(interactions.ratings, test_idx),
        timestamps=_index_or_none(interactions.timestamps, test_idx),
        weights=_index_or_none(interactions.weights, test_idx),
        num_users=interactions.num_users,
        num_items=interactions.num_items,
    )

    return train, test


def user_based_train_test_split(interactions, test_percentage=0.2, random_state=None):
    """
    Split interactions between a train and a test set based on
    user ids, so that a given user's entire interaction history
    is either in the train, or the test set.
    Parameters
    ----------
    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    test_percentage: float, optional
        The fraction of users to place in the test set.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.
    Returns
    -------
    (train, test): (:class:`spotlight.interactions.Interactions`,
                    :class:`spotlight.interactions.Interactions`)
         A tuple of (train data, test data)
    """

    if random_state is None:
        random_state = np.random.RandomState()

    minint = np.iinfo(np.uint32).min
    maxint = np.iinfo(np.uint32).max

    seed = random_state.randint(minint, maxint, dtype=np.int64)

    in_test = (
        murmurhash3_32(interactions.user_ids, seed=seed, positive=True) % 100 / 100.0
    ) < test_percentage
    in_train = np.logical_not(in_test)

    train = Interactions(
        interactions.user_ids[in_train],
        interactions.item_ids[in_train],
        ratings=_index_or_none(interactions.ratings, in_train),
        timestamps=_index_or_none(interactions.timestamps, in_train),
        weights=_index_or_none(interactions.weights, in_train),
        num_users=interactions.num_users,
        num_items=interactions.num_items,
    )
    test = Interactions(
        interactions.user_ids[in_test],
        interactions.item_ids[in_test],
        ratings=_index_or_none(interactions.ratings, in_test),
        timestamps=_index_or_none(interactions.timestamps, in_test),
        weights=_index_or_none(interactions.weights, in_test),
        num_users=interactions.num_users,
        num_items=interactions.num_items,
    )

    return train, test
