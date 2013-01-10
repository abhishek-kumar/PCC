import sklearn.cross_validation
from sklearn.cross_validation import *



def _validate_stratified_shuffle_split(y, test_size, train_size):
    y = unique(y, return_inverse=True)[1]
    if np.min(np.bincount(y)) < 2:
        raise ValueError("The least populated class in y has only 1"
                         " member, which is too few. The minimum"
                         " number of labels for any class cannot"
                         " be less than 2.")

    if isinstance(test_size, float) and test_size >= 1.:
        raise ValueError(
            'test_size=%f should be smaller '
            'than 1.0 or be an integer' % test_size)
    elif isinstance(test_size, int) and test_size >= y.size:
        raise ValueError(
            'test_size=%d should be smaller '
            'than the number of samples %d' % (test_size, y.size))

    if train_size is not None:
        if isinstance(train_size, float) and train_size >= 1.:
            raise ValueError("train_size=%f should be smaller "
                             "than 1.0 or be an integer" % train_size)
        elif isinstance(train_size, int) and train_size >= y.size:
            raise ValueError("train_size=%d should be smaller "
                             "than the number of samples %d" %
                             (train_size, y.size))

    if isinstance(test_size, float):
        n_test = ceil(test_size * y.size)
    else:
        n_test = float(test_size)

    if train_size is None:
        if isinstance(test_size, float):
            n_train = y.size - n_test
        else:
            n_train = float(y.size - test_size)
    else:
        if isinstance(train_size, float):
            n_train = floor(train_size * y.size)
        else:
            n_train = float(train_size)

    if n_train + n_test > y.size:
        raise ValueError('The sum of n_train and n_test = %d, should '
                         'be smaller than the number of samples %d. '
                         'Reduce test_size and/or train_size.' %
                         (n_train + n_test, y.size))

    return n_train, n_test

class StratifiedShuffleSplit(object):
    """Stratified ShuffleSplit cross validation iterator

    Provides train/test indices to split data in train test sets.

    This cross-validation object is a merge of StratifiedKFold and
    ShuffleSplit, which returns stratified randomized folds. The folds
    are made by preserving the percentage of samples for each class.

    Note: like the ShuffleSplit strategy, stratified random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Parameters
    ----------
    y: array, [n_samples]
        Labels of samples.

    n_iterations : int (default 10)
        Number of re-shuffling & splitting iterations.

    test_size : float (default 0.1) or int
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples.

    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test fraction.

    indices: boolean, optional (default True)
        Return train/test split as arrays of indices, rather than a boolean
        mask array. Integer indices are required when dealing with sparse
        matrices, since those cannot be indexed by boolean masks.

    Examples
    --------
    >>> from sklearn.cross_validation import StratifiedShuffleSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> sss = StratifiedShuffleSplit(y, 3, test_size=0.5, random_state=0)
    >>> len(sss)
    3
    >>> print sss       # doctest: +ELLIPSIS
    StratifiedShuffleSplit(labels=[0 0 1 1], n_iterations=3, ...)
    >>> for train_index, test_index in sss:
    ...    print "TRAIN:", train_index, "TEST:", test_index
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 3] TEST: [1 2]
    TRAIN: [0 2] TEST: [1 3]
    TRAIN: [1 2] TEST: [0 3]
    """

    def __init__(self, y, n_iterations=10, test_size=0.1,
                 train_size=None, indices=True, random_state=None):

        self.y = np.asarray(y)
        self.n = self.y.shape[0]
        self.n_iterations = n_iterations
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.indices = indices
        self.n_train, self.n_test = \
            _validate_stratified_shuffle_split(y, test_size, train_size)

    def __iter__(self):
        rng = check_random_state(self.random_state)

        y = self.y.copy()
        n = y.size
        k = ceil(n / self.n_test)
        l = floor((n - self.n_test) / self.n_train)

        for i in xrange(self.n_iterations):
            ik = i % k
            permutation = rng.permutation(self.n)
            idx = np.argsort(y[permutation])
            ind_test = permutation[idx[ik::k]]
            inv_test = np.setdiff1d(idx, idx[ik::k])
            train_idx = idx[np.where(in1d(idx, inv_test))[0]]
            ind_train = permutation[train_idx[::l]][:self.n_train]
            test_index = ind_test
            train_index = ind_train

            if not self.indices:
                test_index = np.zeros(n, dtype=np.bool)
                test_index[ind_test] = True
                train_index = np.zeros(n, dtype=np.bool)
                train_index[ind_train] = True

            yield train_index, test_index

    def __repr__(self):
        return ('%s(labels=%s, n_iterations=%d, test_size=%s, indices=%s, '
                'random_state=%s)' % (
                    self.__class__.__name__,
                    self.y,
                    self.n_iterations,
                    str(self.test_size),
                    self.indices,
                    self.random_state,
                ))

    def __len__(self):
        return self.n_iterations

