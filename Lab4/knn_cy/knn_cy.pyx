# knn_cy.pyx
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def knn_cy(np.ndarray[np.int32_t, ndim=1] class_train,
           np.ndarray[np.float64_t, ndim=2] x_train,
           np.ndarray[np.float64_t, ndim=2] x_test,
           int k):

    cdef Py_ssize_t n_test = x_test.shape[0]
    cdef Py_ssize_t n_train = x_train.shape[0]
    cdef Py_ssize_t dim = x_train.shape[1]

    cdef np.ndarray[np.int32_t, ndim=1] x_pred = np.zeros(n_test, dtype=np.int32)
    cdef double[:, :] mv_x_train = x_train
    cdef double[:, :] mv_x_test = x_test
    cdef int[:] mv_class_train = class_train

    # variables à déclarer dès le début
    cdef Py_ssize_t i, j, l, m
    cdef double dist, diff
    cdef np.ndarray[np.float64_t, ndim=1] distances = np.zeros(n_train, dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] idx
    cdef int[:] knn_labels
    cdef np.ndarray[np.int32_t, ndim=1] counts
    cdef int y_pred

    for i in range(n_test):
        for j in range(n_train):
            dist = 0.0
            for l in range(dim):
                diff = mv_x_train[j, l] - mv_x_test[i, l]
                dist += diff * diff
            distances[j] = dist

        # on affecte à idx sans cdef
        idx = np.argsort(distances)[:k]

        knn_labels = class_train[idx]
        counts = np.bincount(knn_labels, minlength=np.max(class_train)+1)
        y_pred = np.argmax(counts)
        x_pred[i] = y_pred

    return x_pred
