from scipy.stats import mode
from pickle import load, dump
from numpy import empty, mean, linalg, diag, einsum, argsort, genfromtxt, dtype

load_or_pickle_data = 'pickle'   # 'pickle' / 'pickle'
k_PCA = 40
k_NN = 6

# ---------------------------------------------------------------------------------------------------------------------
# Reducing the dimensionality of the MNIST data with PCA (via SVD) before running KNN can save both time and accuracy
# (lower dimensions means less calculations and potentially less overfitting):
def svd_pca(data, k):
    """Reduce the data using its K principal components."""

    data = data.astype("float64")
    data -= mean(data, axis=0)
    U, S, V = linalg.svd(data, full_matrices=False)
    return U[:,:k].dot(diag(S)[:k,:k])

# ---------------------------------------------------------------------------------------------------------------------
# Load or Pickle the data:
def load_data(load_or_pickle_data_, file_):

    X = empty([1,1])

    if load_or_pickle_data_ == 'load':

        # Numpy's genfromtxt function is an easy way to get the .csv data into a matrix:
        X = genfromtxt(file_, delimiter=',', skip_header=1).astype(dtype('uint8'))

        # Serialize the numpy matrix with the pickle module after the first load:
        with open(file_ + '.p', 'wb') as f:
            dump(X, f)

    elif load_or_pickle_data_ == 'pickle':

        # Load the saved pickle object on all subsequent runs (quicker):
        with open(file_ + '.p', 'rb') as f:
            X = load(f)

    return X

# ---------------------------------------------------------------------------------------------------------------------
# KNN class, based on euclidean distance (einsum):
class KNN:
    def __init__(self, data_, labels_, k_):
        self.data = data_
        self.labels = labels_
        self.k = k_

    def predict(self, sample):
        differences = (self.data - sample)
        distances = einsum('ij, ij->i', differences, differences)  # distance[x]=sum((differences[x]).^2)
        nearest = self.labels[argsort(distances)[:self.k]]
        return mode(nearest)[0][0]

# ---------------------------------------------------------------------------------------------------------------------
# Main course:
print "1. Loading Train and Test data..."
train_data_ = load_data(load_or_pickle_data, 'data/train.csv')
test_data_  = load_data(load_or_pickle_data, 'data/test.csv')

print "2. Building and Training a KNN classifier object..."
train_data = svd_pca(train_data_[:,1:], k_PCA)
labels = train_data_[:,0]
KNN_obj = KNN(train_data, labels, k_NN)

print "3. Testing the KNN classifier over the train data..."
for i in range(10):
    pred = KNN_obj.predict(train_data[i])
    res = 'FAIL'
    if pred == labels[i]:
        res = 'PASS'
    print("   - prediction = %d --> %s" % (pred, res))

print "4. Testing the KNN classifier over the test data..."
test_data = svd_pca(test_data_[:,1:], k_PCA)
for i in range(10):
    pred = KNN_obj.predict(test_data[i])
    print("   - prediction = %d" % pred)

print "Done!"
