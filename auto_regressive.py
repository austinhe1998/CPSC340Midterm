import numpy as np
from linear_model import LeastSquares

class AutoRegressive:

    # K is the number of examples we need to predict next one
    def __init__(self, K):
        self.K = K

    def fit(self, D):
        K = self.K
        N, num_features = D.shape
        X = []

        for k in range(K, N + 1):
            X.append(np.concatenate([[1], D[k-K : k-1].flatten()]))

        X = np.array(X, dtype='float')
        y = np.array(D[K-1:, :], dtype='float')

        w = []
        
        model = LeastSquares()

        for f in range(num_features):
            model.fit(X, y[:, f])
            w.append(model.w)
        
        self.X = X
        self.w = np.array(w)

    def predict(self, D, days):
        N, num_features = D.shape
        prev_features = self.X[-1, (num_features + 1):]
        last = D[-1]

        prediction = np.zeros(days)

        for n in range(days):
            features = np.concatenate([[1], prev_features, last.flatten()])
            last = features@self.w.T
            prev_features = features[(num_features + 1):]
            prediction[n] = last[0]
        
        return prediction
        