import numpy as np
from linear_model import LeastSquares

class AutoRegressive:

    # K is the number of examples we need to predict next one
    def __init__(self, K):
        self.K = K

    def fit(self, D):
        K = self.K
        N = D.shape[0]
        X = []

        for k in range(K, N + 1):
            X.append(np.concatenate([[1], D[k-K : k-1]]))

        X = np.array(X, dtype='float')
        y = np.array(D[K-1:], dtype='float')
        model = LeastSquares()
        model.fit(X, y)
        self.y = y
        self.X = X
        self.w = model.w

    def predict(self, D, days):
        prev_features = self.X[-1, 2:]
        last = D[-1]

        prediction = np.zeros(days)

        for n in range(days):
            features = np.concatenate([[1], prev_features, [last]])
            last = features@self.w
            prev_features = features[2:]
            prediction[n] = last
        
        return prediction
        