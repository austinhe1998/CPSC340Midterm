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
            X.append(np.concatenate([[1], D[k-K : k-1].flatten()]))

        X = np.array(X, dtype='float')
        y1 = np.array(D[K-1:, 0], dtype='float')
        y2 = np.array(D[K-1:, 1], dtype='float')
        model1 = LeastSquares()
        model2 = LeastSquares()
        model1.fit(X, y1)
        model2.fit(X, y2)
        self.X = X
        self.w1 = model1.w
        self.w2 = model2.w

    def predict(self, D, days):
        prev_features = self.X[-1, 3:]
        last = D[-1]

        prediction = np.zeros(days)

        for n in range(days):
            features = np.concatenate([[1], prev_features, last.flatten()])
            y1 = features@self.w1
            y2 = features@self.w2
            last = np.array([y1, y2])
            prev_features = features[3:]
            prediction[n] = y2
        
        return prediction
        