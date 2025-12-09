import numpy as np


class Binarizer():

    def __init__(self, resolution):
        self.resolution = resolution
        self.thresholds = None
        self.num_features = 0


    def _get_thresholds(self, X):
        #return np.quantile(X, [np.flip(np.linspace(0, 1.0, self.resolution + 1))]).flatten()
        #return np.quantile(X, [(np.linspace(0 + 1/self.resolution, 1.0 - 1/self.resolution, self.resolution))]).flatten()
        return np.quantile(X, [(np.linspace(0, 1.0, self.resolution + 2))]).flatten()
    
    def _encode_X(self, X):
        if X.ndim == 1:
            return X[:, np.newaxis]
        elif X.ndim == 2:
            return X
        else:
            raise ValueError(f"Too many dimensions: {X.ndim}")
    

    def fit(self, X):
        X = self._encode_X(X)
        self.thresholds = []
        
        for i in range(X.shape[1]):
            self.thresholds.append(self._get_thresholds(X[:, i]))
            self.num_features += 1
        

    def transform(self, X):
        X = self._encode_X(X)
        # Quantize pixel values
        X_bin = np.zeros([X.shape[0], X.shape[1]*(self.resolution +2)], dtype=np.uint32)
        for feature in range(X.shape[1]):
            s = feature * (self.resolution + 2)
            for z in range(0, self.resolution+1):
                X_bin[:, s + z] = X[:, feature] >= self.thresholds[feature][z]
            X_bin[X[:, feature] < self.thresholds[feature][0], s] = 0
            X_bin[X[:, feature] > self.thresholds[feature][-1], s + self.resolution + 1] = 1
                
        return X_bin
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
