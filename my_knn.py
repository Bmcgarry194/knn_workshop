class KNN():
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test, k=3):
        
        predictions = np.zeros(X_test.shape[0])
        
        for i, point in enumerate(X_test):
            distances = self._get_distances(point)
            k_nearest = self._get_k_nearest(distances, k)
            prediction = self._get_predicted_value(k_nearest)
            predictions[i] = prediction
            
        return predictions
    
    #helper functions
    def _get_distances(self, x):
        '''Take an single point and return an array of distances to every point in our dataset'''
        distances = np.zeros(self.X_train.shape[0])
        for i, point in enumerate(self.X_train):
            distances[i] = euc(x, point)
        return distances
    
    def _get_k_nearest(self, distances, k):
        '''Take in the an array of distances and return the indices of the k nearest points'''
        nearest = np.argsort(distances)[:k]
        return nearest
    
    def _get_predicted_value(self, k_nearest):
        '''Takes in the indices of the k nearest points and returns the mean of their target values'''
        return np.mean(self.y_train[k_nearest])