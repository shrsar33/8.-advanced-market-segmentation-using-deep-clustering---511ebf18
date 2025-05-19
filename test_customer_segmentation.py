import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

class TestCustomerSegmentation(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'Age': [25, 34, 45, 52, 23],
            'Income': [50000, 60000, 75000, 80000, 45000],
            'Gender': ['Male', 'Female', 'Female', 'Male', 'Female']
        })

    def test_fill_missing_values(self):
        df = self.df.copy()
        df.loc[1, 'Income'] = np.nan
        df.fillna(df.median(numeric_only=True), inplace=True)
        self.assertFalse(df.isnull().any().any(), "Missing values not filled correctly")

    def test_categorical_encoding(self):
        df = pd.get_dummies(self.df, columns=['Gender'], drop_first=True)
        self.assertIn('Gender_Male', df.columns)

    def test_standard_scaling(self):
        df = pd.get_dummies(self.df, columns=['Gender'], drop_first=True)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
        self.assertEqual(scaled.shape, df.shape)

    def test_pca_components(self):
        df = pd.get_dummies(self.df, columns=['Gender'], drop_first=True)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled)
        self.assertEqual(pca_result.shape[1], 2)

    def test_autoencoder_output(self):
        input_dim = 3
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(2, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        data = np.random.rand(10, input_dim)
        output = autoencoder.predict(data)
        self.assertEqual(output.shape, data.shape)

    def test_kmeans_clustering(self):
        data = np.random.rand(10, 3)
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(data)
        self.assertEqual(len(kmeans.labels_), 10)

if __name__ == '__main__':
    unittest.main()