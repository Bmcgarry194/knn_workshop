import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_classification, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def generate_moons_df(n_samples=50, noise=.6):
    X, y = make_moons(n_samples=n_samples, noise=noise)
    
    df = pd.DataFrame(X, columns=['A', 'B'])
    df['target'] = y
    
    return df

def preprocess(df):
    
    X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'])

    ss = StandardScaler()

    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)

    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)

    X_test_scaled['target'] = y_test.values

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    X_train_scaled['target'] = y_train.values
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def plot_boundaries(model, X_test, X_train, ax, padding = 1, grid_granularity = 0.01, show_test=False, plot_probas=True):

    x_min, x_max = X_train['A'].min() - padding, X_train['A'].max() + padding
    y_min, y_max = X_train['B'].min() - padding, X_train['B'].max() + padding

    xs = np.arange(x_min, x_max, grid_granularity)
    ys = np.arange(y_min, y_max, grid_granularity)

    xx, yy = np.meshgrid(xs, ys)
    
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
    if not plot_probas:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy , Z, cmap='PRGn', levels=20, alpha=.5)

    train_positives = X_train[X_train['target'] == 1]
    train_negatives = X_train[X_train['target'] == 0]

    ax.scatter(train_positives['A'], train_positives['B'], color='forestgreen', edgecolors='lightgreen')
    ax.scatter(train_negatives['A'], train_negatives['B'], color='purple', edgecolors='plum')

    if show_test:
        test_positives = X_test[X_test['target'] == 1]
        test_negatives = X_test[X_test['target'] == 0]

        ax.scatter(test_positives['A'], test_positives['B'], color='forestgreen', edgecolors='black')
        ax.scatter(test_negatives['A'], test_negatives['B'], color='purple', edgecolors='black')
    
