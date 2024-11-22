import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def unit_step_func(x):
    return np.where(x > 0, 1, 0)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_func(linear_output)

dataset = pd.read_csv('valorant_match_clsfc.csv')

dataset['outcome'] = LabelEncoder().fit_transform(dataset['outcome'])

X = dataset[['round_wins', 'round_losses', 'kills', 'deaths', 'assists', 'kdr', 'avg_dmg', 'acs']]
y = dataset['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(X_train.values, y_train.values)

predictions = p.predict(X_test.values)
accuracy = np.sum(y_test.values == predictions) / len(y_test.values)
print("Perceptron Accuracy:", accuracy)

def plot_data_distribution(y_train, y_test):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.countplot(x=y_train, ax=ax[0], palette="pastel")
    ax[0].set_title("Training Data Distribution", fontsize=14, fontweight='bold')
    ax[0].set_xlabel("Outcome (0 = Loss, 1 = Win)", fontsize=12)
    ax[0].set_ylabel("Count", fontsize=12)
    
    for p in ax[0].patches:
        height = p.get_height()
        ax[0].text(p.get_x() + p.get_width() / 2., height + 3,
                   f'{int(height)}', ha="center", fontsize=10, color="black")
    
    sns.countplot(x=y_test, ax=ax[1], palette="pastel")
    ax[1].set_title("Test Data Distribution", fontsize=14, fontweight='bold')
    ax[1].set_xlabel("Outcome (0 = Loss, 1 = Win)", fontsize=12)
    ax[1].set_ylabel("Count", fontsize=12)
    
    for p in ax[1].patches:
        height = p.get_height()
        ax[1].text(p.get_x() + p.get_width() / 2., height + 3,
                   f'{int(height)}', ha="center", fontsize=10, color="black")
    
    sns.despine()  
    plt.tight_layout()
    plt.show()

def plot_accuracy(accuracy):
    plt.figure(figsize=(8, 5))
    plt.bar(['Test Accuracy'], [accuracy], color='#B3E5FC', edgecolor='black', width=0.4)
    plt.title("Model Accuracy on Test Data", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1)
    
    plt.text(0, accuracy + 0.05, f'{accuracy*100:.2f}%', ha='center', fontsize=12, color="black")
    
    sns.despine()  
    plt.show()
    
plot_data_distribution(y_train, y_test)
plot_accuracy(accuracy)
