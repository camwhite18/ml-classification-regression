from sklearn.neural_network import MLPRegressor


def train_gwp_mlp():
    return MLPRegressor(activation='relu', solver='adam', alpha=0.05, hidden_layer_sizes=(50, 100, 50),
                        learning_rate='adaptive')