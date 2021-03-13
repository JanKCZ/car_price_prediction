
from sklearn.compose import ColumnTransformer
import sklearn
from sklearn.neural_network import MLPRegressor


class nnModel(MLPRegressor):
    def __init__(self, hidden_size, n_layers, learning_rate, alpha, solver, activation, dropout = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.solver = solver
        self.activation = activation
    
    # def train_model(self, X_train_final, y_train):
    #     model = MLPRegressor(solver=solver, alpha=0.001, learning_rate_init=learning_rate,
    #                     hidden_layer_sizes=(n_layers, hidden_size), random_state=42,
    #                     batch_size= 32, verbose = True, max_iter = 500,
    #                     learning_rate = 'adaptive', warm_start=True,
    #                     validation_fraction = 0.1, early_stopping = False, activation=activation)
    #     model.fit(X_train_final, y_train.values)
        
    #     return model, model.validation_scores_
        
    
