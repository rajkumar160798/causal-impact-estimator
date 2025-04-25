from econml.metalearners import XLearner
from sklearn.ensemble import GradientBoostingRegressor

def train_x_learner(X_train, T_train, y_train):
    model = GradientBoostingRegressor()
    x_learner = XLearner(overall_model=model)
    x_learner.fit(y_train, T_train, X=X_train)
    return x_learner
