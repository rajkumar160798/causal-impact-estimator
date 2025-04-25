from econml.metalearners import TLearner
from sklearn.ensemble import GradientBoostingRegressor

def train_t_learner(X_train, T_train, y_train):
    model = GradientBoostingRegressor()
    t_learner = TLearner(overall_model=model)
    t_learner.fit(y_train, T_train, X=X_train)
    return t_learner
