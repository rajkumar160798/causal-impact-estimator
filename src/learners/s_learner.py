from econml.metalearners import SLearner
from sklearn.ensemble import GradientBoostingRegressor

def train_s_learner(X_train, T_train, y_train):
    model = GradientBoostingRegressor()
    s_learner = SLearner(overall_model=model)
    s_learner.fit(y_train, T_train, X=X_train)
    return s_learner

