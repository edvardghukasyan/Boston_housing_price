from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
data = load_boston()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data.data, data.target, test_size=0.15, random_state=42)
steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=78, fit_intercept=True))
]
model = Pipeline(steps)
model.fit(Xtrain, Ytrain)
import pickle
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
score = pickle_model.score(Xtest, Ytest)
print("Test score: {0:.2f} %".format(100 * score))
Ypredict = pickle_model.predict(Xtest)
tuple_objects = (model, Xtrain, Ytrain, score)
pickle.dump(tuple_objects, open("tuple_model.pkl", 'wb'))
import joblib
joblib_file = "joblib_model.pkl"
joblib.dump(model, joblib_file)
joblib_model = joblib.load(joblib_file)
score = joblib_model.score(Xtest, Ytest)
print("Test score: {0:.2f} %".format(100 * score))
Ypredict = pickle_model.predict(Xtest)


