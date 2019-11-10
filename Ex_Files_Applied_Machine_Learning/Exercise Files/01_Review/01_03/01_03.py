import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
model = LogisticRegression().fit(iris['data'], iris['target'])
print(model.predict(iris['data']))

