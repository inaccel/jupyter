import argparse
import xgboost as xgb

from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from timeit import default_timer as timestamp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha',
                        type = float,
                        default = 0.0,
                        help = 'L1 regularization term on weights.')
    parser.add_argument('--eta',
                        type = float,
                        default = 0.3,
                        help = 'Step size shrinkage used in update to prevent overfitting.')
    parser.add_argument('--max-depth',
                        type = int,
                        default = 6,
                        help = 'Maximum depth of a tree.')
    parser.add_argument('--name',
                        type = str,
                        default = 'CIFAR_10',
                        choices = ['CIFAR_10', 'Fashion-MNIST', 'mnist_784', 'SVHN'],
                        help = 'String identifier of the dataset.')
    parser.add_argument('--num-boost-round',
                        type = int,
                        default = 10,
                        help = 'Number of boosting iterations.')
    parser.add_argument('--subsample',
                        type = float,
                        default = 1.0,
                        help = 'Subsample ratio of the training instances.')
    parser.add_argument('--test-size',
                        type = float,
                        default = 0.25,
                        help = 'The proportion of the dataset to include in the test split.')
    parser.add_argument('--tree-method',
                        type = str,
                        default = 'auto',
                        choices = ['auto', 'exact', 'approx', 'hist', 'gpu_hist', 'fpga_exact'],
                        help = 'The tree construction algorithm used in XGBoost.')
    args = parser.parse_args()

    X, y = fetch_openml(args.name, return_X_y = True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = args.test_size)

    features = Normalizer()
    X_train = features.fit_transform(X_train)
    X_test = features.transform(X_test)

    label = LabelEncoder()
    y_train = label.fit_transform(y_train)
    y_test = label.transform(y_test)

    params = {
        'alpha': args.alpha,
        'eta': args.eta,
        'max_depth': args.max_depth,
        'num_class': len(label.classes_),
        'objective': 'multi:softmax',
        'subsample': args.subsample,
        'tree_method': args.tree_method
    }

    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)

    start = timestamp()
    model = xgb.train(params, dtrain, args.num_boost_round)
    stop = timestamp()

    print('time=%.3f' % (stop - start))

    predictions = model.predict(dtest)

    print('accuracy=%.3f' % accuracy_score(y_test, predictions))
