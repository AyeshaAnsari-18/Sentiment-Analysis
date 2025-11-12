from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": LinearSVC()
    }

    for name, model in models.items():
        print(f"\n{name} Results:")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(classification_report(y_test, preds))
