from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import pickle

# データの読み込み関数
#def load_data():
#    df = pd.read_csv("data/Titanic.csv")
#    X = df.drop("Survived", axis=1)
#    y = df["Survived"]
#    return train_test_split(X, y, test_size=0.2, random_state=42)
def load_data():
    # ファイルの絶対パスを取得（train_model.pyと同階層のdataフォルダを参照）
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "data", "Titanic.csv")

    df = pd.read_csv(csv_path)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# モデル学習関数
def train_model(X_train, y_train):
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    clf.fit(X_train, y_train)
    return clf

# 実行スクリプト部分（学習＆保存）
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    with open("models/titanic_model.pkl", "wb") as f:
        pickle.dump(model, f)

    baseline_path = "models/titanic_model_baseline.pkl"
    if not os.path.exists(baseline_path):
        with open(baseline_path, "wb") as f:
            pickle.dump(model, f)

    print("✅ モデルとベースラインを保存しました")
