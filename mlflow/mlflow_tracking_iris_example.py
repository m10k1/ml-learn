## Step3
import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# データセットの読み込み
X, y = datasets.load_iris(return_X_y=True, as_frame=True)

# 訓練データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# モデルのハイパーパラメータの定義
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# 訓練
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# テストデータに対して推論を実行
y_pred = lr.predict(X_test)

# メトリクスの計算
accuracy = accuracy_score(y_test, y_pred)

## Step4

# トラッキングサーバーのuriを設定
mlflow.set_tracking_uri(uri="http://localhost:5000")

# 新しいMLflow実験を作成
mlflow.set_experiment("MLflow Quickstart")

# MLflow のrunを開始
with mlflow.start_run():
    # ハイパーパラメータをログ
    mlflow.log_params(params)

    # Lossをログ
    mlflow.log_metric("accuracy", accuracy)

    # タグを設定
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # モデルシグネチャを
    signature = infer_signature(X_train, lr.predict(X_train))

    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )

# Step5
# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

print(result[:4])