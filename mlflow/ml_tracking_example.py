import mlflow

def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("mlflow-test")

    with mlflow.start_run():
        mlflow.log_metric("p", 0)
        mlflow.log_metric("m", 1)

if __name__ == "__main__":
    main()
