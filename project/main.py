## YOUR CODE
## test
from src.train_detection import train_detection
from src.train_classification import train_classification
from src.predict_pipeline import run_pipeline

if __name__ == "__main__":
    train_detection()
    train_classification()
    run_pipeline()