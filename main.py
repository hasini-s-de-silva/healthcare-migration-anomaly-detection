from src.simulate_batches import main as simulate_batches
from src.validate_data import main as validate_data
from src.build_features import main as build_features
from src.train_model import main as train_model
from src.evaluate_model import main as evaluate_model
from src.report_results import main as report_results


def main():
    print("Starting Healthcare Migration Anomaly Detection Pipeline...\n")

    print("Step 1: Simulating migration batches...")
    simulate_batches()

    print("\nStep 2: Validating data quality...")
    validate_data()

    print("\nStep 3: Building model features...")
    build_features()

    print("\nStep 4: Training anomaly detection model...")
    train_model()

    print("\nStep 5: Evaluating model...")
    evaluate_model()

    print("\nStep 6: Generating charts...")
    report_results()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()