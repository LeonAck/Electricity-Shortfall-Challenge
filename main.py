from data_loading import load_data
from data_pipeline import split_data, train_and_evaluate_model, choose_best_model, train_full_model_predict_test_set
from models import get_models
from preprocessing import preprocess_data, TimeAwareKNNImputer, create_preprocessing_pipeline
import pandas as pd
from plots import plot_predictions

imputer = TimeAwareKNNImputer(n_neighbors=5)
submit = False
def main():

    target_column = 'load_shortfall_3h'
    print("Data laden...")
    train_df, test_df, sample_submission = load_data()

    print("Data preprocessen...")
    # Create preprocessing pipeline with your preferred imputation method
    pipeline = create_preprocessing_pipeline(
        imputer=imputer,  # or 'knn' or 'pattern'
        freq='3h',
        fill_method='interpolate'
    )

    # Fit the pipeline on training data
    pipeline.fit(train_df)

    # Transform both training and test data
    train_processed = pipeline.transform(train_df)
    test_processed = pipeline.transform(test_df)
    
    # Load models
    models_to_try = get_models()

    # Model selection
    best_rmse, best_model = choose_best_model(train_processed, models_to_try)


    # Train on full training set and predict on test set
    test_predictions = train_full_model_predict_test_set(best_model, train_processed, test_processed, target_column=target_column)
    
    plot_predictions(test_predictions)
    if submit:
        # Output
        submission_df = pd.DataFrame({
            'time': test_processed.index,
            'load_shortfall_3h': test_predictions
        })
        submission_df.to_csv('sample_submission.csv', index=False)
        print("\nVoorspellingen opgeslagen in 'sample_submission.csv'")

if __name__ == "__main__":
    main()