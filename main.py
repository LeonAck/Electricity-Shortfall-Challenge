from data_loading import load_data
from data_pipeline import split_data, train_and_evaluate_model, choose_best_model, train_full_model_predict_test_set
from models import get_models
from preprocessing import preprocess_data
import pandas as pd
from plots import plot_predictions


def main():

    target_column = 'load_shortfall_3h'
    print("Data laden...")
    train_df, test_df, sample_submission = load_data()

    print("Data preprocessen...")
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    
    # Load models
    models_to_try = get_models()

    # Model selection
    best_rmse, best_model = choose_best_model(train_df, models_to_try)


    # Train on full training set and predict on test set
    test_predictions = train_full_model_predict_test_set(best_model, train_df, test_df, target_column=target_column)
    print(test_predictions)
    
    # Output
    submission_df = pd.DataFrame({
        'time': test_df.index,
        'load_shortfall_3h': test_predictions
    })
    submission_df.to_csv('sample_submission.csv', index=False)
    print("\nVoorspellingen opgeslagen in 'sample_submission.csv'")

if __name__ == "__main__":
    main()