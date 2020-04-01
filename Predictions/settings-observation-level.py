# Modify this file as needed
import os

from sklearn.ensemble import RandomForestRegressor

from src.configuration.settings_template import Settings, SettingsEnumOptions

# Settings relating to the data file used for input and columns
# ----------------------------------------------------------------------------------------------------------------------
# Path to the data file for input

to_predict = "Authenticity" # change this for each specific variable to predict
            #"IsQuestion"
            #"IsInstructionalUtterance"
            #"IsEvaluationElaborated"
            #"IsGoalSpecified"
            #"Uptake"
            #"CogLevel"
            #"IsDisciplinaryTermsPresent"

data_dir = "Jensen-CHI-20/Predictions/"

#These data files are not provided, contact us with questions!
Settings.IO.DATA_INPUT_FILE = os.path.join(data_dir, to_predict+"by_obs_data.csv")

# Folder to output results in, helpful to change if you don't want to overwrite previous results
Settings.IO.RESULTS_OUTPUT_FOLDER = "Jensen-CHI-20/Predictions/obsvervation-level-predictions/"+to_predict

# Names of columns in spreadsheet to identify what should be the input data, and what should be the predicted labels
Settings.COLUMNS.IDENTIFIER = "ObsID"

Settings.COLUMNS.Y_LABELS_TO_PREDICT = [
    "AvgCombinedVarInObs"
]

Settings.FEATURE_INPUT_SOURCES_TO_RUN = [
      SettingsEnumOptions.LanguageFeatureInput.
          with_language_from_column("UtterancesInObs")
]

Settings.PREDICTION = SettingsEnumOptions.Prediction.REGRESSION


# Settings relating to the models to be run and the parameters to be cross validated
# ----------------------------------------------------------------------------------------------------------------------
# Classes of the models to be run

Settings.MODELS_TO_RUN = [RandomForestRegressor]

Settings.CROSS_VALIDATION.NUM_TRAIN_TEST_FOLDS = 5  # Number of folds to use in outer loop to split all data into train / test
Settings.CROSS_VALIDATION.NUM_CV_TRAIN_VAL_FOLDS = 3  # Number of folds to use in nested cross validation to split train data into train / validation

#Settings.CROSS_VALIDATION.SCORING_FUNCTION = 'average_precision'

Settings.CROSS_VALIDATION.HYPER_PARAMS.VECTORIZER = {
            'min_df': [0.01], #, 0.04, 0.05, 0.06],
            'stop_words': [None, 'english'],
            'use_stemming': [False], # True], #, False],
            'min_pmi': [2.0],
            'ngram_range': [(1, 3)] #,(1, 1), (1, 2), (2, 2)] #, (1,3), (2,3)]})
        }


# Add cross validation parameters for each model to run
Settings.CROSS_VALIDATION.HYPER_PARAMS.MODEL = {
    RandomForestRegressor.__name__: {
        # Name of the key must be the name of the parameter to be passed into the constructor of the model
        'n_estimators': [100]
    }
}

