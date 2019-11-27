# Modify this file as needed
import os

from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.configuration.settings_template import Settings, SettingsEnumOptions
# Settings relating to the data file used for input and columns
# ----------------------------------------------------------------------------------------------------------------------
# Path to the data file for input

from src.pipeline.resampling import DatasetSampler

label = "IsQuestion"

data_dir = "C:/Users/emje6419/Dropbox (Emotive Computing)/Emily CETD/Cathlyn_EDM/common-models/"
output_dir = "C:/Users/emje6419/Dropbox (Emotive Computing)/Emily CETD/"

Settings.IO.DATA_INPUT_FILE = os.path.join(data_dir, "all_combined_total_features_renamed_spring_only.csv")
# Folder to output results in
Settings.IO.RESULTS_OUTPUT_FOLDER = os.path.join(output_dir, "utterance-results/"+label)

# Names of columns in spreadsheet to identify what should be the input data, and what should be the predicted labels
Settings.COLUMNS.IDENTIFIER = "ObsID_uttid"

Settings.COLUMNS.Y_LABELS_TO_PREDICT = [
    label
    ]


Settings.FEATURE_INPUT_SOURCES_TO_RUN = [
      SettingsEnumOptions.CombineLanguageAndRegularFeatureInput.
          with_regular_features_from_file(os.path.join(data_dir, "all_feature_names.txt")).
          with_language_from_column("TranscribedUtterance")

      # Or for just language...
      # SettingsEnumOptions.LanguageFeatureInput.
      #   with_language_from_column("TranscribedUtterance")
]

Settings.PREDICTION = SettingsEnumOptions.Prediction.CLASSIFICATION


# Group all data instances with the same TeacherID in the same fold
# Data with same teacher cannot appear in both train and test fold
Settings.COLUMNS.GROUP_BY_COLUMN = "TeacherID"

Settings.COLUMNS.ORDER_IN_GROUPS_BY_COLUMN = 'ObsID'
Settings.COLUMNS.ORDER_IN_GROUPS_SORT_BY_COLUMN = 'UtteranceID'

# OPTIONAL
# Declare the labels in columns in order to specify which label is viewed as negative or positive
# So here, Statement would be the negative label (index 0 in array), Question would be the positive label (index 1)
# For multiclass, labels are assigned numeric values in order of array
Settings.COLUMNS.MAKE_ALL_LABELS_BINARY = True # Labels which are not 1 become 0
Settings.COLUMNS.LABEL_DECLARATIONS = {
    "IsTeacherSource": ["0", "Teacher"],
    "IsQuestion": ["0", "Question"],

    "IsInstructionalStatement": ["0", "Instructional"],
    "IsDisciplinaryStatement": ["0", "Disciplinary"],
    "IsDisciplinaryTermsPresent": ["0", "Yes"],
    "IsInstructionalQuestion": ["0", "Instructional"],
    "IsDisciplinaryQuestion": ["0", "Disciplinary"],
    "IsInstructionalUtterance": ["0", "1"],
    "IsDisciplinaryUtterance": ["0", "1"],

    "IsEvaluationFollowupIncluded": ["0", "Yes"],
    "IsEvaluationElaborated": ["0", "Elaborated"],
    "IsEvaluationValencePositive": ["0", "Positive"],

    "CombinedAuthCogUptake": ["0", "1"],
    "Authenticity": ["0", "Authentic Question"],
    "CogLevel": ["0", "High"],
    "Uptake": ["0", "Uptake/Genuine Uptake"],

    "IsSerialQuestion": ["0", "Yes"],
    "IsGoalSpecified": ["0", "1"],
    "IsStudentResponsePresent": ["0", "Yes"],
}

# ----------------------------------------------------------------------------------------------------------------------


# Settings relating to the models to be run and the parameters to be cross validated
# ----------------------------------------------------------------------------------------------------------------------
# Classes of the models to be run

Settings.MODELS_TO_RUN = [RandomForestClassifier]

Settings.CROSS_VALIDATION.NUM_TRAIN_TEST_FOLDS = 5  # Number of folds to use in outer loop to split all data into train / test
Settings.CROSS_VALIDATION.NUM_CV_TRAIN_VAL_FOLDS = 3  # Number of folds to use in nested cross validation to split train data into train / validation

Settings.CROSS_VALIDATION.SCORING_FUNCTION = 'average_precision'

Settings.CROSS_VALIDATION.HYPER_PARAMS.VECTORIZER = {
            'min_df': [0.01, 0.02, 0.03],
            'stop_words': [None],
            'use_stemming': [False],
            'min_pmi': [2.0], 
            'ngram_range': [(1, 3)]
        }

# # Parameters specifying which type of rebalancing to use on dataset
Settings.CROSS_VALIDATION.HYPER_PARAMS.RESAMPLER = [
     (RandomUnderSampler, {})
]

Settings.CROSS_VALIDATION.HYPER_PARAMS.FEATURE_SCALER = [
    (StandardScaler, {'with_mean': [False]})
]

# Add cross validation parameters for each model to run
Settings.CROSS_VALIDATION.HYPER_PARAMS.MODEL = {
    RandomForestClassifier.__name__: {
        # Name of the key must be the name of the parameter to be passed into the constructor of the model
        'n_estimators': [100]
    }
}

