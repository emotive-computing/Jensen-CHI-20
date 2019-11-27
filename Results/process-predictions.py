# %% Import packages and define useful stuff
import pandas as pd
import numpy as np
import math
from scipy.stats import zscore, spearmanr, t
from sklearn.preprocessing import MinMaxScaler

# Calculates Spearman correlation between labels and predictions
# Input: test labels, predictions
# Output: Spearman correlation
def evaluateSpearman(y_test, predictions):
    rho, _ = spearmanr(y_test, predictions)
    return rho

# Teacher talk features
variables_labels = {
    "IsQuestion": ["0", "Question"],
    "IsDisciplinaryTermsPresent": ["0", "Yes"],
    "IsInstructionalUtterance": ["0", "1"],
    "IsEvaluationElaborated": ["0", "Elaborated"],
    "Authenticity": ["0", "Authentic Question"],
    "CogLevel": ["0", "High"],
    "Uptake": ["0", "Uptake/Genuine Uptake"],
    "IsGoalSpecified": ["0", "1"],
}


# Directories for utterance- and observation-level predictions
utt_level_pred_dir = 'C:/Users/emje6419/Dropbox (Emotive Computing)/Emily CETD/utterance-results/predictions/RandomForestClassifier/'
obs_level_pred_dir = 'C:/Users/emje6419/Dropbox (Emotive Computing)/Emily CETD/Cathlyn_EDM/obs_true_results_'

target_file = 'C:/Users/emje6419/Dropbox (Emotive Computing)/Emily CETD/Deliverables/CHI 2020/Online Results/processed-predictions.csv'
results_dir = 'C:/Users/emje6419/Dropbox (Emotive Computing)/Emily CETD/Deliverables/CHI 2020/Online Results/'

# %% Create long format data of predicted and observed proportion of each feature

first = True

# Calculate per teacher talk feature
for variable in variables_labels:
    print(variable)
    label = variables_labels[variable][1]
    outdf = pd.DataFrame()
    
    # Get utterance-level prediction data
    utt_level_preds = utt_level_pred_dir + variable + '-CombineLanguageAndRegularFeatureInput-predictions.csv'
    data = pd.read_csv(utt_level_preds)
    
    ## Split Observation ID and Utterance ID
    data['ObsID'], data['uttid'] = data['ObsID_uttid'].str.rsplit(pat='_',n=1).str
    
    ## Average at observation level to get predicted proportion
    obs = data.groupby('ObsID')['uttid'].count()
    if label == '1':
        data['masked'] = data['Predicted_value'].mask(data['Predicted_value'].astype(str).ne(label))
    else:
        data['masked'] = data['Predicted_value'].mask(data['Predicted_value'].ne(label.lower()))
    
    utt_raw = data.groupby('ObsID')['masked'].count()/obs    
    outdf['pred_proportion'] = utt_raw
    
    ## Average at observation level to get true proportion
    if label == '1':
        data['masked'] = data['Predicted_value'].mask(data['True_value'].astype(str).ne(label))
    else:
        data['masked'] = data['Predicted_value'].mask(data['True_value'].ne(label.lower()))
        
    human_raw = data.groupby('ObsID')['masked'].count()/obs
    outdf['true_proportion'] = human_raw
    
    outdf['variable'] = variable + '_utt_raw'
    if first:
        outdf.to_csv(target_file)
        first = False
    else:
        outdf.to_csv(target_file, mode='a', header=False)
        
    allind = outdf.index
    
    ## Scale predictions to human range
    scaler = MinMaxScaler(feature_range = (np.min(outdf['true_proportion']),np.max(outdf['true_proportion'])))
    utt_scaled = scaler.fit_transform(utt_raw.values.reshape(-1,1)).reshape(-1)
    outdf['pred_proportion'] = utt_scaled
    outdf['true_proportion'] = human_raw
    outdf['variable'] = variable + '_utt_scaled'
    outdf.to_csv(target_file, header=False, mode='a')
    
    # Get observation-level prediction data
    obs_level_preds = obs_level_pred_dir + variable + '/predictions/RandomForestRegressor/AvgCombinedVarInObs-LanguageFeatureInput-predictions.csv'
    data = pd.read_csv(obs_level_preds).sort_values(by=['ObsID'])
    
    ## Get raw predictions
    obs_raw = data['Predicted_value']
    outdf['pred_proportion'] = obs_raw.values
    outdf['true_proportion'] = human_raw
    outdf['variable'] = variable + '_obs_raw'
    outdf.to_csv(target_file, header=False, mode='a')
    
    ## Scale to human range
    obs_scaled = scaler.fit_transform(obs_raw.values.reshape(-1,1)).reshape(-1)
    outdf['pred_proportion'] = obs_scaled
    outdf['true_proportion'] = human_raw
    outdf['variable'] = variable + '_obs_scaled'
    outdf.to_csv(target_file, header=False, mode='a')
    
    # Calculate mean of utterance- and observation-level scaled predictions
    outdf['pred_proportion'] = np.mean(np.array([utt_scaled,obs_scaled]), axis=0)
    outdf['true_proportion'] = human_raw
    outdf['variable'] = variable + '_mean_raw'
    outdf.to_csv(target_file, header=False, mode='a')
    
    ## Scale to human range
    outdf['pred_proportion'] = scaler.fit_transform(outdf['pred_proportion'].values.reshape(-1,1)).reshape(-1)
    outdf['variable'] = variable + '_mean_scaled'
    outdf.to_csv(target_file, header=False, mode='a')
        
#%% Calculate best models and output summary table for each talk feature

data = pd.read_csv(target_file)
models = ['_utt_scaled','_obs_scaled','_mean_scaled']
model_names = ['Utterance-level','Observation-level','Combined']
values = ['mean','spearman_r','mean_abs_error','confidence_interval']
interval = 0.95

index = list(variables_labels)
columns = ['best_model','computer_mean','spearman_r','mean_abs_error','confidence_interval','human_mean']
temp = np.full((len(index),len(columns)),np.nan,dtype=np.object)
df_big = pd.DataFrame(data=temp,index=index,columns=columns)

first = True

# for each talk feature
for variable in variables_labels:
    print(variable)
    temp = np.full((len(models)+1,len(values)), np.nan, dtype=np.object)
    df = pd.DataFrame(data=temp,index=models+['human'],columns=values)
    df_error = pd.DataFrame()
	# for each potential model (calculated above)
    for model in models:
        # Filter data to get desired predictions
        preds = data.loc[data['variable'] == variable + model,['pred_proportion']]
        true = data.loc[data['variable'] == variable + model,['true_proportion']]
        
        # save some summary statistics and performance results
        df.loc[model,['mean']] = np.mean(preds).values # predicted distribution mean
        df.loc[model,['spearman_r']] = evaluateSpearman(preds,true) # spearman correlation
        abs_error = np.absolute(preds.values - true.values) 
        df.loc[model,['mean_abs_error']] = np.mean(abs_error) #mean absolute error
        n = abs_error.size
        stdev = np.std(abs_error)
        test_stat = t.ppf((interval + 1)/2, n)
        s = "({:.3f}, {:.3f})".format((np.mean(abs_error) - test_stat * stdev / math.sqrt(n)),(np.mean(abs_error) + test_stat * stdev / math.sqrt(n)))
        df.loc[model,['confidence_interval']] = s # confidence interval of MAE
            
        if model == '_utt_scaled':
            df.loc['human',['mean']] = np.mean(true).values # human distribution mean
    
    df.to_csv(results_dir+variable+'_summary.csv')
    
    # Save best results to a big table
    df = pd.read_csv(results_dir+variable+'_summary.csv',index_col=0)
    best_model = df['spearman_r'].idxmax()
    best_model_name = model_names[models.index(best_model)]
    df_big.loc[variable,['best_model']] = best_model_name
    df_big.loc[variable,['computer_mean','spearman_r','mean_abs_error','confidence_interval']] = df.loc[best_model,['mean','spearman_r','mean_abs_error','confidence_interval']].values
  
    df_big.loc[variable,['human_mean']] = df.loc['human','mean']
    
    # Save absolute errors for best models into a separate file for future analysis
    filtered = data.loc[data['variable'] == variable + best_model]
    df_error['ObsID'] = filtered['ObsID']
    df_error['variable'] = variable
    preds = filtered['pred_proportion']
    true = filtered['true_proportion']
    df_error['absolute_error'] = np.absolute(preds.values - true.values)
    if first:
        df_error.to_csv(results_dir+'abs-error.csv', index=False)
        first = False
    else:
        df_error.to_csv(results_dir+'abs-error.csv', index=False, mode='a', header=False)
    
df_big.to_csv(results_dir+'all_var_summary.csv')