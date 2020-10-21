import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')




# Function to map string to float
def cp_mapping(train, test):
    train_, test_ = train.copy(), test.copy()
    cp_type = {'trt_cp': 1., 'ctl_vehicle': 0.}
    cp_dose = {'D1': 1., 'D2': 0.}
    for df in [train_, test_]:
        df['cp_type'] = df['cp_type'].map(cp_type)
        df['cp_dose'] = df['cp_dose'].map(cp_dose)
    return train_, test_

# Filter out control group
def cp_filter(train, train_targets, test):
    train_targets_ = train_targets[train['cp_type'] == 1]
    train_ = train[train['cp_type'] == 1].drop('cp_type', axis=1)
    test_ = test[test['cp_type'] == 1].drop('cp_type', axis=1)
    return train_, train_targets_, test_
    
# Function to scale our data
def scaling(train, test):
    scaler = RobustScaler()
    scaler.fit(pd.concat([train, test], axis = 0))
    train_ = pd.DataFrame(scaler.transform(train), index=train.index, columns=train.columns)
    test_ = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)
    return train_, test_

# Rank Gauss : transform each column to follow a normal distribution 
def rankgauss(train, test):
    GENES = [col for col in train.columns if col.startswith('g-')]
    CELLS = [col for col in train.columns if col.startswith('c-')]
    for col in (GENES + CELLS):
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
        n_sample = len(train[col].values)
        n_sample_sub = len(test[col].values)
        raw_vec = train[col].values.reshape(n_sample, 1)
        raw_vec_sub = test[col].values.reshape(n_sample_sub, 1)
        transformer.fit(raw_vec)
        train[col]=transformer.transform(raw_vec).reshape(1, n_sample)[0]
        test[col] = transformer.transform(raw_vec_sub).reshape(1, n_sample_sub)[0]
    return train, test

# Function to extract common stats features
def fe_stats(train, test):
    GENES = [col for col in train.columns if col.startswith('g-')]
    CELLS = [col for col in train.columns if col.startswith('c-')]
    for df in [train, test]:
        df['g_sum'] = df[GENES].sum(axis = 1)
        df['g_mean'] = df[GENES].mean(axis = 1)
        df['g_std'] = df[GENES].std(axis = 1)
        df['g_kurt'] = df[GENES].kurtosis(axis = 1)
        df['g_skew'] = df[GENES].skew(axis = 1)
        df['c_sum'] = df[CELLS].sum(axis = 1)
        df['c_mean'] = df[CELLS].mean(axis = 1)
        df['c_std'] = df[CELLS].std(axis = 1)
        df['c_kurt'] = df[CELLS].kurtosis(axis = 1)
        df['c_skew'] = df[CELLS].skew(axis = 1)
        df['gc_sum'] = df[GENES + CELLS].sum(axis = 1)
        df['gc_mean'] = df[GENES + CELLS].mean(axis = 1)
        df['gc_std'] = df[GENES + CELLS].std(axis = 1)
        df['gc_kurt'] = df[GENES + CELLS].kurtosis(axis = 1)
        df['gc_skew'] = df[GENES + CELLS].skew(axis = 1)
    return train, test

def c_squared(train, test):
    CELLS = [col for col in train.columns if col.startswith('c-')]
    for df in [train, test]:
        for feature in CELLS:
            df[f'squared_{feature}'] = df[feature] ** 2
    return train, test

# Function to extract pca features
def fe_pca(train, test, n_components_g = 70, n_components_c = 10, SEED = 123):
    def create_pca(train, test, kind = 'g', n_components = n_components_g):
        features = [col for col in train.columns if col.startswith(f"{kind}-")]
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        pca = PCA(n_components = n_components,  random_state = SEED)
        data = pca.fit_transform(data)
        columns = [f'pca_{kind}{i + 1}' for i in range(n_components)]
        data = pd.DataFrame(data, columns = columns)
        train_ = data.iloc[:train.shape[0]]; train_.index = train.index
        test_ = data.iloc[train.shape[0]:]; test_.index = test.index
        train = pd.concat([train, train_], axis = 1)
        test = pd.concat([test, test_], axis = 1)
        return train, test
    train, test = create_pca(train, test, kind = 'g', n_components = n_components_g)
    train, test = create_pca(train, test, kind = 'c', n_components = n_components_c)
    return train, test

# Function to filter out features with low variance
def variance_thresh(train, test, threshold=0.8, start_idx=2):
    var_thresh = VarianceThreshold(threshold)
    data = train.append(test)
    data_transformed = var_thresh.fit_transform(data.iloc[:,start_idx:])
    train_transformed = pd.DataFrame(data_transformed[:train.shape[0]], index=train.index)
    test_transformed = pd.DataFrame(data_transformed[train.shape[0]:], index=test.index)

    train = pd.concat((train.iloc[:,:start_idx], train_transformed), axis=1)
    test = pd.concat((test.iloc[:,:start_idx], test_transformed), axis=1)
    return train, test