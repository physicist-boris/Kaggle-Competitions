import pandas as pd
import functools
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin
from lightgbm import LGBMRegressor


class FillingMissingCatValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.fillna('NAN')


class FillingMissingQuatValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        values = {}
        for index in X.index:
            if pd.isnull(X.loc[index, 'GarageYrBlt']):
                X.loc[index, 'GarageYrBlt'] = X.loc[index, 'YearBuilt']
        sum_nan_series = X.select_dtypes(exclude='object').isna().sum()
        for col in sum_nan_series[sum_nan_series != 0].index:
            if col == 'LotFrontage':
                values[col] = X[col].median()
            elif col == 'MasVnrArea':
                values[col] = X[col].median()
            else:
                values[col] = -1
        return X.fillna(value=values)


class Encoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.enc.fit(X.select_dtypes(include='object'))
        return self

    def transform(self, X, y=None):
        X_object_transformed = self.enc.transform(X.select_dtypes(include='object'))
        columns_object = list(X.select_dtypes(include='object').columns)
        df_object_type_transformed = pd.DataFrame(X_object_transformed, columns=columns_object)
        X.loc[:, columns_object] = df_object_type_transformed
        return X


class FeatureCreation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        bool_year_liste = [X.loc[index, 'YearBuilt'] == X.loc[index, 'YearRemodAdd'] for index in X.index]
        X['HouseRemodelling'] = [0 if bool_year else 1 for bool_year in bool_year_liste]
        X['NbrPorch'] = 0
        for index in X.index:
            porch_nbr = 0
            for col in ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']:
                if X.loc[index, col] != 0:
                    porch_nbr += 1
            X.loc[index, 'NbrPorch'] = porch_nbr
        X['GotPool'] = [0 if pd.isnull(cond_pool) else 1 for cond_pool in X['PoolQC']]
        X['GotWoodDeck'] = [0 if size_wood == 0 else 1 for size_wood in X['WoodDeckSF']]
        X['GotBsmt'] = [0 if pd.isnull(cond_bsmt) else 1 for cond_bsmt in X['BsmtCond']]
        X['GotFireplace'] = [0 if pd.isnull(cond_fireplace) else 1 for cond_fireplace in X['FireplaceQu']]
        return X


class PreprocessorFeature:
    def __init__(self):
        score_func_infos = functools.partial(mutual_info_classif, random_state=0)
        self.model = make_pipeline(FillingMissingQuatValues(), FillingMissingCatValues(), FeatureCreation(), Encoder(),
                                   SelectKBest(score_func=score_func_infos, k=86))

    def transform_train_set(self, X_sample, y=None):
        X_sample_transformed = self.model.fit_transform(X_sample, y)
        # print(self.model.named_steps['selectkbest'].get_feature_names_out())
        return X_sample_transformed

    def transform_test_set(self, X_sample, y=None):
        X_sample_transformed = self.model.transform(X_sample)
        return X_sample_transformed


def evaluation(model_prediction, training_sample, target_y, test_sample):
    model_prediction.fit(training_sample, target_y)
    return model_prediction.predict(test_sample)


pd.set_option('display.max_row', 81)
pd.set_option('display.max_column', 81)
df_train = pd.read_csv('Train_House_Prices.csv')
df_copy_train = df_train.copy()
df_test = pd.read_csv('Test_House_Prices.csv')
df_copy_test = df_test.copy()

# Phase de prétraitement des données
X_train = df_copy_train.drop(columns=['SalePrice'])
y_train = df_copy_train['SalePrice']
preprocessor_feature = PreprocessorFeature()
X_train_transformed = preprocessor_feature.transform_train_set(X_train, y_train)
X_test_transformed = preprocessor_feature.transform_test_set(df_copy_test)

# Construction d'un modèle de Prédiction
model = LGBMRegressor(objective='regression',
                      num_leaves=4,
                      learning_rate=0.01,
                      n_estimators=5000,
                      max_bin=200,
                      bagging_fraction=0.75,
                      bagging_freq=5,
                      bagging_seed=7,
                      feature_fraction=0.2,
                      feature_fraction_seed=7,
                      verbose=-1,
                      )

# Evaluation du modèle et enregistrement des résultats dans un fichier submission.csv
pred_list = evaluation(model, X_train_transformed, y_train, X_test_transformed)
print(pred_list)
# df_submission = df_test[['Id']]
# df_submission['SalePrice'] = pred_list
# df_submission.to_csv('submission.csv', index=False)