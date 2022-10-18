import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier


class Encoder(BaseEstimator, TransformerMixin):
    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        code = {False: 0, True: 1, '55 Cancri e': 0, 'TRAPPIST-1e': 1, 'PSO J318.5-22': 2,
                'Europa': 0, 'Earth': 1, 'Mars': 2, "S": 0, "P": 1, "B": 0, "F": 1, "A": 2, "G": 3, "E": 4,
                "D": 5, "C": 6, "T": 7}
        return X.replace(code)


class Imputer(BaseEstimator, TransformerMixin):
    def fit(self, X, Y=None):
        return self

    def inferring_missing_values_HP(self, row, dataframe):
        dataframe_valid_HP_group = dataframe[
            (dataframe["PassengerGroupId"] == row["PassengerGroupId"]) & (np.isnan(dataframe["HomePlanet"]) is False)]
        if dataframe_valid_HP_group.shape[0] >= 1:
            return dataframe_valid_HP_group.iloc[0, 'HomePlanet']
        else:
            return row['HomePlanet']

    def inferring_missing_values_Age(self, row):
        if row["VIP"]:
            return 37
        else:
            return 27

    def transform(self, X, Y=None):
        X.loc[:, "HomePlanet"] = X.apply(
            lambda x: self.inferring_missing_values_HP(x, X) if np.isnan(x["HomePlanet"]) else x["HomePlanet"], axis=1)
        X.loc[:, 'Age'] = X.apply(lambda x: self.inferring_missing_values_Age(x) if np.isnan(x["Age"]) else x["Age"],
                                  axis=1)
        return X.fillna(-1)


class FeatureDiscovery(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["PassengerGroupId"] = [float(passenger_id.split('_')[0]) for passenger_id in X["PassengerId"]]
        X["PassengerIdInsideGroup"] = [float((passenger_id).split('_')[1]) for passenger_id in
                                       X["PassengerId"]]
        X["PassengerGroupSize"] = X.groupby(['PassengerGroupId']).transform('count')["PassengerId"]
        X["CabinDeck"] = [cabin.split('/')[0] if len(str(cabin).split('/')) > 1 else np.nan for cabin in
                          X["Cabin"]]
        X["CabinNum"] = [float(cabin.split('/')[1]) if len(str(cabin).split('/')) > 1 else np.nan for cabin in
                         X["Cabin"]]
        X["CabinSide"] = [cabin.split('/')[2] if len(str(cabin).split('/')) > 1 else np.nan for cabin in
                          X["Cabin"]]
        X["FamilyName"] = [name.split(' ')[1] if len(str(name).split(' ')) > 1 else np.nan for name in
                           X["Name"]]
        X_features_to_use = X.drop(columns=["PassengerId", "Name",
                                            "FamilyName", "Cabin"])
        return X_features_to_use


class FeatureSelect(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[['CryoSleep', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                  "Age", 'CabinDeck', 'CabinNum', 'HomePlanet', "CabinSide", 'Destination']]


class PreprocessorFeature:
    def __init__(self):
        self.model = make_pipeline(FeatureDiscovery(), Encoder(), Imputer(), FeatureSelect())

    def transform_train_set(self, X_sample, y=None):
        dataframe_transformed = self.model.fit_transform(X_sample)
        return dataframe_transformed

    def transform_test_set(self, X_sample, y=None):
        dataframe_transformed = self.model.transform(X_sample)
        return dataframe_transformed


class PreprocessorTarget:
    def __init__(self):
        pass

    def encode(self, target_array):
        target_array_encoded = [1 if element is True else 0 for element in target_array]
        return target_array_encoded

    def decode(self, target_array):
        target_array_decoded = [True if element == 1 else False for element in target_array]
        return target_array_decoded


def evaluation(model_prediction, training_sample, target_array, test_sample):
    model_prediction.fit(training_sample, target_array)
    return model_prediction.predict(test_sample)


pd.set_option("display.max_column", 17)
pd.set_option("display.max_row", 17)
df = pd.read_csv("train.csv")
df_copy_train = df.copy()
df_test = pd.read_csv("test.csv")
df_copy_test = df_test.copy()

# Phase de prétraitement des données
preprocessor_features = PreprocessorFeature()
X_train = df_copy_train.drop(columns=['Transported'])
y_train = df_copy_train['Transported']
preprocessor_target = PreprocessorTarget()
y_train_transformed = preprocessor_target.encode(y_train)
X_train_transformed = preprocessor_features.transform_train_set(X_train, y_train_transformed)
X_test_transformed = preprocessor_features.transform_test_set(df_copy_test)

# Construction du modèle de prédiction
model = make_pipeline(GradientBoostingClassifier(n_estimators=88, random_state=0))


# Evaluation du modèle et enregistrement des résultats dans un fichier submission.csv
pred_list = evaluation(model, X_train_transformed, y_train_transformed, X_test_transformed)
print(pred_list)
# df_submission = df_test[['PassengerId']]
# df_submission['Transported']= preprocessor_target.decode(pred_list)
# df_submission.to_csv('submission.csv', index=False)
