import pandas as pd
import numpy as np
import re
import regex


import dill
import datetime


from typing import List, Dict, Optional
from googletrans import Translator

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score

from lightgbm import LGBMClassifier


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            mode_cols: Optional[List[str]] = None,
            unknown_cols: Optional[List[str]] = None,
            unknown_value: str = 'unknown'
    ):
        self.mode_cols = mode_cols or []
        self.unknown_cols = unknown_cols or []
        self.unknown_value = unknown_value

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.modes_ = {}
        for col in self.mode_cols:
            self.modes_[col] = X[col].mode()[0]
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame, y = None) -> pd.DataFrame:
        X = X.copy()
        if list(X.columns) != self.feature_names_in_:
            raise ValueError("Колонки не совпадают с обучающей выборкой")

        for col in self.mode_cols:
            X[col] = X[col].fillna(self.modes_[col])
        for col in self.unknown_cols:
            X[col] = X[col].fillna(self.unknown_value)

        return X


class OSByBrandFiller(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            brand_col: str = 'device_brand',
            os_col: str = 'device_os',
            device_category_col: str = 'device_category',
            ios_brand: str = 'Apple',
            macintosh_brand: str = 'Apple',
            #android_brands: Optional[List[str]] = None,
            unknown_value: str = 'unknown'
    ):
        self.brand_col = brand_col
        self.os_col = os_col
        self.device_category_col = device_category_col
        self.ios_brand = ios_brand
        self.macintosh_brand = macintosh_brand
        #self.android_brands = android_brands or []
        self.unknown_value = unknown_value

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.android_only_brands_ = list(
            X.groupby(self.brand_col)[self.os_col]
            .apply(lambda x: set(x.dropna().unique()))
            .loc[lambda s: s == {"Android"}]
            .index
        )
        self.brand_os_counts_ = {}
        for brand, group in X.groupby(self.brand_col):
            counts = group[self.os_col].value_counts(dropna=True)

            if len(counts) == 0:
                continue  # если у бренда только NaN → ничего не делаем

            # самая частая ОС
            top_os = counts.index[0]
            top_ratio = counts.iloc[0] / counts.sum()
            self.brand_os_counts_[brand] = [top_os, top_ratio]
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        if list(X.columns) != self.feature_names_in_:
            raise ValueError("Колонки не совпадают с обучающей выборкой")

        X.loc[
            (X[self.brand_col] == self.ios_brand) &
            (X[self.device_category_col] == "mobile") &
            (X[self.os_col].isna()), self.os_col] = "iOS"

        X.loc[
            (X[self.brand_col] == self.ios_brand) &
            (X[self.device_category_col] == "desktop") &
            (X[self.os_col].isna()), self.os_col] = "Macintosh"

        for i in self.android_only_brands_:
            X.loc[(X[self.brand_col] == i) & (X[self.os_col].isna()), self.os_col] = "Android"

        for brand, group in X.groupby(self.brand_col):
            if brand in self.brand_os_counts_.keys() and self.brand_os_counts_[brand][1] >= 0.95:
                # заполняем NaN этой ОС
                X.loc[(X[self.brand_col] == brand) & (X[self.os_col].isna()), self.os_col] = self.brand_os_counts_[brand][0]

        return X


class GroupFeatures(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            cols_to_group: Optional[Dict[str, int]] = None,
            rare_value: str = 'rare'
    ):
        self.cols_to_group = cols_to_group or {}
        self.rare_value = rare_value

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for col, threshold in self.cols_to_group.items():
            counts = X[col].value_counts()
            rare_categories = counts[counts <= threshold].index
            X[col] = X[col].replace(rare_categories, self.rare_value)
        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            drop_original: bool = True
    ):
        self.drop_original = drop_original
        self.freq_maps_ = {}
        self.cols_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self.cols_ = X.columns
        for col in self.cols_:
            self.freq_maps_[col] = X[col].value_counts(normalize=True)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols_:
            X[f"{col}_freq"] = X[col].map(self.freq_maps_[col])
        if self.drop_original:
            X = X.drop(columns=self.cols_)
        return X


class DeleteOutliers(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            size_col: str = "screen_size",
            group_col: str = "device_category",
            sigma: float = 3.0
    ):
        self.size_col = size_col
        self.group_col = group_col
        self.sigma = sigma

    def fit(self, X: pd.DataFrame, y=None):
        self.mode_ = {}
        self.upper_bound_ = {}
        self.lower_bound_ = {}
        for cat in X[self.group_col].unique():
            mask = X[self.group_col] == cat
            values = X.loc[mask, self.size_col]

            mean = values.mean()
            std = values.std()

            self.lower_bound_[cat] = mean - self.sigma * std
            self.upper_bound_[cat] = mean + self.sigma * std

            # Мода без выбросов
            valid_mask = (values >= self.lower_bound_[cat]) & (values <= self.upper_bound_[cat])
            self.mode_[cat] = values[valid_mask].mode()[0]

        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        if list(X.columns) != self.feature_names_in_:
            raise ValueError("Колонки не совпадают с обучающей выборкой")

        for cat in X[self.group_col].unique():
            mask = X[self.group_col] == cat
            values = X.loc[mask, self.size_col]
            outliers = (values < self.lower_bound_[cat]) | (values > self.upper_bound_[cat])
            X.loc[mask & outliers, self.size_col] = self.mode_[cat]

        return X


def target(df: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df_target = df_target.copy()
    target_actions = [
        'sub_car_claim_click',
        'sub_car_claim_submit_click',
        'sub_open_dialog_click',
        'sub_custom_question_submit_click',
        'sub_call_number_click',
        'sub_callback_submit_click',
        'sub_submit_success',
        'sub_car_request_submit_click'
    ]

    df_target["event_value"] = df_target["event_action"].isin(target_actions).astype(int)
    target = df_target.groupby("session_id")["event_value"].max().reset_index()
    df = df.merge(target, on="session_id", how="left").fillna({"event_value": 0})
    return df


def pre_drop(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_predrop = ['session_id', 'client_id', 'visit_number', 'visit_date', 'visit_time']
    df = df.drop(cols_to_predrop, axis=1)

    return df


def not_set(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy.replace(["(not set)"], np.nan, inplace = True)
    return df_copy


def standard_city(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    translator = Translator()

    # Словарь сокращений
    abbr_dict = {
        "st.": "saint",
        "st": "saint",
        "mt.": "mount",
        "mt": "mount",
        "ft.": "fort",
        "ft": "fort",
        "v.": "villa",
        "v": "villa",
        "nbhd.": "neighborhood",
        "nbhd": "neighborhood",
        "pt.": "point",
        "pt": "point",
        "n.": "north",
        "n": "north",
        "s.": "south",
        "s": "south",
        "e.": "east",
        "e": "east",
        "w.": "west",
        "w": "west"
    }
    def clean_city(city):
        if not isinstance(city, str):
            return "unknown"
        # Убираем спецсимволы типа \u200e
        city = regex.sub(r'\p{C}', '', city)

        # Если строка состоит только из цифр → unknown
        if city.strip().isdigit():
            return "unknown"

        # Убираем скобки и содержимое внутри
        city = re.sub(r"\s*\(.*?\)", "", city)

        # Если несколько слов через запятую, оставить только первое
        city = city.split(",")[0].strip()

        # Замена сокращений через словарь
        words = city.split()
        words = [abbr_dict.get(w.lower(), w) for w in words]
        city = " ".join(words)

        # Если есть кириллица, перевод через Google Translate
        if re.search(r"[а-яА-Я]", city):
            try:
                city = translator.translate(city, src='ru', dest='en').text
            except:
                pass  # если перевод не удался, оставляем оригинал

        # Убираем лишние пробелы
        city = city.strip()

        return city

    df_copy['geo_city'] = df_copy['geo_city'].apply(clean_city)

    return df_copy


def standard_browser(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    def clean_browser(value):
        if isinstance(value, str):
            val = value.lower()
            # Если строка в формате com.something.something, оставляем только третье слово
            match = re.match(r'^com\.[^.]+\.(.+)$', val)
            if match:
                val = match.group(1)
            # Для остальных многословных названий оставляем только первое слово
            else:
                val = val.split()[0]
            # удаляем содержимое скобок
            val = re.sub(r'\s*\(.*?\)', '', val)
            # удаляем цифры и версии
            val = re.sub(r'[\d\.\-_:\[\]]+', '', val)
            val = val.strip()
            return val

    df_copy['device_browser'] = df_copy['device_browser'].apply(clean_browser)
    return df_copy


def screen_size(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy.loc[:,'screen_size'] = df_copy.device_screen_resolution.apply(lambda x: float(x.split('x')[0]) * float(x.split('x')[1]))
    return df_copy


def os_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.device_os = df.device_os.str.split(' ').str[0]
    return df


def delete_space(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        # Заменяем пробелы на _
        df[col] = df[col].str.replace(' ', '_')
    return df


def social_ad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[:, 'social_ad'] = df['utm_source'].apply(lambda x: 1 if x in ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt',
                                                                  'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHNBGo',
                                                                  'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm']
    else 0)
    return df


def organic_traffic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[:,'organic_traffic'] = df['utm_medium'].apply(lambda x: 1 if x in ['organic', 'referral', '(none)'] else 0)
    return df


def low_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].str.lower().str.strip()
    return df


def drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_drop = ['device_screen_resolution']
    df = df.drop(cols_to_drop, axis = 1)

    return df



def main():
    df = pd.read_csv('data/ga_sessions.csv')
    df_hits = pd.read_csv("data/ga_hits-001.csv", usecols=["session_id", "event_action"])

    df = target(df, df_hits)
    df = pre_drop(df)

    X = df.drop('event_value', axis = 1)
    y = df['event_value']
    preprocessor = Pipeline(steps=[
        ("replace_notset", FunctionTransformer(func=not_set)),
        ("os_fill", OSByBrandFiller()),
        ("city_clean", FunctionTransformer(func=standard_city)),
        ("browser_clean", FunctionTransformer(func=standard_browser)),
        ("os_split", FunctionTransformer(func=os_split)),
        ("delete_space", FunctionTransformer(func=delete_space)),
        ("custom_imputer", CustomImputer(mode_cols=['utm_medium','geo_country','utm_source',
                                                    'device_browser','device_screen_resolution'],
                                         unknown_cols=['utm_keyword','device_os','device_brand',
                                                       'utm_adcontent','utm_campaign','geo_city'])),
        ("screen_size", FunctionTransformer(func=screen_size)),
        ("delete_outliers", DeleteOutliers()),
        ("social_ad", FunctionTransformer(func=social_ad)),
        ("traffic", FunctionTransformer(func=organic_traffic)),
        ("lowercase", FunctionTransformer(func=low_str)),
        ("rare_categories", GroupFeatures(cols_to_group={'geo_country':500, 'device_browser':1000, 'utm_medium':500})),
        ("drop", FunctionTransformer(func=drop_cols))
    ])
    transformer = make_column_transformer(
        (StandardScaler(), ['screen_size']),
        (FrequencyEncoder(),
         ['utm_source', 'utm_campaign', 'utm_adcontent', 'utm_keyword', 'device_brand', 'geo_city']
         ),
        (OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['utm_medium','device_category', 'device_os',
                           'device_browser','geo_country'])
    )
    lgbm_params = {
        "num_leaves": 83,
        "max_depth": -1,
        "learning_rate": 0.013174061343601481,
        "n_estimators": 310,
        "feature_fraction": 0.6665254574295094,
        "bagging_fraction": 0.9561231697771445,
        "bagging_freq": 9,
        "reg_alpha": 0.12760369609247932,
        "reg_lambda": 0.28562639958443636,
        "min_child_samples": 43,
        "random_state": 42,
        "n_jobs": -1
    }
    lgb_model = LGBMClassifier(**lgbm_params)
    pipe_lgb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('transformer', transformer),
        ('classifier', lgb_model)
    ])
    score = cross_val_score(pipe_lgb, X, y, cv=4, scoring='roc_auc')
    print(f'model: {type(pipe_lgb.named_steps["classifier"]).__name__}, roc_auc: {score.mean():.4f}')

    pipe_lgb.fit(X, y)

    with open('sberauto_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': pipe_lgb,
            'metadata': {
                'name': 'Auto rent prediction model',
                'author': 'Aleksandra Varfolomeeva',
                'version': 1,
                'date': datetime.datetime.now().isoformat(),
                'type': type(pipe_lgb.named_steps["classifier"]).__name__,
                'roc-auc': score.mean()
            }
        }, file,recurse=True)

if __name__ == '__main__':
    main()