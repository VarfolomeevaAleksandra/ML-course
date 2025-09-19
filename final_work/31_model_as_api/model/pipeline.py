import datetime

import json

import dill

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def delete_outliers(df):
    df_copy = df.copy()

    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)

        return boundaries

    boundaries = calculate_outliers(df_copy['year'])
    df_copy.loc[df_copy['year'] < boundaries[0], 'year'] = round(boundaries[0])
    df_copy.loc[df_copy['year'] > boundaries[1], 'year'] = round(boundaries[1])
    return df_copy


def short_model(df):
    df_copy = df.copy()

    def cut_model(x):
        if not pd.isna(x):
            return x.lower().split(' ')[0]
        else:
            return x

    df_copy.loc[:, 'short_model'] = df_copy['model'].apply(cut_model)
    return df_copy


def age_category(df):
    df_copy = df.copy()
    df_copy.loc[:, 'age_category'] = df_copy['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return df_copy


def filter_data(df):

    columns_to_drop = [
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    return df.drop(columns_to_drop, axis=1)


def main():
    df = pd.read_csv('data/homework.csv').drop('id', axis=1)

    X = df.drop('price_category', axis=1)
    y = df['price_category'].apply(lambda x: 2.0 if x == 'high' else (1.0 if x == 'medium' else 0.0))

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('ejection', FunctionTransformer(delete_outliers)),
        ('short_model', FunctionTransformer(short_model)),
        ('age_category', FunctionTransformer(age_category))
    ])

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns


    transformer = make_column_transformer(
        (Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), make_column_selector(dtype_include=['int64', 'float64'])),

        (Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), make_column_selector(dtype_include=object))
    )


    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        MLPClassifier(activation='logistic', hidden_layer_sizes=(256, 128, 64), max_iter=800)
    )
    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('transformer', transformer),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')

    best_pipe.fit(X, y)



    with open('car_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Car prediction model',
                'author': 'Aleksandra Varfolomeeva',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'accuracy': best_score
            }
        }, file)

if __name__ == '__main__':
    main()

