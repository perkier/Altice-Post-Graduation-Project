import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, KFold, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


def see_all():
    # Alongate the view on DataFrames

    pd.set_option('display.max_rows', 10000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)

def splitting(df):

    split = StratifiedShuffleSplit(n_splits=1, test_size= 0.3, random_state=42)

    for train_index, test_index in split.split(df, df["TuneTime_PeriodofDay"]):

        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

    return strat_train_set, strat_test_set


def define_trainig_set(df, label):

    strat_train_set, strat_test_set = splitting(df)

    savings_train = strat_train_set.drop(label, axis=1)
    savings_labels_train = strat_train_set[label].copy()

    savings_test = strat_test_set.drop(label, axis=1)
    savings_labels_test = strat_test_set[label].copy()

    X = df.drop(label, axis=1)                       #Features
    y = df.loc[:,label].copy()                       #Labels

    X_train = savings_train
    y_train = savings_labels_train
    X_test = savings_test
    y_test = savings_labels_test

    return X_train, y_train, X_test, y_test


def feature_scaling(df):

    df_np = df.to_numpy()

    mean_df = df.describe().loc['mean']
    mean_np = mean_df.to_numpy()

    max_df = df.describe().loc['max']
    max_np = max_df.to_numpy()

    min_df = df.describe().loc['min']
    min_np = min_df.to_numpy()

    max_np = max_np - min_np

    scaling_vector = (df_np - mean_np) / max_np

    scaled_array = {'instructions': '(df_np - mean_np) / max_np; T_c, T_g, spindle_pos', 'mean_np': mean_np, 'max_np': max_np}

    df = pd.DataFrame(data= scaling_vector,
                      index = np.array([i for i in range(len(scaling_vector))]),
                      columns = list(df.columns.values))

    descaled_df = pd.DataFrame(data=[mean_np, max_np],
                              index=np.array([i for i in range(2)]),
                              columns=list(df.columns.values))

    return df, scaled_array, descaled_df


def plotting(df, x, y):

    # df.plot(kind='scatter', x='x', y='y', linestyle='--', marker='o')

    fig, ax = plt.subplots()

    ax.scatter(x=df[x], y=df[y], linestyle='--', marker='o')

    plt.title(f'{x} vs {y}')

    plt.show()



def ML_Playground(X_train, X_test, y_train, y_test, test_set, scaled_array):

    # clf = GradientBoostingRegressor(n_estimators=1,
    #                                 loss='ls', alpha=0.95, max_depth=3,
    #                                 learning_rate=.1, min_samples_leaf=9,
    #                                 min_samples_split=9)

    clf = RandomForestRegressor()
    # clf = GradientBoostingRegressor(n_estimators=75)

    clf.fit(X_train, y_train)

    test_y_predictions = clf.predict(test_set)

    min_mnse = 999
    mnse = mean_squared_error(y_test, test_y_predictions)
    i = 0

    if mnse < min_mnse:

        min_mnse = mnse

        print(f'Trial {i}')
        print('MNSE: ', mnse)
        print('Mean Abs Error: ', mean_absolute_error(y_test, test_y_predictions))
        print('\n')
        print('-' * 20)
        print('-' * 20)
        print('\n')

    Tc = {}
    Tg = {}
    spindle_mm = {}
    j = 0

    for T_c in range(26, 55):
        Tc[j] = T_c
        Tg[j] = 70
        spindle_mm[j] = 8

        j += 1

    print('\n')

    test_set = pd.DataFrame({'T_c': Tc,
                             'T_g': Tg,
                             'spindle_pos': spindle_mm})

    # print(scaled_array)
    # quit()

    test_scaled = (test_set - scaled_array['mean_np']) / scaled_array['max_np']

    test_y_predictions = clf.predict(test_scaled)

    ER = {}

    for i in range(len(test_y_predictions)):
        ER[i] = test_y_predictions[i]

    prediction_set = pd.DataFrame({'T_c': Tc,
                                   'T_g': Tg,
                                   'spindle_pos': spindle_mm,
                                   'ER': ER})

    # predict_critical_temp(prediction_set)

    print(prediction_set)
    plotting(prediction_set, 'T_c', 'ER')

    quit()


def main():

    df = pd.read_csv("/mnt/home/a20201828/Documents/DATA/MEGA_TABLE_30dias_CLASSIFICATION.csv", sep=";")

    df = df.dropna().reset_index(drop=True)

    print(df.head())

    # df = pd.read_csv("/mnt/home/a20201828/Desktop/Share/DATA/MEGA_TABLE_30dias_v3.csv", sep=";")

    df_train = df.loc[df["Viewers_Time_Since_Start"] <= 82]
    df_test = df.loc[df["Viewers_Time_Since_Start"] > 82]

    columns_to_class = ["Viu_Tudo", "TuneTime_PeriodofDay", "TuneTime_WeekDay", "Show_Duration",
                        "filmes","sé?ries","Crianç?as_em_Casa","O_Gamer","Donas_de_Casa","Musica_Non_Stop",
                        "Desporto_outros","Desporto_Futebol","O_Intelectual","Para_Teenager","Actualidade_Noticias",
                        "Tardes_Fim_de_Semana","Manhas_Fim_de_Semana","Influencers","Tudo_Resto"]

    cat_columns = ["filmes", "sé?ries", "Crianç?as_em_Casa", "O_Gamer", "Donas_de_Casa", "Musica_Non_Stop",
                   "Desporto_outros", "Desporto_Futebol", "O_Intelectual", "Para_Teenager", "Actualidade_Noticias",
                   "Tardes_Fim_de_Semana", "Manhas_Fim_de_Semana", "Influencers", "Tudo_Resto"]

    df_train = df_train[columns_to_class]
    # df_train[cat_columns] = df_train[cat_columns].astype('int32')

    df_test = df_test[columns_to_class]
    # df_test[cat_columns] = df_test[cat_columns].astype('int32')

    rs = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    i = 0

    label = "Viu_Tudo"

    train_X = df_train.drop(label, axis=1)                       #Features
    train_y = df_train.loc[:,label].copy()                       #Labels

    test_X = df_test.drop(label, axis=1)                       #Features
    test_y = df_test.loc[:,label].copy()                       #Labels

    max_depth = [25]
    learning_rate = [0.1, 0.2, 0.3, 0.4]

    # Minimum number of samples required to split a node
    min_samples_split = [5]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2]

    depth = 25
    leaf_size = 2
    estimator = 5

    # clf = KNeighborsClassifier(n_neighbors=3)

    clf = RandomForestClassifier(min_samples_split=estimator, min_samples_leaf=leaf_size, max_depth=depth, random_state=42)

    clf.fit(train_X, train_y)

    # test_y_predictions = clf.predict(test_X)
    test_y_predictions = clf.predict_proba(test_X)

    prediction_comparison = pd.DataFrame(
        {"Label": test_y.reset_index(drop=True), "Prediction": test_y_predictions[:, 0]})

    prediction_comparison.to_csv(
        f"/mnt/home/a20201828/Documents/DATA/FINAL_CLASS/Classification_FINAL.csv", sep=';', index=False)


    quit()

    X_train, y_train, X_test, y_test = define_trainig_set(df, "TuneTime_PeriodofDay")

    # Filter methods e wrapper methods


if __name__ == '__main__':

    see_all()
    main()
