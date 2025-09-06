import sklearn as sk
import sklearn.model_selection as skms
import sklearn.linear_model as sklm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import scipy as sp
import seaborn as sb
import torch_geometric.explain
import Portfolio_Centrality_Data as pcd
import Database as db
import torch
import torch_geometric
import networkx as nx
import shap
import Charts
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models.tgn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
from typing import Dict, Tuple


def lin_regression(df: pd.DataFrame):
    x = df.drop(columns=['SPX win percentage', 'average return'])
    y1, y2 = df['SPX win percentage'], df['average return']
    
    x_train, x_test, y1_train, y1_test = skms.train_test_split(x, y1, test_size=0.3, random_state=1)
    x_train, x_test, y2_train, y2_test = skms.train_test_split(x, y2, test_size=0.3, random_state=1)
    
    model1 = sklm.LinearRegression()
    model1.fit(x_train, y1_train)
    model2 = sklm.LinearRegression()
    model2.fit(x_train, y2_train)

    y1_prediction = model1.predict(x_test)
    print('SPX Win % R^2 Score: ', sk.metrics.r2_score(y1_test, y1_prediction))
    print('SPX Win % MSE: ', sk.metrics.mean_squared_error(y1_test, y1_prediction))
    y2_prediction = model2.predict(x_test)
    print('Average Return R^2 Score: ', sk.metrics.r2_score(y2_test, y2_prediction))
    print('Average Return MSE: ', sk.metrics.mean_squared_error(y2_test, y2_prediction))

    # Feature importance
    feature_importance_win_percentage = pd.Series(model1.coef_, index=x.columns).sort_values(ascending=False)
    print(f'Feature importance for SPX win \%: {feature_importance_win_percentage}')
    feature_importance2_average_return = pd.Series(model2.coef_, index=x.columns).sort_values(ascending=False)
    print(f'Feature importance for average return: {feature_importance2_average_return}')

    # Plot SPX Win % Importances
    plt.figure(figsize=(10, 6))
    feature_importance_win_percentage.sort_values().plot(kind='barh')
    plt.title('Feature Importance for SPX Win %')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    # Plot Average Return Importances
    plt.figure(figsize=(10, 6))
    feature_importance2_average_return.sort_values().plot(kind='barh')
    plt.title('Feature Importance for Average Return')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()



def logistic_regression(df: pd.DataFrame, test_size=0.3, random_state=1, plot_coefficients=True):
    x = df.drop(columns=['SPX win percentage', 'average return'])
    y1, y2 = df['SPX win percentage'], df['average return']
    y1 = (y1 > 0.5).astype(int)
    y2 = (y2 > 0).astype(int)

    # Split data
    x_train, x_test, y1_train, y1_test = skms.train_test_split(x, y1, test_size=test_size, random_state=random_state)
    x_train, x_test, y2_train, y2_test = skms.train_test_split(x, y2, test_size=test_size, random_state=random_state)

    # Initialize and fit model
    model1 = sklm.LogisticRegression(max_iter=1000)
    model1.fit(x_train, y1_train)
    model2 = sklm.LogisticRegression(max_iter=1000)
    model2.fit(x_train, y2_train)

    # Predict
    y1_pred = model1.predict(x_test)
    y2_pred = model2.predict(x_test)

    # Evaluation
    print('Accuracy:', sk.metrics.accuracy_score(y1_test, y1_pred))
    print('\nClassification Report:\n', sk.metrics.classification_report(y1_test, y1_pred))
    print('Confusion Matrix:\n', sk.metrics.confusion_matrix(y1_test, y1_pred))
    print('--------------------------------------------------------------')
    print('Accuracy:', sk.metrics.accuracy_score(y2_test, y2_pred))
    print('\nClassification Report:\n', sk.metrics.classification_report(y2_test, y2_pred))
    print('Confusion Matrix:\n', sk.metrics.confusion_matrix(y2_test, y2_pred))

    # Feature importance (coefficients)
    if plot_coefficients and hasattr(x, 'columns'):
        coeffs = pd.Series(model1.coef_[0], index=x.columns).sort_values()
        plt.figure(figsize=(10, 6))
        coeffs.plot(kind='barh')
        plt.title('Logistic Regression Coefficients (SPX Win Percentages)')
        plt.xlabel('Coefficient Value')
        plt.tight_layout()
        plt.show()
        coeffs = pd.Series(model2.coef_[0], index=x.columns).sort_values()
        plt.figure(figsize=(10, 6))
        coeffs.plot(kind='barh')
        plt.title('Logistic Regression Coefficients (Average Returns)')
        plt.xlabel('Coefficient Value')
        plt.tight_layout()
        plt.show()
    

def svm_classifier(df: pd.DataFrame, test_size=0.25, plot_cm=True):
    # Split features and target
    x = df.drop(columns=['SPX_win_percentage', 'average_return', 'portfolio', 'id', 'TMFG_vert_weight_sum', 'TMFG_edge_weight_sum', 'katz', 'degree', 'eigenvector', 'made_from_freq_edge']) # Remove unecessary features and TMFG-related features (b.c. TMFG-related features show up in the nodes/edges of dual TMFG features)
    feature_names = list(x.columns)
    scaler = sk.preprocessing.MinMaxScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    y1, y2 = df['SPX_win_percentage'], df['average_return']
    y1 = (y1 > 0.5).astype(int)
    y2 = (y2 > 0).astype(int)

    # Train/test split
    x1_train, x1_test, y1_train, y1_test = skms.train_test_split(x, y1, test_size=test_size, random_state=2, stratify=y1)
    x2_train, x2_test, y2_train, y2_test = skms.train_test_split(x, y2, test_size=test_size, random_state=2, stratify=y2)

    # Initialize and train SVM
    model1 = sk.svm.SVC(kernel='poly', C=1, coef0=0, degree=4, gamma=5, class_weight='balanced', probability=True)
    model1.fit(x1_train, y1_train)
    model2 = sk.svm.SVC(kernel='poly', C=0.01, coef0=-3, degree=5, gamma='scale', class_weight='balanced', probability=True)
    model2.fit(x2_train, y2_train)

    # Predict
    y1_pred = model1.predict(x1_test)
    y2_pred = model1.predict(x2_test)

    # Determine goal accuracies
    counter1 = 0
    for percentage in y1:
        if percentage > 0.5:
            counter1 += 1
    counter2 = 0
    for avg_ret in y2:
        if avg_ret > 0:
            counter2 += 1
    print(f'Goal accuracy for predicting which portfolios will outperform the S&P 500: {counter1/len(y1)}')
    print(f'Goal accuracy for predicting which portfolios will yield positive average returns: {counter2/len(y2)}')
    print('---------------------------------------------------------------------------------')


    # Evaluation
    print('Predicting which portfolios will likely outperform the S&P 500:')
    print('Accuracy:', sk.metrics.accuracy_score(y1_test, y1_pred))
    print('\nClassification Report:\n', sk.metrics.classification_report(y1_test, y1_pred))
    print('Confusion Matrix:\n', sk.metrics.confusion_matrix(y1_test, y1_pred))
    print('---------------------------------------------------------------------------------')
    print('Predicting which portfolios will likely yield positive average returns:')
    print('Accuracy:', sk.metrics.accuracy_score(y2_test, y2_pred))
    print('\nClassification Report:\n', sk.metrics.classification_report(y2_test, y2_pred))
    print('Confusion Matrix:\n', sk.metrics.confusion_matrix(y2_test, y2_pred))

    # -------------------------------------------------------------------------------------
    # Define parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 0.5, 1, 3, 5, 7, 10],
        'gamma': ['scale', 'auto', 0.1, 0.5, 1, 5, 10],
        'kernel': ['rbf'],
        'degree': [1],  
        'coef0': [0]
    }

    # Create the SVM and perform Grid Search
    svc = sk.svm.SVC()
    grid_search1 = skms.GridSearchCV(svc, param_grid, cv=3, scoring='accuracy')
    grid_search2 = skms.GridSearchCV(svc, param_grid, cv=3, scoring='accuracy')
    grid_search1.fit(x, y1)
    grid_search2.fit(x, y2)

    # Best parameters and score
    print('Best Parameters (S&P 500):', grid_search1.best_params_)
    print('Best Cross-Validation Accuracy (S&P 500):', grid_search1.best_score_)
    print('Best Parameters (Avg returns):', grid_search2.best_params_)
    print('Best Cross-Validation Accuracy (Avg returns):', grid_search2.best_score_)

    # ------------------------------------------------------------------------------------

    # Make confusion matrix
    if plot_cm:
        cm = sk.metrics.confusion_matrix(y1_test, y1_pred)
        sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
        cm = sk.metrics.confusion_matrix(y2_test, y2_pred)
        sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()

     # ---------- SHAP explanations ----------
    shap_out1 = shap_out2 = None
    shap_background_size = 200
    shap_sample_size = 800
    # Helper to build SHAP on a fitted pipeline
    def _explain(best_model, Xtr, Xte, label, feature_names):
        rng = np.random.RandomState(1)

        # Background from TRAIN (for Kernel SHAP expectation)
        bg_idx = rng.choice(len(Xtr), size=min(shap_background_size, len(Xtr)), replace=False)
        X_bg = pd.DataFrame(Xtr.iloc[bg_idx].values, columns=feature_names)

        # Sample subset of TEST to explain
        ex_idx = rng.choice(len(Xte), size=min(shap_sample_size, len(Xte)), replace=False)
        X_explain = pd.DataFrame(Xte.iloc[ex_idx].values, columns=feature_names)

        # Wrapper returns positive-class probability
        f = lambda data: best_model.predict_proba(pd.DataFrame(data, columns=feature_names))[:, 1]

        explainer = shap.Explainer(f, X_bg, nsamples=4096)  # KernelExplainer under the hood
        shap_values = explainer(X_explain)

        # Make nice labels
        feature_labels = {
            'current_flow': 'Current Flow Betweenness Centrality',
            'incindent_edge_weight_sum': 'Incindent Edge Weight Sum',
            'closeness': 'Closeness Centrality',
            'harmonic': 'Harmonic Centrality',
            'pagerank': 'Pagerank',
            'betweenness': 'Betweenness Centrality',
            'katz': 'Katz Centrality',
            'degree': 'Degree Centrality',
            'eigenvector': 'Eigenvector Centrality',
            'communicability_betweenness': 'Communicability Betweenness Centrality',
            'load': 'Load Centrality',
            'second_order': 'Second Order Centrality',
            'vert_weight': 'Node Weight',
            'percolation': 'Percolation Centrality'
        }

        feature_names = [feature_labels.get(f, f) for f in feature_names]
        my_cmap = LinearSegmentedColormap.from_list('my_cmap', ["#00ade6", "#f30000"])

        # Global summary (beeswarm)
        shap.summary_plot(shap_values, X_explain, cmap=my_cmap, feature_names=feature_names, show=False, color_bar_label="Feature Value")
        plt.title(f"SHAP Summary")
        plt.xlabel("SHAP Value (Impact On Model Output)")
        plt.savefig("shap.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Top-3 dependence plots
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        top_idx = np.argsort(-mean_abs)[:3]
        for i in top_idx:
            shap.dependence_plot(
                feature_names[i], shap_values.values, X_explain,
                feature_names=feature_names, interaction_index=None, show=True
            )
            plt.title(f"SHAP dependence: {label} — {feature_names[i]}")
            plt.show()

        return {"explainer": explainer, "values": shap_values, "X_explain": X_explain}

    shap_out1 = _explain(model1, x1_train, x1_test, "SPX outperformance (y1)", feature_names)
    # shap_out2 = _explain(model2, x2_train, x2_test, "Avg return > 0 (y2)", feature_names)

    return model1, model2



def svm_optimizer(df_list1, df_list2, test_size=0.25):
    '''
    Determines 2 sets of hyperparameters that produce the highest average accuracy across all dataframes in df_list1 and applies those 2 sets to df_list2 and analyzed their performance.
    '''
    
    counter = 0
    df_parameters_and_accuracies1 = {}
    df_parameters_and_accuracies2 = {}
    for df in df_list1:
        counter += 1
        # Split features and target
        x = df.drop(columns=['SPX_win_percentage', 'average_return', 'portfolio', 'id', 'TMFG_vert_weight_sum', 'TMFG_edge_weight_sum', 'katz', 'degree', 'eigenvector'])
        scaler = sk.preprocessing.MinMaxScaler()
        x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
        y1, y2 = df['SPX win percentage'], df['average return']
        y1 = (y1 > 0.5).astype(int)
        y2 = (y2 > 0).astype(int)

        # Train/test split
        x1_train, x1_test, y1_train, y1_test = skms.train_test_split(x, y1, test_size=test_size, random_state=2, stratify=y1)
        x2_train, x2_test, y2_train, y2_test = skms.train_test_split(x, y2, test_size=test_size, random_state=2, stratify=y2)

        # Initialize and train SVM
        model1 = sk.svm.SVC(kernel='poly', C=1, coef0=0, degree=4, gamma=5, class_weight='balanced', probability=True)
        model1.fit(x1_train, y1_train)
        model2 = sk.svm.SVC(kernel='poly', C=0.01, coef0=-3, degree=5, gamma='scale', class_weight='balanced', probability=True)
        model2.fit(x2_train, y2_train)

        # Predict
        y1_pred = model1.predict(x1_test)
        y2_pred = model1.predict(x2_test)

        # Define hyperparameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 0.5, 1, 3, 5, 7, 10],
            'gamma': ['scale', 'auto', 0.1, 0.5, 1, 5, 10],
            'kernel': ['poly'],
            'degree': [3, 4, 5],  
            'coef0': [-4, -3, -2, -1, 0, 1, 2, 3, 5]
        }

        # Create the SVM and perform grid search
        precision_for_successful_portfolios = sk.metrics.make_scorer(sk.metrics.precision_score, pos_label=1, average='binary')
        svc = sk.svm.SVC()
        grid_search1 = skms.GridSearchCV(svc, param_grid, cv=3, scoring=precision_for_successful_portfolios)
        grid_search2 = skms.GridSearchCV(svc, param_grid, cv=3, scoring=precision_for_successful_portfolios)
        grid_search1.fit(x, y1)
        grid_search2.fit(x, y2)

        # Store accuracy of each hyperparameter set
        grid_seach1_info = grid_search1.cv_results_
        grid_search1_params = grid_seach1_info['params']
        grid_search1_avg_test_scores = grid_seach1_info['mean_test_score']
        df_parameters_and_accuracies1[counter] = (grid_search1_params, grid_search1_avg_test_scores)
        grid_seach2_info = grid_search2.cv_results_
        grid_search2_params = grid_seach2_info['params']
        grid_search2_avg_test_scores = grid_seach2_info['mean_test_score']
        df_parameters_and_accuracies2[counter] = (grid_search2_params, grid_search2_avg_test_scores)

    

    # Find average accuracy of each set of parameters (SPX win %)
    average_accuracies1 = {}
    for df_counter, parameters_and_accuracies in df_parameters_and_accuracies1.items():
        parameters = parameters_and_accuracies[0]
        for p in parameters:
            average_accuracies1[tuple(p.items())] = 0
    for df_counter, parameters_and_accuracies in df_parameters_and_accuracies1.items():
        parameters = parameters_and_accuracies[0]
        accuracies = parameters_and_accuracies[1]
        for i, p in enumerate(parameters):
            average_accuracies1[tuple(p.items())] += accuracies[i]
    parameters_and_average_accuracies1 = list(average_accuracies1.items())
    best_parameters_and_accuracy1 = max(parameters_and_average_accuracies1, key=lambda x: x[1])
    best_parameters_and_accuracy1 = (best_parameters_and_accuracy1[0], best_parameters_and_accuracy1[1]/len(df_list1))
    # Find average accuracy of each set of parameters (avg returns)
    average_accuracies2 = {}
    for df_counter, parameters_and_accuracies in df_parameters_and_accuracies2.items():
        parameters = parameters_and_accuracies[0]
        for p in parameters:
            average_accuracies2[tuple(p.items())] = 0
    for df_counter, parameters_and_accuracies in df_parameters_and_accuracies2.items():
        parameters = parameters_and_accuracies[0]
        accuracies = parameters_and_accuracies[1]
        for i, p in enumerate(parameters):
            average_accuracies2[tuple(p.items())] += accuracies[i]
    parameters_and_average_accuracies2 = list(average_accuracies2.items())
    best_parameters_and_accuracy2 = max(parameters_and_average_accuracies2, key=lambda x: x[1])
    best_parameters_and_accuracy2 = (best_parameters_and_accuracy2[0], best_parameters_and_accuracy2[1]/len(df_list1))

    # Find out how well the "best parameters" do with each df in df_list2
    best_params1 = best_parameters_and_accuracy1[0]
    best_params2 = best_parameters_and_accuracy2[0]

    for df in df_list2:
        x = df.drop(columns=['SPX_win_percentage', 'average_return', 'portfolio', 'id', 'TMFG_vert_weight_sum', 'TMFG_edge_weight_sum', 'katz', 'degree', 'eigenvector'])
        scaler = sk.preprocessing.MinMaxScaler()
        x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
        y1, y2 = df['SPX win percentage'], df['average return']
        y1 = (y1 > 0.5).astype(int)
        y2 = (y2 > 0).astype(int)

        x1_train, x1_test, y1_train, y1_test = skms.train_test_split(x, y1, test_size=test_size, random_state=2, stratify=y1)
        x2_train, x2_test, y2_train, y2_test = skms.train_test_split(x, y2, test_size=test_size, random_state=2, stratify=y2)

        optimized_model1 = sk.svm.SVC(C=best_params1[0][1], coef0=best_params1[1][1], degree=best_params1[2][1], gamma=best_params1[3][1], kernel='poly', class_weight='balanced', probability=True)
        optimized_model1.fit(x1_train, y1_train)
        optimized_model2 = sk.svm.SVC(C=best_params2[0][1], coef0=best_params2[1][1], degree=best_params2[2][1], gamma=best_params2[3][1], kernel='poly', class_weight='balanced', probability=True)
        optimized_model2.fit(x2_train, y2_train)

        y1_pred = model1.predict(x1_test)
        y2_pred = model1.predict(x2_test)

        # Help calculate goal precisions (for classifications of successful portfolios)
        counter1 = 0
        for percentage in y1:
            if percentage > 0.5:
                counter1 += 1
        counter2 = 0
        for avg_ret in y2:
            if avg_ret > 0:
                counter2 += 1


        # Create the SVM and perform Grid Search
        svc = sk.svm.SVC()
        grid_search1 = skms.GridSearchCV(svc, param_grid, cv=3, scoring=precision_for_successful_portfolios)
        grid_search2 = skms.GridSearchCV(svc, param_grid, cv=3, scoring=precision_for_successful_portfolios)
        grid_search1.fit(x, y1)
        grid_search2.fit(x, y2)
        


        # Optimal parameters and score
        print('Best Parameters (S&P 500):', grid_search1.best_params_)
        print('Best Precision for Successful Portfolios (S&P 500):', grid_search1.best_score_)
        print('Best Parameters (Avg returns):', grid_search2.best_params_)
        print('Best Precision for Successful Portfolios (Avg returns):', grid_search2.best_score_)
        print('---------------------------------------------------------------------------------')

        # Evaluation
        print('Predicting which portfolios will likely outperform the S&P 500:')
        print(f'Goal Precision (1): {counter1/len(y1)}')
        print(f'Best Precision (1): {grid_search1.best_score_}')
        print('Accuracy:', sk.metrics.accuracy_score(y1_test, y1_pred))
        print('\nClassification Report:\n', sk.metrics.classification_report(y1_test, y1_pred))
        print('Confusion Matrix:\n', sk.metrics.confusion_matrix(y1_test, y1_pred))
        print('---------------------------------------------------------------------------------')
        print('Predicting which portfolios will likely yield positive average returns:')
        print(f'Goal Precision (1): {counter2/len(y2)}')
        print(f'Best Precision (1): {grid_search2.best_score_}')
        print('Accuracy:', sk.metrics.accuracy_score(y2_test, y2_pred))
        print('\nClassification Report:\n', sk.metrics.classification_report(y2_test, y2_pred))
        print('Confusion Matrix:\n', sk.metrics.confusion_matrix(y2_test, y2_pred))




        cm = sk.metrics.confusion_matrix(y1_test, y1_pred)
        sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Predicting Outperformance of SPX)')
        plt.tight_layout()
        plt.show()
        cm = sk.metrics.confusion_matrix(y2_test, y2_pred)
        sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Predicting Positivity of Average Returns)')
        plt.tight_layout()
        plt.show()

        print('Done!')


def db_svm_optimizer(df_list1, df_list2, test_size=0.25):
    '''
    Determines 2 sets of hyperparameters that produce the highest average accuracy across all dataframes in df_list1 and applies those 2 sets to df_list2 and analyzes their performance.
    '''
    
    counter = 0
    df_parameters_and_accuracies1 = {}
    df_parameters_and_accuracies2 = {}
    for df in df_list1:
        counter += 1
        # Split features and target
        df = df[df['made_from_freq_edge'] == 1]
        x = df.drop(columns=['SPX_win_percentage', 'average_return', 'portfolio', 'id', 'TMFG_vert_weight_sum', 'TMFG_edge_weight_sum', 'katz', 'degree', 'eigenvector', 'made_from_freq_edge'])
        scaler = sk.preprocessing.MinMaxScaler()
        x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
        y1, y2 = df['SPX_win_percentage'], df['average_return']
        y1 = (y1 > 0.5).astype(int)
        # y2 = (y2 > 0).astype(int)

        # Train/test split
        # x1_train, x1_test, y1_train, y1_test = skms.train_test_split(x, y1, test_size=test_size, random_state=2, stratify=y1)
        # x2_train, x2_test, y2_train, y2_test = skms.train_test_split(x, y2, test_size=test_size, random_state=2, stratify=y2)

        # Initialize and train SVM
        # model1 = sk.svm.SVC(kernel='rbf', C=1, coef0=0, degree=1, gamma=5, class_weight='balanced', probability=True)
        # model1.fit(x1_train, y1_train)
        # model2 = sk.svm.SVC(kernel='poly', C=0.01, coef0=-3, degree=5, gamma='scale', class_weight='balanced', probability=True)
        # model2.fit(x2_train, y2_train)

        # Predict
        # y1_pred = model1.predict(x1_test)
        # y2_pred = model1.predict(x2_test)

        # Define hyperparameter grid
        # param_grid = {
        #     'C': [0.001, 0.01, 0.1, 0.5, 1, 3, 5, 7, 10],
        #     'gamma': ['scale', 'auto', 0.1, 0.5, 1, 5, 10],
        #     'kernel': ['poly'],
        #     'degree': [3, 4, 5],  
        #     'coef0': [-4, -3, -2, -1, 0, 1, 2, 3, 5]
        # }
        param_grid = {
            'C': [0.001, 0.01, 0.1, 0.5, 1, 3, 5, 7, 10],
            'gamma': ['scale', 'auto', 0.1, 0.5, 1, 5, 10],
            'kernel': ['rbf'],
            'degree': [1],   # Ignored with rbf kernel
            'coef0': [0]     # Ignored with rbf kernel
        }

        # Create the SVM and perform grid search
        precision_for_successful_portfolios = sk.metrics.make_scorer(sk.metrics.precision_score, pos_label=1, average='binary')
        cv3 = skms.StratifiedKFold(n_splits=3, shuffle=True, random_state=2)
        svc = sk.svm.SVC(class_weight='balanced')  # weight during CV
        grid_search1 = skms.GridSearchCV(svc, param_grid, cv=cv3, scoring=precision_for_successful_portfolios)
        # grid_search2 = skms.GridSearchCV(svc, param_grid, cv=cv3, scoring=precision_for_successful_portfolios)
        grid_search1.fit(x, y1)
        # grid_search2.fit(x, y2)



        # Store accuracy of each hyperparameter set
        grid_seach1_info = grid_search1.cv_results_
        grid_search1_params = grid_seach1_info['params']
        grid_search1_avg_test_scores = grid_seach1_info['mean_test_score']
        df_parameters_and_accuracies1[counter] = (grid_search1_params, grid_search1_avg_test_scores)
        # grid_seach2_info = grid_search2.cv_results_
        # grid_search2_params = grid_seach2_info['params']
        # grid_search2_avg_test_scores = grid_seach2_info['mean_test_score']
        # df_parameters_and_accuracies2[counter] = (grid_search2_params, grid_search2_avg_test_scores)

    

    # Find average accuracy of each set of parameters (SPX win %)
    average_accuracies1 = {}
    for df_counter, parameters_and_accuracies in df_parameters_and_accuracies1.items():
        parameters = parameters_and_accuracies[0]
        for p in parameters:
            average_accuracies1[tuple(p.items())] = 0
    for df_counter, parameters_and_accuracies in df_parameters_and_accuracies1.items():
        parameters = parameters_and_accuracies[0]
        accuracies = parameters_and_accuracies[1]
        for i, p in enumerate(parameters):
            average_accuracies1[tuple(p.items())] += accuracies[i]
    parameters_and_average_accuracies1 = list(average_accuracies1.items())
    best_parameters_and_accuracy1 = max(parameters_and_average_accuracies1, key=lambda x: x[1])
    best_parameters_and_accuracy1 = (best_parameters_and_accuracy1[0], best_parameters_and_accuracy1[1]/len(df_list1))
    # Find average accuracy of each set of parameters (avg returns)
    # average_accuracies2 = {}
    # for df_counter, parameters_and_accuracies in df_parameters_and_accuracies2.items():
    #     parameters = parameters_and_accuracies[0]
    #     for p in parameters:
    #         average_accuracies2[tuple(p.items())] = 0
    # for df_counter, parameters_and_accuracies in df_parameters_and_accuracies2.items():
    #     parameters = parameters_and_accuracies[0]
    #     accuracies = parameters_and_accuracies[1]
    #     for i, p in enumerate(parameters):
    #         average_accuracies2[tuple(p.items())] += accuracies[i]
    # parameters_and_average_accuracies2 = list(average_accuracies2.items())
    # best_parameters_and_accuracy2 = max(parameters_and_average_accuracies2, key=lambda x: x[1])
    # best_parameters_and_accuracy2 = (best_parameters_and_accuracy2[0], best_parameters_and_accuracy2[1]/len(df_list1))

    # Find out how well the "best parameters" do with each df in df_list2
    best_params1 = best_parameters_and_accuracy1[0]
    # best_params2 = best_parameters_and_accuracy2[0]

    for df in df_list2:
        df = df[df['made_from_freq_edge'] == 1]
        portfolios = df['portfolio'].reset_index(drop=True)
        print(df)
        x = df.drop(columns=['SPX_win_percentage', 'average_return', 'portfolio', 'id', 'TMFG_vert_weight_sum', 'TMFG_edge_weight_sum', 'katz', 'degree', 'eigenvector', 'made_from_freq_edge'])
        print(x)
        scaler = sk.preprocessing.MinMaxScaler()
        x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
        y1, y2 = df['SPX_win_percentage'], df['average_return']
        y1 = (y1 > 0.5).astype(int)
        # y2 = (y2 > 0).astype(int)

        x1_train, x1_test, y1_train, y1_test, ports_train, ports_test = skms.train_test_split(x, y1, portfolios, test_size=test_size, random_state=2, stratify=y1)
        # x2_train, x2_test, y2_train, y2_test = skms.train_test_split(x, y2, test_size=test_size, random_state=2, stratify=y2)

        optimized_model1 = sk.svm.SVC(C=best_params1[0][1], coef0=best_params1[1][1], degree=best_params1[2][1], gamma=best_params1[3][1], kernel='rbf', class_weight='balanced', probability=True)
        optimized_model1.fit(x1_train, y1_train)
        # optimized_model2 = sk.svm.SVC(C=best_params2[0][1], coef0=best_params2[1][1], degree=best_params2[2][1], gamma=best_params2[3][1], kernel='poly', class_weight='balanced', probability=True)
        # optimized_model2.fit(x2_train, y2_train)

        y1_pred = optimized_model1.predict(x1_test)
        # y2_pred = model1.predict(x2_test)

        # Help calculate goal precisions (for classifications of successful portfolios)
        counter1 = 0
        for percentage in y1:
            if percentage > 0.5:
                counter1 += 1
        print(counter1/len(y1))
        counter2 = 0
        for avg_ret in y2:
            if avg_ret > 0:
                counter2 += 1


        # Create the SVM and perform Grid Search
        svc = sk.svm.SVC()
        grid_search1 = skms.GridSearchCV(svc, param_grid, cv=3, scoring=precision_for_successful_portfolios)
        # grid_search2 = skms.GridSearchCV(svc, param_grid, cv=3, scoring=precision_for_successful_portfolios)
        grid_search1.fit(x, y1)
        # grid_search2.fit(x, y2)
        


        # Optimal parameters and score
        print('Best Parameters (S&P 500):', grid_search1.best_params_)
        print('Best Precision for Successful Portfolios (S&P 500):', grid_search1.best_score_)
        # print('Best Parameters (Avg returns):', grid_search2.best_params_)
        # print('Best Precision for Successful Portfolios (Avg returns):', grid_search2.best_score_)
        print('---------------------------------------------------------------------------------')

        # Evaluation
        print('Predicting which portfolios will likely outperform the S&P 500:')
        print(f'Goal Precision (1): {counter1/len(y1)}')
        print(f'Best Precision (1): {grid_search1.best_score_}')
        print('Accuracy:', sk.metrics.accuracy_score(y1_test, y1_pred))
        print('\nClassification Report:\n', sk.metrics.classification_report(y1_test, y1_pred))
        print('Confusion Matrix:\n', sk.metrics.confusion_matrix(y1_test, y1_pred))
        print('---------------------------------------------------------------------------------')
        # print('Predicting which portfolios will likely yield positive average returns:')
        # print(f'Goal Precision (1): {counter2/len(y2)}')
        # print(f'Best Precision (1): {grid_search2.best_score_}')
        # print('Accuracy:', sk.metrics.accuracy_score(y2_test, y2_pred))
        # print('\nClassification Report:\n', sk.metrics.classification_report(y2_test, y2_pred))
        # print('Confusion Matrix:\n', sk.metrics.confusion_matrix(y2_test, y2_pred))




        # cm = sk.metrics.confusion_matrix(y1_test, y1_pred)
        # sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')
        # plt.title('Confusion Matrix (Predicting Outperformance of SPX)')
        # plt.tight_layout()
        # plt.show()
        # cm = sk.metrics.confusion_matrix(y2_test, y2_pred)
        # sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')
        # plt.title('Confusion Matrix (Predicting Positivity of Average Returns)')
        # plt.tight_layout()
        # plt.show()

        print('Done!')
        print(ports_test[y1_pred == 1].tolist())
        return ports_test[y1_pred == 1].tolist()







class PortfolioGAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=16):
        super(PortfolioGAT, self).__init__()
        self.gat1 = torch_geometric.nn.GATConv(num_features, hidden_channels, heads=2, concat=True)
        self.gat2 = torch_geometric.nn.GATConv(hidden_channels * 2, 1, heads=1, concat=False)

    # Return logits
    def forward(self, x, edge_index):
        x = torch.nn.functional.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x  
    
def networkx_to_GAT(g, feature_dict, success_label):
    if 'TMFG size' in feature_dict.keys():
        feature_dict.pop('TMFG size')
    data = torch_geometric.utils.from_networkx(g)
    
    # Define node features
    data.x = torch.tensor([[feature_dict[k][n] for k in feature_dict] for n in g.nodes], dtype=torch.float)
    feature_names = list(feature_dict.keys())
    
    # Create binary labels
    data.y = torch.tensor([g.nodes[n][success_label] for n in g.nodes], dtype=torch.float).unsqueeze(1)
    
    # Edge attributes
    data.edge_attr = torch.tensor([g.edges[e]['weight'] for e in g.edges], dtype=torch.float).view(-1, 1)
    
    return data, feature_names
    
# Return accuracy score, precision score, and recall score
def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = torch.sigmoid(logits).cpu().numpy() > 0.5
        y_true = data.y.cpu().numpy()
        
        acc = sk.metrics.accuracy_score(y_true, preds)
        prec = sk.metrics.precision_score(y_true, preds)
        rec = sk.metrics.recall_score(y_true, preds)
        
    return acc, prec, rec


def find_important_features_per_node(model, data, feature_names, success_metric, epochs=200):
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.explain.config import ModelConfig, ModelReturnType

    explainer = Explainer(
        model,
        GNNExplainer(epochs=epochs),
        "model",
        ModelConfig(
            mode="binary_classification",
            task_level="node",
            return_type=ModelReturnType.raw
        ),
        "attributes",
        "object"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = data.to(device)
    model.eval()

    feature_scores = torch.zeros(data.num_node_features, device=device)
    for i in range(data.num_nodes):
        target = int(data.y[i].item())
        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            target=target,
            index=i
        )

        feature_mask = explanation.node_mask[i].detach()
        feature_scores += feature_mask


    avg_importance = feature_scores / data.num_nodes
    num_features = len(avg_importance)
    indices = np.arange(num_features)

    plt.figure(figsize=(10, 5))
    plt.bar(indices, avg_importance.numpy())
    plt.xticks(indices, feature_names, rotation=45, ha="right")
    plt.ylabel("Average Importance")
    plt.title("Node Feature Importance (Averaged Across All Nodes) for " + str(success_metric))
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.show()
    return avg_importance.cpu()

def trainGAT(model, data, feature_names, success_metric, epochs=500, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Lists to store metrics for plotting
    losses = []
    accuracies = []
    precisions = []
    recalls = []
    epoch_ticks = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()

        # Store metrics every 10 epochs
        if epoch % 10 == 0:
            acc, prec, rec = evaluate(model, data)
            print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")
            losses.append(loss.item())
            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            epoch_ticks.append(epoch)
    for i in range(3):
        epoch_ticks.pop(0)
        losses.pop(0)
        accuracies.pop(0)
        precisions.pop(0)
        recalls.pop(0)

    # Plot all metrics
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_ticks, losses, label='Loss')
    plt.plot(epoch_ticks, accuracies, label='Accuracy')
    plt.plot(epoch_ticks, precisions, label='Precision')
    plt.plot(epoch_ticks, recalls, label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('GNN Training Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    find_important_features_per_node(model, data, feature_names, success_metric)

import torch
import networkx as nx
from typing import Dict, Tuple


class DotProductPredictor(torch.nn.Module):
    def forward(self, emb):
        scores = torch.matmul(emb, emb.T)
        return torch.sigmoid(scores)

def predict_tmfg_from_temporal_graphs(
    graph_dict: Dict[str, nx.Graph],
    node_feature_key: str = "percent_change",
    raw_msg_dim: int = 1,
    memory_dim: int = 32,
    time_dim: int = 10
) -> nx.Graph:
    """
    Predicts the next TMFG (planar graph) using TGNMemory from a series of historical TMFGs.

    Parameters:
    ------------
    graph_dict : Dict[str, nx.Graph]
        Keys are date strings, values are NetworkX graphs (TMFGs) with node and edge features.
    node_feature_key : str
        Node attribute used as node feature (e.g., percent change).
    edge_feature_key : str
        Edge attribute used as edge message (e.g., normalized covariance).

    Returns:
    --------
    nx.Graph
        A predicted TMFG structure as a NetworkX graph (planar).
    """
    # Step 1: Build consistent node index mapping
    all_nodes = sorted(set().union(*[g.nodes for g in graph_dict.values()]))
    node_to_id = {node: i for i, node in enumerate(all_nodes)}
    id_to_node = {i: node for node, i in node_to_id.items()}
    num_nodes = len(all_nodes)

    # Step 2: Gather temporal edge and node feature data
    sorted_times = sorted(graph_dict.keys())
    past_times = sorted_times[:-1]

    src_list, dst_list, t_list, raw_msg_list = [], [], [], []
    node_features = torch.zeros((num_nodes, 1))  # only last known features used

    for t_idx, t in enumerate(past_times):
        time_float = float(t_idx)
        g = graph_dict[t]

        for u, v, attr in g.edges(data=True):
            for a, b in [(u, v), (v, u)]:
                src_list.append(node_to_id[a])
                dst_list.append(node_to_id[b])
                t_list.append(time_float)
                raw_msg_list.append([attr['weight']])

        for n, attr in g.nodes(data=True):
            node_features[node_to_id[n], 0] = attr['weight']  # latest known

    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    t = torch.tensor(t_list, dtype=torch.float)
    raw_msg = torch.tensor(raw_msg_list, dtype=torch.float)

    # Step 3: Initialize memory
    memory = TGNMemory(
        num_nodes=num_nodes,
        raw_msg_dim=raw_msg_dim,
        memory_dim=memory_dim,
        time_dim=time_dim,
        message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
        aggregator_module=LastAggregator()
    )
    memory.reset_state()
    memory.update_state(src, dst, t, raw_msg)

    # Step 4: Extract memory embeddings
    all_node_ids = torch.arange(num_nodes)
    mem, _ = memory(all_node_ids)

    # Step 5: Predict edge scores
    predictor = DotProductPredictor()
    probs = predictor(mem)
    edge_candidates = [
        (i, j, float(probs[i, j]))
        for i in range(num_nodes)
        for j in range(i + 1, num_nodes)
    ]
    edge_candidates.sort(key=lambda x: x[2], reverse=True)

    # Step 6: Build predicted planar graph (TMFG-compatible)
    G_pred = nx.Graph()
    G_pred.add_nodes_from([(id_to_node[i], {node_feature_key: float(node_features[i])}) for i in range(num_nodes)])

    for u, v, score in edge_candidates:
        G_pred.add_edge(id_to_node[u], id_to_node[v], prob=score)
        is_planar, _ = nx.check_planarity(G_pred)
        if not is_planar:
            G_pred.remove_edge(id_to_node[u], id_to_node[v])

    # Draw predicted TMFG
    plt.figure(figsize=(12, 8))
    plt.title('TMFG')
    pos0 = nx.planar_layout(g, scale=2)
    # nx.draw(g, pos=pos0, node_color='#5192b8', node_size=650)
    # nx.draw(g, pos=pos0, with_labels=True, node_color='#8fd6ff', edge_color='#5192b8', node_size=600, font_size=8)
    nx.draw(g, pos=pos0, with_labels=True, edge_color='#5192b8', node_size=600, font_size=8)
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(g, 'weight')
    for k, v in edge_labels.items():
        edge_labels[k] = round(v, 2)
    nx.draw_networkx_edge_labels(g, pos=pos0, edge_labels=edge_labels, font_size=6)
    plt.tight_layout()
    plt.show()

    return G_pred


def evaluate_graph_prediction(
    G_pred: nx.DiGraph,
    G_true: nx.DiGraph,
    use_probabilities: bool = False,
    threshold: float = 0.5
) -> dict:
    """
    Evaluates a predicted graph against a true graph using binary classification metrics.

    Parameters:
    -----------
    G_pred : nx.DiGraph
        Predicted graph, optionally with edge attribute 'prob' for probability of edge.
    G_true : nx.DiGraph
        Ground-truth graph to compare against.
    use_probabilities : bool
        If True, use edge probabilities in G_pred['prob'] for ROC AUC. Otherwise binarize with threshold.
    threshold : float
        Threshold for binarizing edge predictions if use_probabilities is False.

    Returns:
    --------
    metrics : dict
        Dictionary containing precision, recall, f1 score, and optionally ROC AUC.
    """
    all_nodes = set(G_true.nodes()) | set(G_pred.nodes())
    possible_edges = {(u, v) for u in all_nodes for v in all_nodes if u != v}

    y_true = []
    y_pred = []
    y_scores = []

    for u, v in possible_edges:
        true = 1 if G_true.has_edge(u, v) else 0
        y_true.append(true)

        if G_pred.has_edge(u, v):
            prob = G_pred[u][v].get("prob", 1.0)
        else:
            prob = 0.0

        if use_probabilities:
            y_scores.append(prob)
        y_pred.append(1 if prob >= threshold else 0)

    precision = sk.metrics.precision_score(y_true, y_pred)
    recall = sk.metrics.recall_score(y_true, y_pred)
    f1 = sk.metrics.f1_score(y_true, y_pred)

    results = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    if use_probabilities:
        auc = sk.metrics.roc_auc_score(y_true, y_scores)
        results["roc_auc"] = auc
    
    print(results)

    return results

def visualize_tmfg_prediction(G_pred: nx.Graph, G_true: nx.Graph, title: str = "Predicted TMFG"):
    """
    Visualizes the predicted TMFG with color-coded edges:
      - Green = correctly predicted (exists in both predicted and true graph)
      - Red = incorrect (exists only in predicted)
    
    Parameters:
    -----------
    G_pred : nx.Graph
        The predicted TMFG.
    G_true : nx.Graph
        The ground-truth TMFG.
    title : str
        Title of the plot.
    """
    pos = nx.planar_layout(G_pred)  # consistent layout
    
    correct_edges = set(G_pred.edges()) & set(G_true.edges())
    incorrect_edges = set(G_pred.edges()) - correct_edges

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G_pred, pos, node_color='lightgray', node_size=500)

    nx.draw_networkx_edges(G_pred, pos, edgelist=correct_edges, edge_color='green', width=2, label='Correct')
    nx.draw_networkx_edges(G_pred, pos, edgelist=incorrect_edges, edge_color='red', width=1.5, style='dashed', label='Incorrect')

    nx.draw_networkx_labels(G_pred, pos, font_size=10, font_color='black')
    
    plt.title(title)
    plt.axis('off')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()



    

poster_df = db.table_into_df('stocks:30 1:2024-11-01 2:2024-11-29 3:2024-12-02 4:2024-12-31')

# df01 = db.table_into_df('!stocks:60 1:2020-12-23 2:2021-02-05 3:2021-02-08 4:2021-05-14')
# df02 = db.table_into_df('!stocks:60 1:2021-03-04 2:2021-10-28 3:2021-10-29 4:2021-12-17')
# df03 = db.table_into_df('!stocks:60 1:2023-03-13 2:2023-05-26 3:2023-05-30 4:2023-07-28')
# df04 = db.table_into_df('!stocks:60 1:2023-11-14 2:2024-01-24 3:2024-01-25 4:2024-04-05')
# db_svm_optimizer([df01, df02], [df03, df04])


# df01 = db.table_into_df('stocks:60 1:2023-01-03 2:2023-01-31 3:2023-02-01 4:2023-02-28')
# df02 = db.table_into_df('stocks:60 1:2023-02-01 2:2023-02-28 3:2023-03-01 4:2023-03-31')
# df03 = db.table_into_df('stocks:60 1:2023-03-01 2:2023-03-31 3:2023-04-03 4:2023-04-28')
# df04 = db.table_into_df('stocks:60 1:2023-04-03 2:2023-04-28 3:2023-05-01 4:2023-05-31')
# df05 = db.table_into_df('stocks:60 1:2023-05-01 2:2023-05-31 3:2023-06-01 4:2023-06-30')
# df06 = db.table_into_df('stocks:60 1:2023-06-01 2:2023-06-30 3:2023-07-03 4:2023-07-31')
# df07 = db.table_into_df('stocks:60 1:2023-07-03 2:2023-07-31 3:2023-08-01 4:2023-08-31')
# df08 = db.table_into_df('stocks:60 1:2023-08-01 2:2023-08-31 3:2023-09-01 4:2023-09-29')
# df09 = db.table_into_df('stocks:60 1:2023-09-01 2:2023-09-29 3:2023-10-02 4:2023-10-31')
# df10 = db.table_into_df('stocks:60 1:2023-10-02 2:2023-10-31 3:2023-11-01 4:2023-11-30')
# df11 = db.table_into_df('stocks:60 1:2023-11-01 2:2023-11-29 3:2023-12-01 4:2023-12-29')
# df12 = db.table_into_df('stocks:60 1:2023-12-01 2:2023-12-29 3:2024-01-02 4:2024-01-31')
# df13 = db.table_into_df('stocks:60 1:2024-01-02 2:2024-01-31 3:2024-02-01 4:2024-02-29')
# df14 = db.table_into_df('stocks:60 1:2024-02-01 2:2024-02-29 3:2024-03-01 4:2024-03-28')
# df15 = db.table_into_df('stocks:60 1:2024-03-01 2:2024-03-28 3:2024-04-01 4:2024-04-30')
# df16 = db.table_into_df('stocks:60 1:2024-04-01 2:2024-04-30 3:2024-05-01 4:2024-05-31')
# df17 = db.table_into_df('stocks:60 1:2024-05-01 2:2024-05-31 3:2024-06-03 4:2024-06-28')
# df18 = db.table_into_df('stocks:60 1:2024-06-04 2:2024-06-28 3:2024-07-01 4:2024-07-31')
# df19 = db.table_into_df('stocks:60 1:2024-07-01 2:2024-07-31 3:2024-08-01 4:2024-08-30')
# df20 = db.table_into_df('stocks:60 1:2024-08-01 2:2024-08-30 3:2024-09-03 4:2024-09-30')
# df21 = db.table_into_df('stocks:60 1:2024-09-03 2:2024-09-30 3:2024-10-01 4:2024-10-31')
# df22 = db.table_into_df('stocks:60 1:2024-10-01 2:2024-10-31 3:2024-11-01 4:2024-11-29')
# df23 = db.table_into_df('stocks:60 1:2024-11-01 2:2024-11-29 3:2024-12-02 4:2024-12-31')


# db_svm_optimizer([df01, df02], [df03]) # Goal: 37%, P1: 38%
# db_svm_optimizer([df02, df03], [df04]) # Goal: 52%, P1: 50%
# db_svm_optimizer([df03, df04], [df05]) # Goal: 42%, P1: 29%
# db_svm_optimizer([df04, df05], [df06]) # Goal: 39%, P1: 55%
# db_svm_optimizer([df05, df06], [df07]) # Goal: 52%, P1: 50%
# db_svm_optimizer([df06, df07], [df08]) # Goal: 47%, P1: 75%
# db_svm_optimizer([df07, df08], [df09]) # Goal: 55%, P1: 100%
# db_svm_optimizer([df08, df09], [df10]) # Goal: 16%, P1: 0%
# db_svm_optimizer([df09, df10], [df11]) # Goal: 69%, P1: 57%
# db_svm_optimizer([df10, df11], [df12]) # Goal: 46%, P1: 38%
# db_svm_optimizer([df11, df12], [df13]) # 
# db_svm_optimizer([df12, df13], [df14]) # 
# db_svm_optimizer([df13, df14], [df15]) # 
# db_svm_optimizer([df14, df15], [df16]) # 
# db_svm_optimizer([df15, df16], [df17]) #
# db_svm_optimizer([df16, df17], [df18])
# db_svm_optimizer([df17, df18], [df19]) 
# db_svm_optimizer([df18, df19], [df20])
# db_svm_optimizer([df19, df20], [df21])
# db_svm_optimizer([df20, df21], [df22])
# db_svm_optimizer([df21, df22], [df23])


# With polynomial kernel
# db_svm_optimizer([df02], [df03]) # Bad
# db_svm_optimizer([df03], [df04]) # Bad
# db_svm_optimizer([df04], [df05]) # Bad
# db_svm_optimizer([df05], [df06]) # Good
# db_svm_optimizer([df06], [df07]) # Bad
# db_svm_optimizer([df07], [df08]) # Good
# db_svm_optimizer([df08], [df09]) # Good
# db_svm_optimizer([df09], [df10]) # Good
# db_svm_optimizer([df10], [df11]) # Neutral
# db_svm_optimizer([df11], [df12]) # Bad
# db_svm_optimizer([df12], [df13]) # Bad
# db_svm_optimizer([df13], [df14]) # 
# db_svm_optimizer([df14], [df15]) # 
# db_svm_optimizer([df15], [df16]) # 
# db_svm_optimizer([df16], [df17]) # 
# db_svm_optimizer([df17], [df18]) # 
# db_svm_optimizer([df18], [df19])
# db_svm_optimizer([df19], [df20])
# db_svm_optimizer([df20], [df21])
# db_svm_optimizer([df21], [df22])
# db_svm_optimizer([df22], [df23])



# With rbf kernel
# db_svm_optimizer([df01, df02, df03, df04, df05], [df06]) # 
# db_svm_optimizer([df02, df03, df04, df05, df06], [df07]) # 
# db_svm_optimizer([df03, df04, df05, df06, df07], [df08]) # 
# db_svm_optimizer([df04, df05, df06, df07, df08], [df09]) # 
# db_svm_optimizer([df05, df06, df07, df08, df09], [df10]) #
# db_svm_optimizer([df06, df07, df08, df09, df10], [df11]) #  
# predicted_portfolios = db_svm_optimizer([df07, df08, df09, df10, df11], [df12]) # 
# db_svm_optimizer([df08, df09, df10, df11, df12], [df13]) #
# db_svm_optimizer([df09, df10, df11, df12, df13], [df14]) # 
# db_svm_optimizer([df10, df11, df12, df13, df14], [df15]) # 
# db_svm_optimizer([df11, df12, df13, df14, df15], [df16]) # 
# db_svm_optimizer([df12, df13, df14, df15, df16], [df17]) # 
# db_svm_optimizer([df13, df14, df15, df16, df17], [df18]) #  
# db_svm_optimizer([df14, df15, df16, df17, df18], [df19]) # 
# db_svm_optimizer([df15, df16, df17, df18, df19], [df20]) # 
# db_svm_optimizer([df16, df17, df18, df19, df20], [df21]) # 
# db_svm_optimizer([df17, df18, df19, df20, df21], [df22]) # 
# db_svm_optimizer([df18, df19, df20, df21, df22], [df23]) # 




# db_svm_optimizer([df02], [df03]) # Bad | Goal: 0.41, P1: 0.30
# db_svm_optimizer([df03], [df04]) # Bad | Goal: 0.41, P1: 0.30
# db_svm_optimizer([df04], [df05]) # Bad | Goal: 0.36, P1: 0.50
# db_svm_optimizer([df05], [df06]) # Bad | Goal: 0.30, P1: 0.29
# db_svm_optimizer([df06], [df07]) # 
# db_svm_optimizer([df07], [df08]) # 
# db_svm_optimizer([df08], [df09]) # 
# db_svm_optimizer([df09], [df10]) # 
# db_svm_optimizer([df10], [df11]) #
# db_svm_optimizer([df11], [df12]) #
# db_svm_optimizer([df12], [df13]) # 
# db_svm_optimizer([df13], [df14]) # 
# db_svm_optimizer([df14], [df15]) # 
# db_svm_optimizer([df15], [df16]) # 
# db_svm_optimizer([df16], [df17]) # 
# db_svm_optimizer([df17], [df18]) # 
# db_svm_optimizer([df18], [df19])
# db_svm_optimizer([df19], [df20])
# db_svm_optimizer([df20], [df21])
# db_svm_optimizer([df21], [df22])
# db_svm_optimizer([df22], [df23])




# svm_classifier(df01)
# svm_classifier(df02)
# svm_classifier(df03)
# svm_classifier(df04)
# svm_classifier(df05)
# svm_classifier(df06)
# svm_classifier(df07)
# svm_classifier(df08)




# df04 = db.table_into_df('stocks:60 1:2023-04-03 2:2023-04-28 3:2023-05-01 4:2023-05-31')

# df1 = pd.DataFrame(pcd.infodict1_60stocks)
# df2 = pd.DataFrame(pcd.infodict2_60stocks)
# df3 = pd.DataFrame(pcd.infodict3_60stocks)
# df4 = pd.DataFrame(pcd.infodict4_60stocks)
# df5 = pd.DataFrame(pcd.infodict5_60stocks)
# df6 = pd.DataFrame(pcd.infodict6_60stocks)
# df7 = pd.DataFrame(pcd.infodict7_60stocks)
# df8 = pd.DataFrame(pcd.infodict8_60stocks)
# df9 = pd.DataFrame(pcd.infodict9_60stocks)

# svm_optimizer([df1, df2, df4, df5, df7, df8], [df3, df6, df9])
# svm_optimizer([df1, df2], [df3])
# svm_optimizer([df4, df5], [df6])
# svm_optimizer([df7, df8], [df9])




# df = pd.DataFrame(info_dict)

# print(df)
# lin_regression(df)
# logistic_regression(df)


# svm_classifier(poster_df)









# df01 = db.table_into_df('stocks:30 1:2023-01-03 2:2023-01-31 3:2023-02-01 4:2023-02-28')
# df02 = db.table_into_df('stocks:30 1:2023-02-01 2:2023-02-28 3:2023-03-01 4:2023-03-31')
# df03 = db.table_into_df('stocks:30 1:2023-03-01 2:2023-03-31 3:2023-04-03 4:2023-04-28')
# df04 = db.table_into_df('stocks:30 1:2023-04-03 2:2023-04-28 3:2023-05-01 4:2023-05-31')
# df05 = db.table_into_df('stocks:30 1:2023-05-01 2:2023-05-31 3:2023-06-01 4:2023-06-30')
# df06 = db.table_into_df('stocks:30 1:2023-06-01 2:2023-06-30 3:2023-07-03 4:2023-07-31')
# df07 = db.table_into_df('stocks:30 1:2023-07-03 2:2023-07-31 3:2023-08-01 4:2023-08-31')
# df08 = db.table_into_df('stocks:30 1:2023-08-01 2:2023-08-31 3:2023-09-01 4:2023-09-29')
# df09 = db.table_into_df('stocks:30 1:2023-09-01 2:2023-09-29 3:2023-10-02 4:2023-10-31')
# df10 = db.table_into_df('stocks:30 1:2023-10-02 2:2023-10-31 3:2023-11-01 4:2023-11-30')
# df11 = db.table_into_df('stocks:30 1:2023-11-01 2:2023-11-29 3:2023-12-01 4:2023-12-29')
# df12 = db.table_into_df('stocks:30 1:2023-12-01 2:2023-12-29 3:2024-01-02 4:2024-01-31')
# df13 = db.table_into_df('stocks:30 1:2024-01-02 2:2024-01-31 3:2024-02-01 4:2024-02-29')
# df14 = db.table_into_df('stocks:30 1:2024-02-01 2:2024-02-29 3:2024-03-01 4:2024-03-28')
# df15 = db.table_into_df('stocks:30 1:2024-03-01 2:2024-03-28 3:2024-04-01 4:2024-04-30')
# df16 = db.table_into_df('stocks:30 1:2024-04-01 2:2024-04-30 3:2024-05-01 4:2024-05-31')
# df17 = db.table_into_df('stocks:30 1:2024-05-01 2:2024-05-31 3:2024-06-03 4:2024-06-28')
# df18 = db.table_into_df('stocks:30 1:2024-06-04 2:2024-06-28 3:2024-07-01 4:2024-07-31')
# df19 = db.table_into_df('stocks:30 1:2024-07-01 2:2024-07-31 3:2024-08-01 4:2024-08-30')
# df20 = db.table_into_df('stocks:30 1:2024-08-01 2:2024-08-30 3:2024-09-03 4:2024-09-30')
# df21 = db.table_into_df('stocks:30 1:2024-09-03 2:2024-09-30 3:2024-10-01 4:2024-10-31')
# df22 = db.table_into_df('stocks:30 1:2024-10-01 2:2024-10-31 3:2024-11-01 4:2024-11-29')
# df23 = db.table_into_df('stocks:30 1:2024-11-01 2:2024-11-29 3:2024-12-02 4:2024-12-31')

# db_svm_optimizer([df01, df02, df03, df04, df05], [df06]) # 
# db_svm_optimizer([df02, df03, df04, df05, df06], [df07]) # 
# db_svm_optimizer([df03, df04, df05, df06, df07], [df08]) # 
# db_svm_optimizer([df04, df05, df06, df07, df08], [df09]) # 
# db_svm_optimizer([df05, df06, df07, df08, df09], [df10]) #
# db_svm_optimizer([df06, df07, df08, df09, df10], [df11]) #  
# db_svm_optimizer([df07, df08, df09, df10, df11], [df12]) # 
# db_svm_optimizer([df08, df09, df10, df11, df12], [df13]) #
# db_svm_optimizer([df09, df10, df11, df12, df13], [df14]) # 
# db_svm_optimizer([df10, df11, df12, df13, df14], [df15]) # 
# db_svm_optimizer([df11, df12, df13, df14, df15], [df16]) # 
# db_svm_optimizer([df12, df13, df14, df15, df16], [df17]) # 
# db_svm_optimizer([df13, df14, df15, df16, df17], [df18]) #  
# db_svm_optimizer([df14, df15, df16, df17, df18], [df19]) # 
# db_svm_optimizer([df15, df16, df17, df18, df19], [df20]) # 
# db_svm_optimizer([df16, df17, df18, df19, df20], [df21]) # 
# db_svm_optimizer([df17, df18, df19, df20, df21], [df22]) # 
# db_svm_optimizer([df18, df19, df20, df21, df22], [df23]) # 




# db_svm_optimizer([df01, df02, df03], [df04]) # 
# db_svm_optimizer([df02, df03, df04], [df05]) # 
# db_svm_optimizer([df03, df04, df05], [df06]) # 
# db_svm_optimizer([df04, df05, df06], [df07]) # 
# db_svm_optimizer([df05, df06, df07], [df08]) # 
# db_svm_optimizer([df06, df07, df08], [df09]) # 
# db_svm_optimizer([df07, df08, df09], [df10]) #
# db_svm_optimizer([df08, df09, df10], [df11]) #  
# db_svm_optimizer([df09, df10, df11], [df12]) # 
# db_svm_optimizer([df10, df11, df12], [df13]) #
# db_svm_optimizer([df11, df12, df13], [df14]) # 
# db_svm_optimizer([df12, df13, df14], [df15]) # 
# db_svm_optimizer([df13, df14, df15], [df16]) # 
# db_svm_optimizer([df14, df15, df16], [df17]) # 
# db_svm_optimizer([df15, df16, df17], [df18]) #  
# db_svm_optimizer([df16, df17, df18], [df19]) # 
# db_svm_optimizer([df17, df18, df19], [df20]) # 
# db_svm_optimizer([df18, df19, df20], [df21]) # 
# db_svm_optimizer([df19, df20, df21], [df22]) # 
# db_svm_optimizer([df20, df21, df22], [df23]) # 