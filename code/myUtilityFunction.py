#!/usr/bin/env python3
# Core python library
# Core python library
import os
import gc
import pandas as pd
import numpy as np
from numpy import median
import statsmodels.api as sm
pd.set_option('display.max_columns', 500)

# ignore the warning message
import warnings
warnings.filterwarnings('ignore')

# visualize related
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
from IPython.display import display, HTML
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)

# check the VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor  

# Machine learning related
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import (
    precision_score, recall_score, f1_score, precision_recall_curve
)

# Visualize the tree model
import graphviz

def quick_df_explorer(df: pd.DataFrame):
    print('Number of observations: %s, Number of columns / features: %s' % \
      (df.shape[0], df.shape[1]))

    # sort=False because pandas cannot order a StringDtype against a numpy dtype
    _dtypes = df.columns.to_series().groupby(df.dtypes, sort=False).groups
    print('\nThe data types are %s' % {k.name: v for k, v in _dtypes.items()}) 
    print('\nThe statistic of each columns:\n')
    display(df.describe())


def get_missing_values(df, 
                       return_missing_df_flag=False):
    '''
    Function to explore how many missing values (NaN) in the pandas
    @Args:
       df: pandas dataframe
       return_missing_df_flag: The boolean flag to return the missing pandas or not
       
    Return:
       Depends on the boolean flag; it will return the missing data table
    '''
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'}
    )
    mis_val_table_ren_columns = (
        mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0]
        .sort_values('% of Total Values', ascending=False).round(1)
    )
    print(
        "There are %s columns that have missing values"
        % str(mis_val_table_ren_columns.shape[0])
    )
    if return_missing_df_flag:
        return mis_val_table_ren_columns


def plot_count_numpy(array: np.ndarray, plot_name: str=''):
    '''
    Function to plot nomidal or categorical variables from numpy array
    @Args:
      array: a numpy array containing the data to plot
      plot_name: string to be title of the plot
      
    Return:
      No object returned, but the visualization is display on console or notebook
    '''
    data_ = array
    hist_ = np.histogram(data_)
    binsize_ = hist_[1][1] - hist_[1][0]
    
    _trace = go.Histogram(x=data_,
                          name='count',
                          marker=dict(color='white')
                         )
    
    layout = go.Layout(title=dict(text=plot_name, 
                                  font=dict(family='Helvetica',
                                            size=28,
                                            color='white'
                                           )
                                 ),
                       bargap=.2,
                       paper_bgcolor='#000',
                       plot_bgcolor='#000',
                       autosize=False,
                       width=600,
                       height=400,
                       xaxis=dict(color='gray', showgrid=False),
                       yaxis=dict(color='gray', showgrid=False)
                      )
    trace_data = [_trace]
    fig = go.Figure(data=trace_data, layout=layout)
    py.iplot(fig)


def plot_categorical(df: pd.DataFrame , col:str):
    """
    Function to plot the categorical data on piechart using Plotly
    @Args:
      df: pandas data frame
      col: A string column name within pandas data frame to plot
      
    Return:
      No object return, only visualization
    """
    value_ = df[col].value_counts().values
    idx_ = df[col].value_counts().index
    
    trace = [{'values': value_,
              'labels': idx_,
              'name': col,
              'hoverinfo': 'label+value+name',
              'hole': 0.4,
              'type': 'pie'
             }]

    layout = {'title': '<b>%s</b> categorical distribution' % col,
              'paper_bgcolor': '#e8e8e8',
              'plot_bgcolor': '#e8e8e8',
              'autosize': False,
              'width': 800,
              'height': 400,
              'annotations': [{'text' : '<b>%s</b>' % col,
                              'font': {'size': 11,
                                       'color': 'black'},
                              'x': 0.5,
                              'y': 0.5,
                              'showarrow': False
                              }]
             }
    py.iplot({'data': trace, 'layout': layout})


def prediction_evaluation (algorithm, X_train, X_test, y_train, y_test, 
                           predictor_cols, cf = 'features'):
    """
     Function to predict and evaluate the provided algorithm by using Plotly library 
       to visualize the confusion matrix, ROC curve as well as provided the
       feature importances.
     @Args:
       algorithm: the model algorithm object
       X_train: the predictor features of the training pandas data frame
       X_test: the predictor features of the testing pandas data frame
       y_train: the target variable of the training pandas data frame
       y_test: the target variable of the testing pandas data frame
       cf: toggle the mode on how to get the informaiton out from the model, 
         the input only accepts 2 possible list of values. 
         LOV - 'coefficients': specifically for logistic regression
             - 'features': specifically for tree-based model
     Return:
        prediction and probabilities
    """
    if cf not in ['features', 'coefficients']:
        # Exception case - return None
        print("ERROR: Mode Toggle (cf parameters) is not in LOV. Please recheck")
        return None, None
    
    algorithm.fit(X_train, y_train)
    y_pred = algorithm.predict(X_test)
    y_prob = algorithm.predict_proba(X_test)
    algorithm_name = str(algorithm).split('(', 1)[0] 
    
    if cf == 'coefficients':
        coeff = pd.DataFrame(algorithm.coef_.ravel())
    elif cf == 'features':
        coeff = pd.DataFrame(algorithm.feature_importances_)
        
    col_df = pd.DataFrame(predictor_cols)
    coef_smry = pd.merge(coeff, col_df, left_index=True, right_index=True, how='left')
    coef_smry.columns = ['coefficients', 'features']
    coef_smry = coef_smry.sort_values(by='coefficients', ascending=False)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # compute metric
    tp = conf_matrix[1,1]
    fn = conf_matrix[1,0]
    fp = conf_matrix[0,1]
    tn = conf_matrix[0,0]
    accuracy_  = ((tp + tn) / (tp + tn + fp + fn))
    precision_ = (tp / (tp + fp))
    recall_    = (tp / (tp + fn))
    f1_score_  = f1_score(y_test, y_pred)
    model_roc_auc = roc_auc_score(y_test, y_pred)
    
    # Print report
    print(algorithm)
    print("\nClassification report: \n", classification_report(y_test, y_pred))
    print("\nAccuracy Score: ", np.round(accuracy_score(y_test, y_pred), 4))
    print("F1 Score: ", np.round(f1_score_, 4))
    print("Area Under Curve: ", np.round(model_roc_auc, 4), "\n")
    
    # Trace 1: plot confusion matrix
    trace1 = go.Heatmap(z = conf_matrix,
                        x = ['Not Leave', 'Leave'],
                        y = ['Not Leave', 'Leave'],
                        showscale = False,
                        colorscale = 'Picnic',
                        name = "Confusion Matrix"
                       )
    
    # Trace 2: plot model metrics
    show_metrics = pd.DataFrame(data=[[accuracy_ , precision_, recall_, f1_score_]])
    show_metrics = show_metrics.T
    colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue']
    trace2 = go.Bar(x=(show_metrics[0].values), 
                    y=['Accuracy', 'Precision', 'Recall', 'F1 score'], 
                    text=np.round(show_metrics[0].values,4),
                    name='',
                    textposition='auto',
                    orientation='h', 
                    opacity=0.8,
                    marker=dict(color=colors,
                                line=dict(color='#000000',
                                          width=1.5)
                               )
                   )

    # Trace 3: plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    trace3 = go.Scatter(x = fpr,
                        y = tpr,
                        name = "ROC: " + str(model_roc_auc),
                        line = dict(color = 'rgb(22, 96, 197)',
                                    width = 2
                                   )
                       )
    trace4 = go.Scatter(x = [0, 1],
                        y = [0, 1],
                        line = dict(color = 'rgb(205, 12, 24)',
                                     width = 1.5,
                                     dash = 'dot'
                                   )
                       )
    
    # Trace 4: plot precision-recall curve
    __precision, __recall, t = precision_recall_curve(y_test, y_prob[:, 1])
    trace5 = go.Scatter(x=__recall, 
                        y=__precision,
                        name="Precision %s" % str(__precision),
                        line=dict(color=('lightcoral'),
                                  width = 2), 
                        fill='tozeroy'
                       )
    
    # Trace 5: plot coeffs
    trace6 = go.Bar(x = coef_smry['features'],
                    y = coef_smry['coefficients'],
                    name = "coefficients",
                    marker = dict(color = coef_smry['coefficients'],
                                  colorscale = 'Picnic',
                                  line = dict(width = .6, color = 'black')
                                 )
                   )
    
    # subplots
    fig = tls.make_subplots(rows = 3, cols = 2, 
                            specs = [[{}, {}], 
                                     [{}, {}],
                                     [{'colspan': 2}, None]],
                            subplot_titles = ('Confusion Matrix',
                                              'Metrics',
                                              'Receiver Operating Characteristics (ROC)',
                                              'Precision - Recall curve',
                                              'Feature Importances'
                                             )
                           )
    
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 2, 1)
    fig.append_trace(trace4, 2, 1)
    fig.append_trace(trace5, 2, 2)
    fig.append_trace(trace6, 3, 1)
    
    fig['layout'].update(showlegend = False,
                         title = "Model Performance of {}".format(algorithm_name),
                         autosize = False,
                         height = 1000,
                         width = 800,
                         plot_bgcolor = 'rgba(240, 240, 240, 0.95)',
                         paper_bgcolor = 'rgba(240, 240, 240, 0.95)',
                         margin = dict(b = 195)
                        )
    fig['layout']['xaxis1'].update(dict(title="Prediction"))
    fig['layout']['yaxis1'].update(dict(title="Actual"))
    fig["layout"]["xaxis2"].update((dict(range=[0, 1])))
    fig['layout']['xaxis3'].update(dict(title="False Positive Rate"))
    fig['layout']['yaxis3'].update(dict(title='True Positive Rate'))
    fig["layout"]["xaxis4"].update(dict(title="recall"), 
                                   range=[0, 1.05])
    fig["layout"]["yaxis4"].update(dict(title="precision"), 
                                   range=[0, 1.05])
    fig['layout']['xaxis5'].update(dict(showgrid=True,
                                        tickfont=dict(size = 10),
                                        tickangle=90
                                       )
                                  )
    fig.layout.title.font.size = 14
    py.iplot(fig)
    return y_pred, y_prob


def _cumulative_gain_curve(y_true, y_score, pos_label):
    """
    Function to compute one cumulative gain curve: the share of all positives that is
      captured once the sample is ranked by score and read from the highest score down.
      This is the shared building block of the gain chart and the lift chart.
    @Args:
      y_true: the array of actual labels
      y_score: the array of predicted probabilities for the class of interest
      pos_label: the label value that counts as the positive class for this curve

    Return:
      percentages: the fraction of the sample read so far, starting at 0.
      gains: the fraction of all positives captured so far, starting at 0.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    is_pos = (y_true == pos_label)
    sorted_idx = np.argsort(y_score, kind='mergesort')[::-1]
    is_pos = is_pos[sorted_idx]

    gains = np.cumsum(is_pos) / float(np.sum(is_pos))
    percentages = np.arange(1, len(is_pos) + 1) / float(len(is_pos))
    # Prepend the origin so both curves start from (0, 0)
    return np.insert(percentages, 0, 0.), np.insert(gains, 0, 0.)


def plot_cumulative_gain(y_true, y_probas, title: str='Cumulative Gains Curve',
                         figsize: tuple=None, ax=None):
    """
    Function to plot the cumulative gains chart of a binary classifier.
      It answers "if I contact the top X% of employees by attrition score, what
      share of the
      leavers do I reach?". The diagonal baseline is what contacting people at
      random gives you.
      Written here with plain matplotlib rather than pulling in a charting dependency.
    @Args:
      y_true: the array of actual labels
      y_probas: the (n_samples, 2) array of predicted probabilities, as returned
        by predict_proba
      title: string to be title of the plot
      figsize: tuple of the figure size, passed straight to matplotlib
      ax: an existing matplotlib axes to draw on; a new one is created when this is None

    Return:
      The matplotlib axes holding the plot
    """
    y_true = np.asarray(y_true)
    y_probas = np.asarray(y_probas)
    classes = np.unique(y_true)
    if len(classes) != 2:
        print("ERROR: plot_cumulative_gain only supports 2 classes. Please recheck")
        return None

    pct_0, gains_0 = _cumulative_gain_curve(y_true, y_probas[:, 0], classes[0])
    pct_1, gains_1 = _cumulative_gain_curve(y_true, y_probas[:, 1], classes[1])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(pct_0, gains_0, lw=3, label='Class {}'.format(classes[0]))
    ax.plot(pct_1, gains_1, lw=3, label='Class {}'.format(classes[1]))
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Baseline')

    ax.set_title(title, fontsize='large')
    ax.set_xlabel('Percentage of sample', fontsize='medium')
    ax.set_ylabel('Gain', fontsize='medium')
    ax.set_xlim([0., 1.])
    ax.set_ylim([0., 1.])
    ax.tick_params(labelsize='medium')
    ax.grid(True)
    ax.legend(loc='lower right', fontsize='medium')
    return ax


def plot_lift_curve(y_true, y_probas, title: str='Lift Curve',
                    figsize: tuple=None, ax=None):
    """
    Function to plot the lift chart of a binary classifier.
      Lift is the cumulative gain divided by the share of the sample read, so it reads as
      "how many times better than random is the model over the top X%". A lift of
      1 is random.
      Written here with plain matplotlib rather than pulling in a charting dependency.
    @Args:
      y_true: the array of actual labels
      y_probas: the (n_samples, 2) array of predicted probabilities, as returned
        by predict_proba
      title: string to be title of the plot
      figsize: tuple of the figure size, passed straight to matplotlib
      ax: an existing matplotlib axes to draw on; a new one is created when this is None

    Return:
      The matplotlib axes holding the plot
    """
    y_true = np.asarray(y_true)
    y_probas = np.asarray(y_probas)
    classes = np.unique(y_true)
    if len(classes) != 2:
        print("ERROR: plot_lift_curve only supports 2 classes. Please recheck")
        return None

    percentages, gains_0 = _cumulative_gain_curve(y_true, y_probas[:, 0], classes[0])
    _, gains_1 = _cumulative_gain_curve(y_true, y_probas[:, 1], classes[1])

    # Drop the origin, lift is undefined when no sample has been read yet
    percentages = percentages[1:]
    gains_0 = gains_0[1:] / percentages
    gains_1 = gains_1[1:] / percentages

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(percentages, gains_0, lw=3, label='Class {}'.format(classes[0]))
    ax.plot(percentages, gains_1, lw=3, label='Class {}'.format(classes[1]))
    ax.plot([0, 1], [1, 1], 'k--', lw=2, label='Baseline')

    ax.set_title(title, fontsize='large')
    ax.set_xlabel('Percentage of sample', fontsize='medium')
    ax.set_ylabel('Lift', fontsize='medium')
    ax.tick_params(labelsize='medium')
    ax.grid(True)
    ax.legend(loc='lower left', fontsize='medium')
    return ax


def gen_pred_band (val):
    '''
    Function to add the data band to the data
    Args:
      val: series of number range from [0.0 - 1.0]
    
    Return:
      this: the string of data band
    '''
    if(val <= 0.05):
        this = '001. [0.00 - 0.05]'
    elif (val <= 0.10):
        this = '002. (0.05 - 0.10]'
    elif (val <= 0.15):
        this = '003. (0.10 - 0.10]'
    elif (val <= 0.20):
        this = '004. (0.15 - 0.20]'
    elif (val <= 0.25):
        this = '005. (0.20 - 0.25]'
    elif (val <= 0.30):
        this = '006. (0.25 - 0.30]'
    elif (val <= 0.35):
        this = '007. (0.30 - 0.35]'
    elif (val <= 0.40):
        this = '008. (0.35 - 0.40]'
    elif (val <= 0.45):
        this = '009. (0.40 - 0.45]'
    elif (val <= 0.50):
        this = '010. (0.45 - 0.50]'
    elif (val <= 0.55):
        this = '011. (0.50 - 0.55]'
    elif (val <= 0.60):
        this = '012. (0.55 - 0.60]'
    elif (val <= 0.65):
        this = '013. (0.60 - 0.65]'
    elif (val <= 0.70):
        this = '014. (0.65 - 0.70]'
    elif (val <= 0.75):
        this = '015. (0.70 - 0.75]'
    elif (val <= 0.80):
        this = '016. (0.75 - 0.80]'
    elif (val <= 0.85):
        this = '017. (0.80 - 0.85]'
    elif (val <= 0.90):
        this = '018. (0.85 - 0.90]'
    elif (val <= 0.95):
        this = '019. (0.90 - 0.95]'
    else:
        this = '020. (0.95 - 1.00]'
        
    return this

def prepareDeciles(
    df,probability_col='probability',decile_columns='deciles',inplace=True
):
    """
    Function to get deciles from probability values
    
    @Arguments:
      df = A pandas dataframe with atleast one probability columns
      probability_col = name of probability column
      decile_columns = used only if inplace=True; name of decile columns
      inplace = append directly in df
    
    @Returns:
      Deciles
    """
    _,bins = pd.qcut(df[probability_col],10,retbins=True,duplicates='drop')
    bins[0] -= 0.001
    bins[-1] += 0.001
    bins_labels = ['%d'%(9-x[0]) for x in enumerate(zip(bins[:-1],bins[1:]))]
    bins_labels[0] = bins_labels[0].replace('(','[')
    if inplace:
        df[decile_columns]=pd.cut(df[probability_col],bins=bins,labels=bins_labels)
    else:
        return pd.cut(df[probability_col],bins=bins,labels=bins_labels)
    
    
def get_deciles_analysis(df,score="prob",target="actual"):
    """
    *Deprecated use decile_analysis instead*
    Get decile analysis; see distibution of events(ones) and non-events(zeros)
    on different deciles
    
    Arguments:
    df = A pandas dataframe with atleast two columns one with calculated
         probabilities using model and another with  actual label
    score = name of probability column
    target = name of actual columns
    
    Returns:
    Decile analysis dataframe
    
    """
    df1 = df[[score,target]].dropna()
    _,bins = pd.qcut(df1[score],10,retbins=True,duplicates='drop')
    bins[0] -= 0.001
    bins[-1] += 0.001
    bins_labels = [
        '%d.(%0.2f,%0.2f]'%(9-x[0],x[1][0],x[1][1])
        for x in enumerate(zip(bins[:-1],bins[1:]))
    ]
    bins_labels[0] = bins_labels[0].replace('(','[')
    df1['Decile']=pd.cut(df1[score],bins=bins,labels=bins_labels)
    df1['Population']=1
    df1['Zeros']=1-df1[target]
    df1['Ones']=df1[target]
    summary=df1.groupby(['Decile'])[['Ones','Zeros','Population']].sum()
    summary=summary.sort_index(ascending=False)
    summary['TargetRate']=summary['Ones']/summary['Population']
    summary['CumulativeTargetRate']=(
        summary['Ones'].cumsum()/summary['Population'].cumsum()
    )
    summary['TargetsCaptured']=summary['Ones'].cumsum()/summary['Ones'].sum()
    return summary


def decile_analysis(estimator, X, Y):
    """
    Get decile analysis; see distibution of events(ones) and non-events(zeros)
    on different deciles
    
    Arguments:
    estimator = model with which probabilities will be calculated
    X = X variables for scoring
    Y = target labels
    
    Returns:
    Decile analysis dataframe
    
    By:
    Cognizant-Aetna Team
    """
    return get_deciles_analysis(
        pd.DataFrame({"prob":estimator.predict_proba(X)[:,1],"actual":Y})
    )


if __name__ == '__main__':
    print('Run function..')