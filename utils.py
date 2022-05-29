import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics

import plotly.offline as py
import plotly.graph_objs as go     



def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def NRMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())/(np.max(targets)-np.min(targets))*100


def SMAPE(F, A):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def MAE(F, A):
    return 1/len(A) * np.sum(np.abs(F - A))


def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


########################### Helpers #################################################################################
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    
## Multiprocess Runs
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df


def Test_evaluation(df, name):
    print('Hybrid approach {}: RMSE is {}, MAPE is {}, MAE is {}, NRMSE is {}, R^2 is {}'.format(name,
                                                             '%.4f' % RMSE(df['Record'],df['Hybrid pred']), 
                                                             '%.4f' % MAPE(df['Hybrid pred'],df['Record']), 
                                                             '%.4f' % MAE(df['Record'],df['Hybrid pred']),
                                                             '%.4f' % NRMSE(df['Record'],df['Hybrid pred']),
                                                             '%.4f' % metrics.r2_score(df['Hybrid pred'],df['Record'])))
    print('Pure ML {}: RMSE is {}, MAPE is {}, MAE is {}, NRMSE is {}, R^2 is {}'.format(name,
                                                             '%.4f' % RMSE(df['Record'],df['Pure ML pred']), 
                                                             '%.4f' % MAPE(df['Pure ML pred'],df['Record']), 
                                                             '%.4f' % MAE(df['Record'],df['Pure ML pred']),
                                                             '%.4f' % NRMSE(df['Record'],df['Pure ML pred']),
                                                             '%.4f' % metrics.r2_score(df['Pure ML pred'],df['Record'])))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
          x = df['Time'],
          y = df['Record'],
          mode = 'lines',
          name = 'Record'   
    ))
    fig.add_trace(go.Scatter(
          x = df['Time'],
          y = df['Simu_Record'],
          mode = 'lines',
          name = 'Simu_Record'   
    ))
    fig.add_trace(go.Scatter(
          x = df['Time'],
          y = df['Hybrid pred'],
          mode = 'lines',
          name = 'Hybrid approach'   
    ))
    fig.add_trace(go.Scatter(
          x = df['Time'],
          y = df['Pure ML pred'],
          mode = 'lines',
          name = 'Pure ML'   
    ))
    fig.update_layout(
        title=name,
        xaxis_title="Time",
        yaxis_title="Load (kWh)",
        font=dict(size=16)
    )

    fig.show()
    

def evaluation(df, name, TARGET='Simu_Record'):
    print('Performance evaluation {}: RMSE is {}, MAPE is {}, MAE is {}, NRMSE is {}, R^2 is {}'.format(name,
                                                             '%.4f' % RMSE(df['Record'],df[TARGET]), 
                                                             '%.4f' % MAPE(df[TARGET],df['Record']), 
                                                             '%.4f' % MAE(df['Record'],df[TARGET]),
                                                             '%.4f' % NRMSE(df['Record'],df[TARGET]),
                                                             '%.4f' % metrics.r2_score(df[TARGET],df['Record'])))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
          x = df['Time'],
          y = df['Record'],
          mode = 'lines',
          name = 'Record'   
    ))
    fig.add_trace(go.Scatter(
          x = df['Time'],
          y = df[TARGET],
          mode = 'lines',
          name = TARGET   
    ))
    fig.update_layout(
        title=name,
        xaxis_title="Time",
        yaxis_title="Load (kWh)",
        font=dict(size=16)
    )

    fig.show()