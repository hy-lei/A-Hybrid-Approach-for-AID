import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

def RMSE(X, Y):
    """
        X: is a 1d array storing traffic parameters for one station for each time t
        Y: 1d array with traffic parameters forecast for each time t
    """
    return np.sqrt(np.mean((Y - X) ** 2))

def RMSE_numerical(X, Y):
    M, N = X.shape
    return np.sqrt(np.sum((Y - X) ** 2) / N)

def S1(X, prev_S1, alpha):
    """
        Order 1 smoothing
    """
    return alpha * X + (1 - alpha) * prev_S1

def S2(S1, prev_S2, alpha):
    """
        Order 2 smoothing
    """
    return alpha * S1 + (1 - alpha) * prev_S2

def A(S1, S2):
    return 2 * S1 - S2

def B(S1, S2, alpha):
    return 1 / (1 - alpha) * (S1 - S2)

def plot_time_series(X, Y, X_label=None, Y_label=None):
    num, = X.shape
    f, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(range(num), X, lw=1, color="black", label=X_label)
    ax.plot(range(num), Y, lw=1, color="red", label=Y_label)
    ax.set_xlim(0, num)
    ax.set_ylim(0, max(Y))
    ax.set_xticklabels(ax.get_xticks(), fontsize=10)
    ax.set_yticklabels(ax.get_yticks(), fontsize=10)
    ax.legend(fontsize=10, loc=0)
    plt.show()

def visualize_all(X, Y):
    plot_time_series(X[0], Y[0], "Occupancy", "Prediction")
    plot_time_series(X[1], Y[1], "Speed", "Prediction")
    plot_time_series(X[2], Y[2], "Flow", "Prediction")

def find_ups_downs(abs_pm, stations_abs_pm):
    num_stations = stations_abs_pm.size
    upstream = None
    downstream = None
    for i in range(num_stations):
        if stations_abs_pm[i] > abs_pm:
            downstream = i
            upstream = i - 1 if i >= 1 else None
            break
    if downstream == None:
        upstream = num_stations - 1
    return (upstream, downstream)

def df_to_X(train_df, feature_list):
    num_timestamp = train_df["Date"].values.size
    X = np.ndarray((3, num_timestamp))
    for i, feature in enumerate(feature_list):
        X[i, :] = train_df[feature].values
    return X

def TSA_construct_X(incidents, data, station_id, feature_list, start_date, end_date):
    days = data["Date"].unique()
    days = [day for day in days if (day >= start_date and day < end_date)]
    # check affected days
    affected_incidents = incidents.loc[incidents["Upstream"].isin([station_id]) | incidents["Downstream"].isin([station_id])]
    affected_days = affected_incidents["Date"].unique()
    # get unaffected days for station i
    train_date = [day for day in days if day not in affected_days]
    
    normal_train_data = data.loc[data["Station ID"].isin([station_id]) & data["Date"].isin(train_date)]
    
    return df_to_X(normal_train_data, feature_list)

def TSA_train(X):
    """
        X: feature matrix saving traffic variables across time, for each time, there are three variables - 
           speed, flow and occupancy
           dimension: n * m, where n = number of variables, m = number of data points
           
        return: best snapshot, and best alpha vector
    """
    RMSEs = np.zeros((3, 1000))
    alphas = np.array(range(1000)) * 0.001
    num_timestamp = X[0].size
    num_var = X[:, 0].size
    min_idx = [-1, -1, -1]
    best_S1 = np.zeros(num_timestamp)
    best_S2 = np.zeros(num_timestamp)
    best_Y = np.zeros((num_var, num_timestamp))
    best_alpha = np.zeros(num_var)
    
    for i in range(num_var):
        min_rmse_i = 1000
        print("Variable " + str(i) + ":")
        for j, alpha in enumerate(alphas):
            S1_j = np.zeros(num_timestamp)
            S2_j = np.zeros(num_timestamp)
            Y_j = np.zeros(num_timestamp)
            S1_j[9] = np.mean(X[i, 0:9])
            for k in range(5):
                S2_j[9] += (X[i, 2 * k + 1] - X[i, 2 * j])
            S2_j[9] /= 5.
            
            for t in range(num_timestamp - 1):
                if t > 9:
                    S1_j[t] = S1(X[i, t], S1_j[t-1], alpha)
                    S2_j[t] = S2(S1_j[t], S2_j[t-1], alpha)
                    Y_j[t+1] = A(S1_j[t], S2_j[t]) + B(S1_j[t], S2_j[t], alpha)
            
            # Compute RMSE for j-th alpha
            RMSEs[i, j] = RMSE(X[i, 10:], Y_j[10:])
            print(str(alpha) + ": " + str(RMSEs[i, j]))
            if min_rmse_i > RMSEs[i, j]:
                min_rmse_i = RMSEs[i, j]
                min_idx[i] = j
                best_S1 = S1_j
                best_S2 = S2_j
                best_Y[i, :] = copy.deepcopy(Y_j)
                best_alpha[i] = alpha
    
    return best_alpha, best_S1, best_S2, best_Y

# proprocessing functions
def filter_holes(concat, txt_path, csv_path):
    """
       concat: a pandas dataframe object having at least the following columns:
               - Station ID: name of a station
               - datetime: date and time of the observation
       txt_path: the path of the output text file.  The text file contains information about,
                 for each station,
                 - hole between two dates
                 - hole between different time within the same day
       csv_path: the path of the output csv file.  The csv file has the following columns:
                 - Station ID
                 - start date: has value iff type of record is 'd'
                 - end date: has value iff type of record is 'd'
                 - start time: has value iff type of record is 't'
                 - end time: has value iff type of record is 't'
    """
    output_str = ""
    stations_arr = []
    start_date_arr = []
    end_date_arr = []
    start_time_arr = []
    end_time_arr = []
    type_arr = []
    
    stations = concat["Station ID"].unique()
    for station in stations:
        print(station)
        output_str += str(station) + "\n"
        substr_date = ""
        substr_min = ""
        df = concat.loc[concat["Station ID"] == station]
        prev_date_time = None
        for date_time in df["datetime"].values:
            if prev_date_time is not None:
                if date_time != prev_date_time + np.timedelta64(5, 'm'):
                    prev_date = pd.Timestamp(prev_date_time).date()
                    current_date = pd.Timestamp(date_time).date()
                    stations_arr.append(station)
                    
                    if prev_date != current_date:
                        substr_date += str(prev_date) + " " + str(current_date) + "\n"
                        
                        start_date_arr.append(str(prev_date))
                        end_date_arr.append(str(current_date))
                        start_time_arr.append(None)
                        end_time_arr.append(None)
                        type_arr.append('d')
                        
                        print(prev_date, current_date)
                    else:
                        substr_min += str(prev_date_time) + " " + str(date_time) + "\n"
                        
                        start_date_arr.append(prev_date)
                        end_date_arr.append(current_date)
                        start_time_arr.append(prev_date_time)
                        end_time_arr.append(date_time)
                        type_arr.append('t')
                        
                        print(prev_date_time, date_time)
            prev_date_time = date_time
        output_str += "date:\n" + substr_date + "\nmin:\n" + substr_min + "\n"
    
    report_df = pd.DataFrame({
        'Station ID': stations_arr,
        'start date': start_date_arr,
        'end date': end_date_arr,
        'start time': start_time_arr,
        'end time': end_time_arr,
        'type': type_arr
    })
    
    if txt_path != None:
        with open(txt_path, "w") as text_file:
            print(output_str, file=text_file)
    
    if csv_path != None:
        report_df.to_csv(csv_path, index=False)
    
    return report_df
