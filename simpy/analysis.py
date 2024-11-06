import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot

file_name =  ""
data_path = ""
picture_path = ""
bco_d = pd.read_csv(f'{data_path}/{file_name}')
move_type_list = ['RAIL', 'TRUCK']
cont_type_list = ['dry', 'reefer']
period_list = [7, 30, 'W', 'M']
#bco_d = bco_d[bco_d['DATE_OCCURRED'] == '2024-10-25']
#bco_d = bco_d[bco_d['CONTAINER_MOVEMENT'] == 'RAIL']

def extend_df(df, value_column, date_column, period):
    #df = df.set_index(date_column)  # Replace 'date' with the name of your date column
    df = df.asfreq(period)  # Ensure it's in weekly frequency

    # Number of additional weeks needed
    n_extend = (104 if period == 'W' else 24) - len(df)
    if period == 'W':
        offset = pd.Timedelta(weeks=1)
    elif period == 'M':
        offset = pd.DateOffset(months=1)
    # Extend the index by 15 weeks
    future_dates = pd.date_range(df.index[-1] + offset, periods=n_extend, freq=period)
    future_df = pd.DataFrame(index=future_dates)

    # Use the last known value to fill the future values, or apply a rolling mean if more appropriate
    future_df[value_column] = df[value_column].iloc[-1]  # or use a more sophisticated forecast method
    print("future_df columns are ", future_df.columns)
    # Combine original and future data
    return pd.concat([df, future_df])
def decompose(df, value_column, date_column, period, move_type, cont_type, show_plot = False, show_autocorrelation = False):
    df_sample = df[[value_column, date_column]].copy()
    use_period = period
    if period in ['W','M']:        
        df_sample = df_sample.resample(period, on=date_column).sum().copy()
        use_period = 12 if period == 'M' else 52
        df_sample = extend_df(df_sample, value_column, date_column, period)
        df_sample = df_sample.reset_index()        
        #print("df_sample columns are ", df_sample.columns)
        df_sample.rename(columns={'index': date_column}, inplace=True)
    decomposition = seasonal_decompose(df_sample[value_column], model='additive', period=use_period)  # Adjust 'period' based on data frequency
# Extract components
    df_sample['seasonal'] = decomposition.seasonal
    df_sample['trend'] = decomposition.trend
    df_sample['residual'] = decomposition.resid
    if show_autocorrelation:
        plt.figure(figsize=(10, 6))
        autocorrelation_plot(df_sample[value_column])
        plt.show()

    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    ax[0].plot(df_sample[date_column], df_sample[value_column], label= f'Original')
    ax[0].set_title(f'Original Series of {move_type} and {cont_type} and {period}')

    ax[1].plot(df_sample[date_column], df_sample['trend'], label='Trend')
    ax[1].set_title('Trend Component')

    ax[2].plot(df_sample[date_column], df_sample['seasonal'], label='Seasonality')
    ax[2].set_title('Seasonal Component')

    ax[3].plot(df_sample[date_column], df_sample['residual'], label='Residual')
    ax[3].set_title('Residual Component')

    plt.tight_layout()
    plt.savefig(f"{picture_path}/{move_type}_{cont_type}_{period}.jpg", format="jpg", dpi=300)
    if show_plot: 
        plt.show()

def plot_grouped_data(df, date_column, group_columns, agg_columns):
    # Ensure the date column is parsed as a datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Number of subplots needed
    n_plots = len(agg_columns)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 5 * n_plots), sharex=True)

    # Convert axes to a list for consistency if there's only one subplot
    if n_plots == 1:
        axes = [axes]

    # Plot each aggregation column in its own subplot
    for i, agg_col in enumerate(agg_columns):
        # Loop through each unique group in group_columns
        for group_keys, group_data in df.groupby(group_columns):
            # Convert group_keys to a label if there are multiple group columns
            label = ', '.join(str(k) for k in group_keys) if isinstance(group_keys, tuple) else str(group_keys)
            axes[i].plot(group_data[date_column], group_data[agg_col], label=label)
        
        # Set title, labels, and legend
        axes[i].set_title(f'Time Series of {agg_col}')
        axes[i].set_ylabel(agg_col)
        axes[i].legend(title="Categories")
    
    # Set x-axis label
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.show()



#group_columns = ['DATE_OCCURRED','MERCHANT_OR_CARRIER_HAULAGE', 'CONTAINER_MOVEMENT', 'END_DESTINATION_ZIP3', 'RAIL_TERMINAL_CBP_D', 'CONTAINER_TYPE']
group_columns = ['DATE_OCCURRED', 'CONTAINER_MOVEMENT', 'CONTAINER_TYPE']
bco_grouped = bco_d.groupby(group_columns).agg(**{
    'TEUS' : ('TEUS', 'sum'),
    'CONTAINER_20_FT' : ('CONTAINER_20_FT', 'sum'),
    'CONTAINER_40_FT': ('CONTAINER_40_FT', 'sum')
}).reset_index()
bco_grouped['DATE_OCCURRED'] = pd.to_datetime(bco_grouped['DATE_OCCURRED'])

#df['trend_rolling_mean'] = df['value_column'].rolling(window=12).mean() 

#plot_grouped_data(bco_grouped, 'DATE_OCCURRED', ['CONTAINER_MOVEMENT', 'CONTAINER_TYPE'], ['TEUS', 'CONTAINER_20_FT', 'CONTAINER_40_FT'])
for period in period_list:
    for move_type in move_type_list:
        decompose(bco_grouped[(bco_grouped['CONTAINER_MOVEMENT'] == move_type)], \
                    'TEUS', 'DATE_OCCURRED', period, move_type, None)
        for cont_type in cont_type_list:
            decompose(bco_grouped[(bco_grouped['CONTAINER_MOVEMENT'] == move_type) & (bco_grouped['CONTAINER_TYPE'] == cont_type)], \
                    'TEUS', 'DATE_OCCURRED', period, move_type, cont_type)
    for cont_type in cont_type_list:
        decompose(bco_grouped[(bco_grouped['CONTAINER_TYPE'] == cont_type)], \
                    'TEUS', 'DATE_OCCURRED', period, None, cont_type)


#print(bco_grouped.head(1))