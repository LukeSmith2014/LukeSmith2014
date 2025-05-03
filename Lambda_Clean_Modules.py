import pandas as pd
import numpy as np
import re
import os
import boto3
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the column names of a pandas DataFrame by:
    - Converting to lowercase
    - Replacing spaces and special characters with underscores
    - Removing leading/trailing underscores

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: DataFrame with standardized column names
    """
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame

    df.columns = [re.sub(r'[^0-9a-zA-Z]+', '_', col).strip('_').lower() for col in df.columns]
    print(df.columns)
    return df


def create_key_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new feature 'key' by concatenating 'equipment_id' and 'booking_number',
    ensuring both 'booking_number' features are int64.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: DataFrame with the new 'key' feature
    """
    df = df.copy()
    if 'equipment_id' in df.columns and 'booking_number' in df.columns:
        df['booking_number'] = pd.to_numeric(df['booking_number'], errors='coerce').astype('Int64')
        df['key'] = df['equipment_id'].astype(str) + '_' + df['booking_number'].astype(str)
    return df


def map_on_key(df1: pd.DataFrame, df2: pd.DataFrame, key_col) -> pd.DataFrame:
    """
    Maps all columns from df2 onto df1 based on the 'key' feature instead of merging.
    Handles duplicate keys in df2 by taking the first occurrence.

    Parameters:
    df1 (pd.DataFrame): First DataFrame (Base DataFrame).
    df2 (pd.DataFrame): Second DataFrame (Data to be mapped onto df1).
    key_col (str): Column name used for mapping.

    Returns:
    pd.DataFrame: df1 with additional columns from df2 mapped onto it.
    """
    df1 = df1.copy()

    # Remove duplicates in df2 based on key_col, keeping the first occurrence
    df2 = df2.drop_duplicates(subset=[key_col], keep='first')

    # Loop through each column in df2 (excluding key_col) and map it to df1
    for col in df2.columns:
        if col != key_col:
            df1[col] = df1[key_col].map(df2.set_index(key_col)[col])

    return df1


def create_voyage_code(df, vessel_col, voyage_col, new_feature_name="voyage_code"):
    """
    Combines three features into a single concatenated feature called 'voyage_code'.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    vessel_col (str): Column name for vessel code.
    voyage_col (str): Column name for voyage.
    new_feature_name (str): Name of the new concatenated feature (default is 'voyage_code').

    Returns:
    pd.DataFrame: The DataFrame with the new 'voyage_code' feature added.
    """

    df[new_feature_name] = df[vessel_col].astype(str) + "_" + df[voyage_col].astype(str) + "_" + "E"
    return df


def add_next_voyage(df, feature_name):
    """
    Adds a new feature 'next_voyage' to the DataFrame by adding 1 to the specified feature.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    feature_name (str): The column name to increment.

    Returns:
    pd.DataFrame: The DataFrame with the new 'next_voyage' column.
    """
    # Convert to numeric and coerce errors to NaN
    df[feature_name] = pd.to_numeric(df[feature_name], errors='coerce')

    # Add 1 and cast to int, NaNs will remain
    df['next_voyage'] = (df[feature_name] + 1).astype('Int64')

    return df


def create_voyage_code_1(df, vessel_col, voyage_col, direction_col, new_feature_name="voyage_code"):
    """
    Combines three features into a single concatenated feature called 'voyage_code'.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    vessel_col (str): Column name for vessel code.
    voyage_col (str): Column name for voyage.
    direction_col (str): Column name for direction sequence.
    new_feature_name (str): Name of the new concatenated feature (default is 'voyage_code').

    Returns:
    pd.DataFrame: The DataFrame with the new 'voyage_code' feature added.
    """

    df[new_feature_name] = df[vessel_col].astype(str) + "_" + df[voyage_col].astype(str) + "_" + df[direction_col].astype(str)
    return df


def add_arrival_time(main_df, ref_df):
    """
    Adds an 'arrival_time' column to main_df by mapping values from ref_df based on 'voyage_code',
    without adding extra rows. Handles duplicate 'voyage_code' in ref_df by taking the first occurrence.

    Parameters:
    main_df (pd.DataFrame): The main DataFrame that needs the 'arrival_time' column.
    ref_df (pd.DataFrame): The reference DataFrame that contains 'voyage_code' and 'arrival_timestamp'.

    Returns:
    pd.DataFrame: The updated main DataFrame with 'arrival_time' added.
    """
    # Remove duplicates in ref_df based on voyage_code, keeping the first occurrence
    ref_df = ref_df.drop_duplicates(subset=['voyage_code'], keep='first')

    # Create a mapping dictionary from ref_df
    voyage_to_arrival = ref_df.set_index('voyage_code')['arrival_timestamp']

    # Map arrival timestamps to main_df based on voyage_code
    main_df['arrival_time'] = main_df['voyage_code'].map(voyage_to_arrival)

    return main_df


def remove_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Removes specified columns from the dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    columns (list): List of column names to be removed.

    Returns:
    pd.DataFrame: The dataframe with specified columns removed.
    """
    df = df.copy()
    return df.drop(columns=columns, errors='ignore')


def standardize_datetime_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Converts specified columns to a standardized datetime format while maintaining datetime dtype.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to be converted to datetime format

    Returns:
    pd.DataFrame: DataFrame with specified datetime columns standardized
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def add_pickup_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the 'event_date' for the record where 'event_type' is 'OGT' and 'inventory_location_code' is 'STE1' or 'STE2'.
    This date is added as 'pickup_date' for each unique 'key' value and mapped back onto the dataframe.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: DataFrame with 'pickup_date' mapped onto 'key'
    """
    df = df.copy()

    # Filter to find the relevant pickup dates
    pickup_dates = df[(df['event_type'] == 'OGT') & (df['inventory_location_code'].isin(['STE1', 'STE2']))]

    # Create a mapping of 'key' -> 'pickup_date'
    pickup_date_mapping = pickup_dates.groupby('key')['event_date'].first()  # Use .first() in case of duplicates

    # Map pickup dates back to the original dataframe
    df['pickup_date'] = df['key'].map(pickup_date_mapping)

    return df


def count_hlp_occurrences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts the number of times 'HLP' appears in the 'event_type' feature for each unique 'key'.
    Adds this count as a new feature 'hlp_count' using mapping instead of merging.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: DataFrame with 'hlp_count' mapped on 'key'
    """
    df = df.copy()

    # Count occurrences of 'HLP' for each key
    hlp_count_mapping = df[df['event_type'] == 'HLP'].groupby('key').size()

    # Map the counts back to the original DataFrame
    df['hlp_count'] = df['key'].map(hlp_count_mapping).fillna(0).astype('Int64')

    return df


def calculate_storage_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates 'storage_time' as the difference in hours between the 'event_date' of the record where 'event_type' is 'AVP'
    and the 'pickup_date'. Maps the calculated 'storage_time' back onto the DataFrame based on 'key'.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: DataFrame with 'storage_time' added
    """
    df = df.copy()

    # Filter to find AVP event dates
    avp_dates = df[df['event_type'] == 'AVP']

    # Create a mapping of 'key' -> 'avp_date'
    avp_date_mapping = avp_dates.groupby('key')['event_date'].first()  # Use .first() in case of duplicates

    # Map AVP dates back to the original DataFrame
    df['avp_date'] = df['key'].map(avp_date_mapping)

    # Calculate storage time in hours
    df['storage_time'] = (df['pickup_date'] - df['avp_date']).dt.total_seconds() / 3600

    return df


def add_day_of_week(df, datetime_col):
    """
    Takes a DataFrame and a column name containing datetime values,
    and creates a new column with the corresponding day of the week.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    datetime_col (str): The column name containing datetime values.

    Returns:
    pd.DataFrame: DataFrame with a new 'day_of_week' column.
    """
    df[datetime_col] = pd.to_datetime(df[datetime_col])  # Ensure it's in datetime format
    df['avp_day_of_week'] = df[datetime_col].dt.day_name()  # Extract day of the week
    return df


def calculate_time_til_avp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates 'time_til_avp' as the difference in hours

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: DataFrame with 'time_til_avp' added
    """
    df = df.copy()
    df['time_til_avp'] = (df['avp_date'] - df['arrival_time']).dt.total_seconds() / 3600
    return df


def add_time_to_unload_feature(df, key_col, arrival_col, event_col, event_type, event_date_col):
    """
    Adds a new feature 'time_to_unload' (in hours) to the dataframe, which represents the time difference
    between 'arrival_time' and the most recent 'event_date' for each unique 'key'
    where 'event_type' == specified type.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    key_col (str): Column name representing the unique key.
    arrival_col (str): Column name for arrival timestamps.
    event_col (str): Column name for event type.
    event_type (str): The specific event type to filter on (e.g., 'DFV').
    event_date_col (str): Column name for event timestamps.

    Returns:
    pd.DataFrame: Original DataFrame with new 'time_to_unload' feature in hours.
    """

    df = df.copy()

    # Convert timestamps to datetime format if not already
    df[arrival_col] = pd.to_datetime(df[arrival_col])
    df[event_date_col] = pd.to_datetime(df[event_date_col])

    # Filter rows where event_type matches the specified event_type
    df_filtered = df[df[event_col] == event_type]

    # Get the most recent event_date for each key
    latest_event_mapping = df_filtered.groupby(key_col)[event_date_col].max()

    # Map the latest event date back to the original dataframe
    df['dfv_event_date'] = df[key_col].map(latest_event_mapping)

    # Calculate the time difference in hours
    df['time_to_unload'] = (df['dfv_event_date'] - df[arrival_col]).dt.total_seconds() / 3600

    return df


def arrival_to_pickup_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates 'arrival_to_pickup_time' by taking the difference between 'pickup_date' and 'arrival_time' in hours.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: DataFrame with 'arrival_to_pickup_time' added
    """
    df = df.copy()
    df['arrival_to_pickup_time'] = (df['pickup_date'] - df['arrival_time']).dt.total_seconds() / 3600
    return df


def port_to_ste_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates 'port_to_ste_time' by taking the difference between 'time_til_avp' and 'time_to_unload' in hours.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: DataFrame with 'port_to_ste_time' added
    """
    df = df.copy()
    df['port_to_ste_time'] = (df['time_til_avp'] - df['time_to_unload'])
    return df


def remove_negative_values(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Removes all rows where any of the specified feature columns contain negative values.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    feature_names (list): List of feature column names to check for negative values.

    Returns:
    pd.DataFrame: DataFrame with rows containing negative values removed.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    return df[(df[feature_names] >= 0).all(axis=1)]


def add_load_date(df, key_col, event_col, event_type, event_date_col):
    """
    Adds a new feature 'load_date', which represents the time/date at which a
    container was lifted to its vessel.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    key_col (str): Column name representing the unique key.
    event_col (str): Column name for event type.
    event_type (str): The specific event type to filter on (e.g., 'DFV').
    event_date_col (str): Column name for event timestamps.

    Returns:
    pd.DataFrame: Original DataFrame with new 'load_date' feature.
    """

    df = df.copy()

    # Convert timestamps to datetime format if not already
    df[event_date_col] = pd.to_datetime(df[event_date_col])

    # Filter rows where event_type matches the specified event_type
    df_filtered = df[df[event_col] == event_type]

    # Get the most recent event_date for each key
    latest_event_mapping = df_filtered.groupby(key_col)[event_date_col].max()

    # Map the latest event date back to the original dataframe
    df['load_date'] = df[key_col].map(latest_event_mapping)

    # Convert this new feature to date time format
    df['load_date'] = pd.to_datetime(df['load_date'])

    return df


def create_binary_feature(df, feature_name, threshold, new_feature_name):
    """
    Creates a new binary feature in the DataFrame based on whether a given feature's value is less than a threshold.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    feature_name (str): The column name to evaluate.
    threshold (float or int): The threshold value.
    new_feature_name (str): The name of the new binary feature.

    Returns:
    pd.DataFrame: The DataFrame with the new binary feature added.
    """
    df[new_feature_name] = (df[feature_name] < threshold).astype(int)
    return df


def remove_null_records(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Removes all records with null values in the specified columns.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    columns (list): List of column names to check for null values.

    Returns:
    pd.DataFrame: The dataframe with rows containing null values in the specified columns removed.
    """
    df = df.copy()
    return df.dropna(subset=columns)


def calculate_journey_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the time (in hours) between 'sail_date' and 'avp_date' and creates a new feature 'journey_length'.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: The dataframe with the new 'journey_length' feature.
    """
    df = df.copy()

    # Ensure 'sail_date' and 'avp_date' are in datetime format
    df['sail_date'] = pd.to_datetime(df['sail_date'], errors='coerce')
    df['avp_date'] = pd.to_datetime(df['avp_date'], errors='coerce')

    # Calculate the difference in hours between 'sail_date' and 'avp_date'
    df['journey_length'] = (df['avp_date'] - df['sail_date']).dt.total_seconds() / 3600

    return df


def fill_null_values(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """
    Fills null values in a specified column with a given value.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    column (str): The column name in which to fill null values.
    value: The value to replace nulls with.

    Returns:
    pd.DataFrame: The dataframe with null values replaced in the specified column.
    """
    df = df.copy()
    df[column] = df[column].fillna(value)
    return df


def apply_one_hot_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applies one-hot encoding to the specified categorical columns and then converts the newly created boolean
    columns to integers.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    columns (list): List of categorical column names to be one-hot encoded.

    Returns:
    pd.DataFrame: The dataframe with one-hot encoded features, with booleans converted to integers.
    """
    df = df.copy()

    # Apply one-hot encoding to specified columns
    df = pd.get_dummies(df, columns=columns, drop_first=True)

    # Convert the newly created boolean columns to integers
    df = df.applymap(lambda x: 1 if x is True else (0 if x is False else x))

    return df


def target_encode(df, feature_list, target_col, alpha=10):
    """
    Applies target encoding to a list of categorical features.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    feature_list (list): List of categorical column names to be target encoded.
    target_col (str): Name of the target variable.
    alpha (int): Smoothing parameter to reduce overfitting (higher values give more weight to global mean).

    Returns:
    pd.DataFrame: DataFrame with target-encoded features.
    """
    df_encoded = df.copy()
    global_mean = df[target_col].mean()

    for feature in feature_list:
        category_stats = df.groupby(feature)[target_col].agg(['mean', 'count'])
        category_stats['smooth'] = (category_stats['count'] * category_stats['mean'] + alpha * global_mean) / (category_stats['count'] + alpha)

        # Replace categorical values with smoothed target encoding
        df_encoded[feature] = df_encoded[feature].map(category_stats['smooth'])

    return df_encoded


def binary_encode(df, categorical_features):
    """
    Apply binary encoding to specified categorical features in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        categorical_features (list): List of column names to apply binary encoding.

    Returns:
        pd.DataFrame: A new DataFrame with binary-encoded features.
    """
    df_encoded = df.copy()

    for feature in categorical_features:
        # Convert categorical values to unique integer labels
        df_encoded[feature + "_int"] = df_encoded[feature].astype('category').cat.codes

        # Find the number of bits required to encode the max integer
        max_val = df_encoded[feature + "_int"].max()
        num_bits = int(max_val).bit_length()  # Compute required bits

        # Convert integers to binary and create separate columns for each bit
        binary_columns = df_encoded[feature + "_int"].apply(lambda x: list(format(x, f'0{num_bits}b')))

        # Create new column names
        binary_col_names = [f"{feature}_bin{i}" for i in range(num_bits)]

        # Expand binary values into separate columns
        df_encoded[binary_col_names] = pd.DataFrame(binary_columns.tolist(), index=df_encoded.index)

        # Drop the original and integer-encoded columns
        df_encoded.drop(columns=[feature, feature + "_int"], inplace=True)

    return df_encoded


def remove_duplicates(df):
    """
    Removes duplicate rows from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame from which duplicates will be removed.

    Returns:
    pandas.DataFrame: A DataFrame with duplicates removed.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    return df_cleaned


def convert_objects_to_ints(df):
    """
    Converts all object (categorical) columns in a DataFrame to integer values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with object columns converted to integers.
        dict: A dictionary mapping original categorical values to integers for each column.
    """
    df_encoded = df.copy()
    mappings = {}

    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = df_encoded[col].astype('category')
        mappings[col] = dict(enumerate(df_encoded[col].cat.categories))  # Store mappings
        df_encoded[col] = df_encoded[col].cat.codes  # Convert to integers

    return df_encoded, mappings

def engineer_customer_pickup_features(df):
    # Ensure datetime type for safety
    if not pd.api.types.is_datetime64_any_dtype(df['arrival_time']):
        df['arrival_time'] = pd.to_datetime(df['arrival_time'])

    # Ensure correct sort order
    df = df.sort_values(['consignee_name', 'arrival_time']).copy()

    # 1. History count
    df['consignee_history_count'] = df.groupby('consignee_name').cumcount()

    # 2. Low history flag
    df['low_history_flag'] = df['consignee_history_count'] < 5

    # 3. Lifetime pickup avg/std (excluding current row)
    df['lifetime_avg_pickup'] = (
        df.groupby('consignee_name')['arrival_to_pickup_time']
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    df['lifetime_std_pickup'] = (
        df.groupby('consignee_name')['arrival_to_pickup_time']
        .expanding()
        .std()
        .shift(1)
        .reset_index(level=0, drop=True)
        .fillna(-1)  # sentinel value for new customers with no history
    )

    # 4. Lifetime pickup median
    df['lifetime_median_pickup'] = (
        df.groupby('consignee_name')['arrival_to_pickup_time']
        .expanding()
        .median()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # 5. Rolling averages (recent 5, 10 pickups)
    for window in [5, 10]:
        df[f'recent_{window}_avg_pickup'] = (
            df.groupby('consignee_name')['arrival_to_pickup_time']
            .transform(lambda x: x.shift(1).rolling(window).mean())
        )

    # 6. Weighted average pickup time (decayed)
    def weighted_avg(series, decay=0.9):
        series = series.dropna().values
        n = len(series)
        if n == 0:
            return np.nan
        weights = np.array([decay**(n - i - 1) for i in range(n)])
        weights = weights / weights.sum()
        return np.dot(series, weights)

    df['weighted_avg_pickup'] = (
        df.groupby('consignee_name')['arrival_to_pickup_time']
        .transform(lambda x: x.shift(1).expanding().apply(weighted_avg, raw=False))
    )

    # 7. Recent 24h pickup rate (last 5 shipments)
    df['recent_5_24h_rate'] = (
        df.groupby('consignee_name')['one_day_pickup']
        .transform(lambda x: x.shift(1).rolling(5).mean())
    )

    # 8. Last pickup within 24h (binary)
    df['last_one_day_pickup'] = (
        df.groupby('consignee_name')['one_day_pickup']
        .shift(1)
    )

    # 9. Time-based features
    df['arrival_hour'] = df['arrival_time'].dt.hour
    df['arrival_weekday'] = df['arrival_time'].dt.dayofweek
    df['is_weekend'] = df['arrival_weekday'].isin([5, 6])

     # 10. Fill NaNs with sentinel value for modeling
    sentinel_fill = -1
    cols_to_fill = [
        'recent_5_avg_pickup',
        'recent_10_avg_pickup',
        'weighted_avg_pickup',
        'recent_5_24h_rate',
        'last_one_day_pickup',
        'lifetime_median_pickup',
        'lifetime_avg_pickup'
    ]
    df[cols_to_fill] = df[cols_to_fill].fillna(sentinel_fill)

    # 11. Fill zeros in lifetime avg pickup to -1 to serve as a sentinel value for brand new customers without historical records.
    # df['lifetime_avg_pickup'] = df['lifetime_avg_pickup'].replace(0, -1)

    return df


def encode_consignee_features(df, target_col='one_day_pickup', rare_thresh=3, global_weight=5):
    """
    Adds frequency encoding, smoothed target encoding, and rare consignee flag to the dataframe.

    Parameters:
        df (pd.DataFrame): Original dataframe containing 'consignee_name' and target column
        target_col (str): Target binary column (e.g., pickup within 24h)
        rare_thresh (int): Minimum appearances to not be considered rare
        global_weight (float): Strength of smoothing toward global mean

    Returns:
        train_df, test_df (pd.DataFrame): Train/test with encoded consignee features
    """
    df = df.copy()

    # 1. Frequency encoding from full dataset (optional: just train_df if you prefer)
    freq_map = df['consignee_name'].value_counts().to_dict()
    for d in [df]:
        d['consignee_freq'] = d['consignee_name'].map(freq_map)

    # 2. Rare consignee flag
    for d in [df]:
        d['is_rare_consignee'] = d['consignee_freq'] < rare_thresh

    # 3. Smoothed target encoding (from train only)
    global_mean = train_df[target_col].mean()

    stats = train_df.groupby('consignee_name')[target_col].agg(['mean', 'count'])
    stats['smoothed'] = (
        (stats['mean'] * stats['count'] + global_mean * global_weight) /
        (stats['count'] + global_weight)
    )
    target_map = stats['smoothed'].to_dict()

    # 4. Apply to train/test (fill with global mean for unseen consignees in test)
    train_df['consignee_target_enc'] = train_df['consignee_name'].map(target_map)
    test_df['consignee_target_enc'] = test_df['consignee_name'].map(target_map).fillna(global_mean)

    return df


