
import boto3
import pandas as pd
import io
import os

from cleaning_script_trimmed_cleaned import (
    standardize_column_names,
    create_key_feature,
    map_on_key,
    create_voyage_code,
    add_next_voyage,
    create_voyage_code_1,
    add_arrival_time,
    remove_columns,
    standardize_datetime_columns,
    count_hlp_occurrences,
    add_pickup_date,
    calculate_storage_time,
    calculate_time_til_avp,
    add_day_of_week,
    add_time_to_unload_feature,
    arrival_to_pickup_time,
    port_to_ste_time,
    remove_negative_values,
    add_load_date,
    create_binary_feature,
    remove_null_records,
    calculate_journey_length,
    fill_null_values,
    apply_one_hot_encoding,
    remove_duplicates,
    engineer_customer_pickup_features
)

s3 = boto3.client("s3")

def clean_container_data(event_df, vessel_df, schedule_df):
    event_df = standardize_column_names(event_df)
    vessel_df = standardize_column_names(vessel_df)
    schedule_df = standardize_column_names(schedule_df)

    event_df = create_key_feature(event_df)
    vessel_df = create_key_feature(vessel_df)

    df = map_on_key(event_df, vessel_df, "key")

    schedule_df = create_voyage_code(schedule_df, 'vessel_code', 'voyage_number')
    df = add_next_voyage(df, 'voyage')
    df = create_voyage_code_1(df, 'vessel_code', 'next_voyage', 'direction_seq')
    df = add_arrival_time(df, schedule_df)

    drop_list = ['check_digit', 'event_record_type_code', 'booking_number_y',
                 'priority_stow_code', 'vessel_code_y', 'voyage_y', 'direction_seq_y',
                 'equipment_id_y', 'empty_full_code_y', 'equipment_voyage', 'equipment_direction_seq',
                 'load_port_code_y', 'discharge_port_code_y', 'destination_port_code_y',
                 'previous_load_port_code', 'previous_load_vessel_code', 'mounted_to_equipment_id',
                 'life_cycle_status', 'current_direction_seq', 'actual_temperature',
                 'actual_temperature_scale', 'ingate_trucker_code', 'create_user',
                 'create_date', 'last_update_user', 'last_update_date', 'auto_cfs_code',
                 'b_p_bol_origin_port_code', 'b_p_bol_destination_port_code']
    df = remove_columns(df, drop_list)

    date_cols = ['event_date', 'sail_date', 'stow_date', 'ingate_date',
                 'create_date', 'last_update_date', 'arrival_time']
    df = standardize_datetime_columns(df, date_cols)

    df = count_hlp_occurrences(df)
    df = add_pickup_date(df)
    df = calculate_storage_time(df)
    df = calculate_time_til_avp(df)
    df = add_day_of_week(df, 'avp_date')
    df = add_time_to_unload_feature(df, 'key', 'arrival_time', 'event_type', 'DFV', 'event_date')
    df = arrival_to_pickup_time(df)
    df = port_to_ste_time(df)

    neg_features = ['port_to_ste_time', 'arrival_to_pickup_time', 'time_to_unload',
                    'time_til_avp', 'storage_time']
    df = remove_negative_values(df, neg_features)

    df = add_load_date(df, 'key', 'event_type', 'LTV', 'event_date')
    df = create_binary_feature(df, 'arrival_to_pickup_time', 24, 'one_day_pickup')

    drop_null_list = ['consignee_name', 'shipper_name', 'booking_number',
                      'storage_time', 'arrival_time', 'carrier_code']
    df = remove_null_records(df, drop_null_list)

    df = calculate_journey_length(df)

    df = fill_null_values(df, 'hazard_type_code', 'Unknown')
    df = fill_null_values(df, 'current_damage_code', 'Unknown')
    df = fill_null_values(df, 'equipment_type_height_code', 'Unkown')
    df = fill_null_values(df, 'commodity', 'Unkown')
    df = fill_null_values(df, 'life_cycle_status_code', 'Unknown')
    df = fill_null_values(df, 'pds_code', 'Unknown')
    df = fill_null_values(df, 'discharge_port_code', 'Unknown')

    OHE_list = ['empty_full_code', 'trade_route', 'pds_code', 'plan_equip_type_code']
    df = apply_one_hot_encoding(df, OHE_list)

    df = remove_duplicates(df)
    df = df.drop_duplicates(subset='key', keep='last')
    df = engineer_customer_pickup_features(df)

    return df

def lambda_handler(event, context):
    print("Event received:", event)

    input_bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    filename = os.path.basename(key)

    output_bucket = "matson-cleaned-files"

    obj = s3.get_object(Bucket=input_bucket, Key=key)
    vessel_df = pd.read_csv(io.BytesIO(obj['Body'].read()))

    event_df = pd.read_csv("s3://raw-container-inputs/static/Event_data_01012024_06302024_v1.csv")
    schedule_df = pd.read_csv("s3://raw-container-inputs/static/LAX_to_SHA_Schedule.csv")

    cleaned_df = clean_container_data(event_df, vessel_df, schedule_df)

    out_buffer = io.BytesIO()
    cleaned_df.to_json(out_buffer, orient="records")
    out_key = f"cleaned/{filename.replace('.csv', '.json')}"

    s3.put_object(Bucket=output_bucket, Key=out_key, Body=out_buffer.getvalue())
    print(f"Cleaned file written to: s3://{output_bucket}/{out_key}")

    return {
        'statusCode': 200,
        'body': f"Cleaned file saved to s3://{output_bucket}/{out_key}"
    }
