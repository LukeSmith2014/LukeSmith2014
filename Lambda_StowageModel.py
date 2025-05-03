import boto3
import json
import io

runtime = boto3.client('sagemaker-runtime')
s3 = boto3.client('s3')
ENDPOINT_NAME = 'stowage-priority-endpoint-v28'

# Columns to return in final predictions
FINAL_OUTPUT_COLUMNS = [
    "equipment_id",
    "booking_number",
    "consignee_name",
    "voyage_code",
    "stowage_priority"
]

def lambda_handler(event, context):
    try:
        print("Event received:", json.dumps(event))

        # Step 1: Extract bucket/key from S3 event
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        print(f"Reading file from s3://{bucket}/{key}")

        # Step 2: Read the cleaned JSON file from S3
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj['Body'].read()
        records = json.loads(body)

        print(f"Loaded {len(records)} records")

        # Step 3: Call SageMaker endpoint once with all records
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(records)
        )

        # Step 4: Parse model response
        result = response['Body'].read().decode('utf-8')
        predictions = json.loads(result)

        # Step 5: Trim to only desired output columns
        trimmed_records = [
            {k: rec[k] for k in FINAL_OUTPUT_COLUMNS if k in rec}
            for rec in predictions
        ]

        # Step 6: Write back to S3 as one file
        output_key = key.replace("cleaned/", "predictions/").replace(".json", "_predictions.json")
        out_buffer = io.BytesIO()
        out_buffer.write(json.dumps(trimmed_records).encode("utf-8"))
        s3.put_object(Bucket=bucket, Key=output_key, Body=out_buffer.getvalue())

        print(f"Predictions written to: s3://{bucket}/{output_key}")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Inference completed',
                'output_s3': f"s3://{bucket}/{output_key}"
            })
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }