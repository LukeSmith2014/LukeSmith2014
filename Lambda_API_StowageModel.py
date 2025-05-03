
import boto3
import json
import io

runtime = boto3.client('sagemaker-runtime')
ENDPOINT_NAME = 'stowage-priority-endpoint-v28'

# Final output columns
FINAL_OUTPUT_COLUMNS = [
    "equipment_id",
    "consignee_name",
    "voyage_code",
    "stowage_priority"
]

def lambda_handler(event, context):
    try:
        print("Incoming event:", json.dumps(event))

        # Step 1: Parse input from API Gateway
        body = event.get("body")
        if isinstance(body, str):
            records = json.loads(body)
        else:
            records = body  # Already parsed

        print(f"Received {len(records)} records")

        # Step 2: Call SageMaker endpoint
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(records)
        )

        # Step 3: Parse model response
        result = response['Body'].read().decode('utf-8')
        predictions = json.loads(result)

        # Step 4: Trim output
        trimmed_predictions = [
            {k: rec[k] for k in FINAL_OUTPUT_COLUMNS if k in rec}
            for rec in predictions
        ]

        return {
            'statusCode': 200,
            'headers': {"Content-Type": "application/json"},
            'body': json.dumps(trimmed_predictions)
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }
