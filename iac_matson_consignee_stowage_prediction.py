# -*- coding: utf-8 -*-
"""IaC_Matson_Consignee_Stowage_Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uNcjFmlshiylWB_74PWMQ76isQ4C6OzF
"""

# CDK-based Infrastructure as Code for Matson Container Stowage Prediction Pipeline

from aws_cdk import (
    Stack,
    Duration,
    Aws,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_iam as iam,
    aws_s3_notifications as s3n,
    aws_apigatewayv2 as apigw,
    aws_apigatewayv2_integrations as integrations,
    aws_sagemaker as sagemaker,
)
from constructs import Construct

from aws_cdk.aws_ecr_assets import DockerImageAsset
from aws_cdk.aws_sagemaker import CfnModel

class StowagePipelineStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # S3 Buckets
        raw_bucket = s3.Bucket(self, "RawContainerInputs", bucket_name=f"raw-container-inputs-{Aws.ACCOUNT_ID}")
        cleaned_bucket = s3.Bucket(self, "CleanedContainerFiles", bucket_name=f"matson-cleaned-files-{Aws.ACCOUNT_ID}")

        # Lambda Role (shared for all)
        lambda_role = iam.Role(
            self, "LambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ]
        )

        raw_bucket.grant_read(lambda_role)
        cleaned_bucket.grant_read_write(lambda_role)

        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=["sagemaker:InvokeEndpoint"],
            resources=["*"]
        ))

        # Cleaning Lambda
        cleaning_lambda = _lambda.Function(
            self, "CleaningLambda",
            function_name="CleanUploadedContainerFile",
            runtime=_lambda.Runtime.PYTHON_3_9,
            code=_lambda.Code.from_asset("./lambda/cleaning"),
            handler="lambda_function.lambda_handler",
            role=lambda_role,
            timeout=Duration.seconds(60),
            memory_size=1024
        )

        # Trigger: raw upload triggers cleaning lambda
        raw_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(cleaning_lambda),
            s3.NotificationKeyFilter(prefix="raw/clean_input/", suffix=".csv")
        )

        # Inference Lambda
        inference_lambda = _lambda.Function(
            self, "InferenceLambda",
            function_name="StowageModelLambda",
            runtime=_lambda.Runtime.PYTHON_3_9,
            code=_lambda.Code.from_asset("./lambda/inference"),
            handler="lambda_function.lambda_handler",
            role=lambda_role,
            timeout=Duration.seconds(60),
            memory_size=1024
        )

        # Trigger: cleaned file triggers inference lambda
        cleaned_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(inference_lambda),
            s3.NotificationKeyFilter(prefix="cleaned/", suffix=".json")
        )

        # API Lambda
        api_lambda = _lambda.Function(
            self, "ApiLambda",
            function_name="StowageModelLambdaAPI",
            runtime=_lambda.Runtime.PYTHON_3_9,
            code=_lambda.Code.from_asset("./lambda/inference"),
            handler="lambda_function.lambda_handler",
            role=lambda_role,
            timeout=Duration.seconds(60),
            memory_size=1024
        )

        # API Gateway
        http_api = apigw.HttpApi(
            self, "StowageAPI",
            api_name="StowagePredictionAPI",
        )

        http_api.add_routes(
            path="/predict",
            methods=[apigw.HttpMethod.POST],
            integration=integrations.HttpLambdaIntegration("LambdaIntegration", api_lambda)
        )

        self.http_api_url = http_api.url

        # SageMaker Model Setup
        model_artifact = "s3://amzn-s3-asu-matson-project/models/stowage-priority/model.tar.gz"
        model_name = "stowage-priority-model"

        # IAM role for SageMaker
        sagemaker_role = iam.Role(
            self, "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3ReadOnlyAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly")
            ]
        )

        # Build and reference the Docker container
        docker_image = DockerImageAsset(self, "StowageDockerImage", directory=".")

        # Define the SageMaker Model
        model = CfnModel(
            self, "StowageModel",
            model_name=model_name,
            execution_role_arn=sagemaker_role.role_arn,
            primary_container=CfnModel.ContainerDefinitionProperty(
                image=docker_image.image_uri,
                model_data_url=model_artifact
            )
        )

        # Create Endpoint Config
        endpoint_config = sagemaker.CfnEndpointConfig(
            self, "StowageEndpointConfig",
            endpoint_config_name="stowage-endpoint-config",
            production_variants=[{
                "initialInstanceCount": 1,
                "instanceType": "ml.t2.medium",
                "modelName": model.model_name,
                "variantName": "AllTraffic"
            }]
        )
        endpoint_config.add_dependency(model)

        # Deploy Endpoint
        endpoint = sagemaker.CfnEndpoint(
            self, "StowageEndpoint",
            endpoint_name="stowage-priority-endpoint",
            endpoint_config_name=endpoint_config.endpoint_config_name
        )
        endpoint.add_dependency(endpoint_config)
