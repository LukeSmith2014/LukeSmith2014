#!/usr/bin/env python3
import aws_cdk as cdk
from iac_matson_consignee_stowage_prediction import StowagePipelineStack

app = cdk.App()
StowagePipelineStack(app, "StowagePipelineStack")
app.synth()
