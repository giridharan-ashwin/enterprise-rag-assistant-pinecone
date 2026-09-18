# Synthetic Enterprise Policy — Cloud Aws

## AWS Region Selection
Production workloads should use an approved AWS region based on latency, data residency, and service availability requirements.

## AWS IAM Roles
Workloads should use IAM roles instead of long-lived access keys whenever possible.

## AWS Logging
CloudTrail and approved service logs should be enabled for production AWS accounts and retained according to data policy.

## AWS Cost Tags
Production AWS resources should include project and cost-center tags for financial reporting.

## AWS Backup
Critical AWS data stores should have an approved backup and recovery strategy with periodic restore validation.
