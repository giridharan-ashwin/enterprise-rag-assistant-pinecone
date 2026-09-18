# Synthetic Enterprise Policy — Engineering Deployment

## Deployment Approval
Production deployments require an approved change record unless an emergency procedure is invoked.

## Deployment Strategy
Services with high availability requirements should use a rolling, blue-green, or canary strategy appropriate to risk.

## Rollback
Every production release should have a documented rollback path. Rollback steps should be validated during release preparation.

## Release Verification
Production releases should include post-deployment verification of health checks, critical metrics, and representative user flows.

## Change Records
Change records should capture release scope, risk, implementation window, and rollback plan.
