# Synthetic Enterprise Policy — Cloud Kubernetes

## Kubernetes Namespaces
Production workloads should be isolated by namespace according to application ownership and security boundaries.

## Kubernetes Resources
Production deployments should define CPU and memory requests and limits appropriate to workload behavior.

## Kubernetes Health Checks
Production workloads should define readiness and liveness probes where supported by the service.

## Kubernetes Secrets
Application secrets should be supplied through approved secret-management mechanisms rather than committed to container images.

## Kubernetes Autoscaling
Workloads with variable demand should use horizontal autoscaling when supported by meaningful utilization metrics.
