# Synthetic Enterprise Policy — Architecture Standards

## Architecture Review
Material architecture changes should undergo review before implementation. The review should document key trade-offs.

## Service Boundaries
Service boundaries should align with business capabilities, ownership, and change patterns where practical.

## Event-Driven Design
Event-driven designs should define ownership, delivery expectations, idempotency, and failure handling.

## Caching
Caches should have explicit invalidation behavior, ownership, and acceptable staleness requirements.

## Resilience
Critical services should define timeouts, retries, circuit-breaking or equivalent controls appropriate to downstream dependencies.
