# Helios Cloud Storage: API Rate Limits and Object Limits

## Request rate limits (per bucket)
- Starter: 100 PUT/POST/DELETE requests per second and 500 GET requests per second.
- Business: 3,500 PUT/POST/DELETE requests per second and 5,500 GET requests per second.
- Enterprise: limits are set per contract; the default is the same as Business and can be raised on request.

Requests above the limit receive HTTP 429 (Too Many Requests) with a Retry-After header. Clients should retry with exponential backoff and jitter.

## Object size limits
The maximum size of a single object is 5 TB. A single PUT request can upload at most 5 GB; larger objects must use multipart upload. Multipart uploads support between 1 and 10,000 parts, and every part except the last must be at least 5 MB.

## Listing
ListObjects returns at most 1,000 keys per response. Use the continuation token to page through larger buckets.

## Presigned URLs
Presigned URLs can be valid for at most 7 days. Presigned URLs created with an API key stop working if that key is disabled or deleted.
