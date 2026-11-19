# Helios Cloud Storage: Security and Compliance

## Encryption
All objects are encrypted at rest with AES-256. By default, Helios manages the keys (SSE-H). Business and Enterprise customers can use customer-managed keys stored in the Helios Key Service (SSE-K). Enterprise customers can additionally bring their own external key manager over KMIP (SSE-X). Data in transit is protected with TLS 1.2 or higher; TLS 1.0 and 1.1 connections are rejected.

## Access control
Access is controlled with bucket policies and API keys. API keys can be scoped to specific buckets and to read-only, write-only, or read-write permissions. Keys that have not been used for 180 days are automatically disabled on Business and Enterprise plans.

## Single sign-on
SAML 2.0 single sign-on for the web console is available on Enterprise only. SCIM user provisioning is also Enterprise only.

## Certifications
Helios holds SOC 2 Type II and ISO/IEC 27001 certifications. Helios is HIPAA-eligible: customers who store protected health information must sign a Business Associate Agreement (BAA), which is offered on Enterprise only. Helios is not FedRAMP authorized.

## Audit logs
Access logs record every API request and are retained for 90 days on Business and 400 days on Enterprise. Starter accounts do not have access logging.
