# Helios Cloud Storage: Data Retention, Versioning, and Deletion

## Object versioning
Versioning is disabled by default and can be enabled per bucket. Once enabled, versioning can be suspended but not fully disabled. Each stored version is billed as regular storage.

## Soft delete
When an object is deleted, Helios keeps it in a soft-deleted state for 7 days on Starter, 30 days on Business, and 90 days on Enterprise. Soft-deleted objects can be restored with the RestoreObject API call and are billed at the normal storage rate while they are retained.

## Account closure
After an account is closed, all data is retained for 14 days to allow reactivation, then permanently deleted within the following 30 days. Permanent deletion is irreversible, and Helios cannot recover data after that point.

## Object lock
Object Lock (write-once-read-many, WORM) is available on Business and Enterprise plans. Compliance mode locks cannot be shortened or removed by any user, including the account root user. Governance mode locks can be removed by users with the BypassGovernance permission.

## Legal hold
A legal hold prevents deletion of an object regardless of its retention period and remains in place until it is explicitly removed. Legal holds are available on Enterprise only.
