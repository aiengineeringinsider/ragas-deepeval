# Helios Cloud Storage: Regions and Replication

## Regions
Helios operates five regions: us-east (Virginia), us-west (Oregon), eu-central (Frankfurt), ap-southeast (Singapore), and ap-south (Mumbai). A bucket's region is chosen at creation time and cannot be changed later; to move data, create a new bucket and copy the objects.

## Durability
Each object is stored with erasure coding across at least three availability zones within its region. Helios designs for 99.999999999% (eleven nines) annual durability. Durability is a design target and is not covered by the SLA.

## Cross-region replication
Cross-region replication (CRR) asynchronously copies new objects to a bucket in another region. CRR is available on Business and Enterprise plans. Replicated data is billed as storage in the destination region plus a replication transfer fee of $0.02 per GB. Existing objects are not replicated automatically; use the Batch Replicate job to copy them.

## Replication time
On Enterprise, the Replication Time Control option guarantees that 99.9% of new objects replicate within 15 minutes. Without Replication Time Control, most objects replicate within minutes but there is no time guarantee.

## Data residency
Data stored in eu-central never leaves the European Union unless the customer configures cross-region replication to a non-EU region.
