-- Singular test: fails (returns rows) if any applicant has TARGET set
-- inconsistently with is_train - training rows must have a label, scoring
-- rows must not.

select sk_id_curr
from {{ ref('mart_applicant_features') }}
where (is_train and target is null)
   or (not is_train and target is not null)
