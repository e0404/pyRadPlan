# Constraint tables (templates)

**PLACEHOLDER VALUES — NOT FOR CLINICAL USE.**

Used by `pyRadPlan.ai.agents` (`get_site_constraints`). Copy a template, replace the
values with your institution's validated protocol and point the agent to it
(`load_constraint_table(path)`).

Schema
- `dose_reference.type`:
  - `"EQD2"`: dose values are EQD2 (2 Gy fractions), converted to the plan fractionation
    with each structure's `alpha_beta`.
  - `"fractionation"`: dose values are physical doses for the site's `prescription`
    (`total_dose_gy`, `num_fractions`); converted via EQD2 if the plan differs.
- `sites.<site>.structures.<CanonicalName>`: `alpha_beta` (Gy) + `constraints`.
- `sites.<site>.targets.<Name>`: target goals; `unit: "%Rx"` = percent of the plan's
  prescription (not fractionation-converted).
- constraint: `metric` (`mean`, `max`, `min`, `D<vol%>`, `V<dose>Gy`), `op` (`<=`, `>=`),
  `value` (Gy for dose metrics, % for V metrics), `hard`, optional `unit`, `endpoint`, `note`.

Structure names are canonical (TG-263-like); the agent maps them to the patient's
VOIs with `map_structures`.
