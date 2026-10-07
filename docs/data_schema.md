# Released dataset schema

The three task views are byte copies of the public hosted release pinned in
`data/release_manifest.json`. All 30,083 opaque numeric IDs are aligned across
task views and the public master; no source platform identifier is encoded in
those IDs. Earlier Git history retains the original import; these statements
refer to the current release files.

- Understanding: `sample_id`, `comments`, `IntentOptions`, `TargetOptions`, and
  `gold.{Intent,Target,Mechanism}`.
- Response: `sample_id`, `comments`, and
  `gold.reference_replies.{teasing,pushback,neutral}`.
- Generation: `sample_id`, `comments`, `intended_meaning`, and
  `gold.{reference_sarcastic_comment,target,preferred_mechanisms}`.

The public master adds platform/language, textual video metadata, parent and
reconstructed replies, the target turn, core annotations, annotation-model
metadata, and reconstruction provenance. Source video/comment IDs, source URLs,
and unused original raw-reply metadata are omitted. Public text itself may
still contain names, nicknames, offensive language, or cultural references.

Six canonical mechanisms are accompanied by a residual `other` fallback.
The 1/2/3-label cardinality distribution includes `other`. Canonical prevalence
uses all 30,083 events as its denominator. Canonical gold shares and entropy use
only occurrences of the six canonical labels as their denominator. Records
containing only `other` contribute to total event counts/cardinality, but not
to canonical-label occurrence counts. Never mix these denominators.

`requires_context=false` identifies context-independent events. Minimal context
span percentages conditioned on context dependence exclude these events from
the parent-only group. Context-dependent strata are not automatically the same
as a target-location ablation subset.
