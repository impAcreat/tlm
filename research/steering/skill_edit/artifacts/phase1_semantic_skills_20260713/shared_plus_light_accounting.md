# Rough ALFWorld Skill V1

Always output one `<think>...</think>` block followed by exactly one `<action>...</action>` block.

Choose an action from the admissible actions. Do not invent actions.

If the previous action did not change the observation, choose a different admissible action.

## Goal Identity and Completion Accounting

Treat the requested object type as exact: never substitute a semantically related or merely convenient visible object for the named target. If the task requires multiple instances, count only distinct successful placements of matching target objects that satisfy every requested condition at the destination.

A correctly delivered goal object is committed progress: never take it back out, relocate it, or replace it with another object. Continue only until the required number of matching objects has been successfully delivered; do not declare completion after a partial quantity.

## Examine Objects with Light

For tasks asking to examine an object with a desklamp, find the target and a desklamp, turn the lamp on with `use <desklamp>`, and take or carry the target as needed. Do not rely on `examine <object>` producing confirmation: its feedback may be uninformative even when the required lamp-and-object condition has been established.
