# Shared ALFWorld Skill

Always output one `<think>...</think>` block followed by exactly one `<action>...</action>` block.

Choose an action from the admissible actions. Do not invent actions.

If the previous action did not change the observation, choose a different admissible action.

Treat the requested object type as exact. Never substitute a related or convenient visible object for the named target.

## Multi-Object Placement

For each requested instance, first `take` the matching object, then `go to` the destination receptacle before using `move` to place it. Do not attempt to `move` a held object into a destination while elsewhere.

Count only distinct successful placements of matching target objects. A correctly delivered object is committed progress: never take it back out, relocate it, or replace it. Continue until the required number of distinct objects has been successfully delivered; do not declare completion after a partial quantity.
