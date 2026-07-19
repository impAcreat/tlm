# Rough ALFWorld Skill V1

Always output one `<think>...</think>` block followed by exactly one `<action>...</action>` block.

Choose an action from the admissible actions. Do not invent actions.

If the previous action did not change the observation, choose a different admissible action.

Treat the requested object type as exact: never substitute a semantically related or merely convenient visible object for the named target.

## Location-Bound Placement

For a placement task, first `take` the requested object, then `go to` the destination receptacle before using `move` to place it. Do not attempt to `move` a held object into a destination receptacle while elsewhere.

## Examine Objects with Light

For tasks asking to examine an object with a desklamp, find the target and a desklamp, turn the lamp on with `use <desklamp>`, and take or carry the target as needed. Do not rely on `examine <object>` producing confirmation: its feedback may be uninformative even when the required lamp-and-object condition has been established.
