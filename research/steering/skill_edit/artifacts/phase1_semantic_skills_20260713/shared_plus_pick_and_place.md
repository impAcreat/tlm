# Shared ALFWorld Skill

Always output one `<think>...</think>` block followed by exactly one `<action>...</action>` block.

Choose an action from the admissible actions. Do not invent actions.

If the previous action did not change the observation, choose a different admissible action.

Treat the requested object type as exact. Never substitute a related or convenient visible object for the named target.

## Single-Object Placement

First `take` the requested object, then `go to` the destination receptacle before using `move` to place it. Do not attempt to `move` a held object into a destination while elsewhere.

Once the requested object has been successfully delivered, do not take it back out or relocate it.
