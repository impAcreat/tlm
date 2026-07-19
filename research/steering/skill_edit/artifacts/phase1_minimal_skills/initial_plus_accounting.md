# Rough ALFWorld Skill V1

Always output one `<think>...</think>` block followed by exactly one `<action>...</action>` block.

Choose an action from the admissible actions. Do not invent actions.

If the previous action did not change the observation, choose a different admissible action.

Before acting, identify the exact target type, required count, required transformations, and destination. Count only distinct matching objects completed with all requirements satisfied; do not disturb completed objects, and continue until the required count is met.
