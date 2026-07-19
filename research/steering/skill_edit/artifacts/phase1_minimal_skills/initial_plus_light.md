# Rough ALFWorld Skill V1

Always output one `<think>...</think>` block followed by exactly one `<action>...</action>` block.

Choose an action from the admissible actions. Do not invent actions.

If the previous action did not change the observation, choose a different admissible action.

For a light-examination task, take the target object to a desklamp and turn the desklamp on; do not require informative `examine` feedback.
