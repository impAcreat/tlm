# Minimal Look-at-Light Procedural Skill

Always respond with exactly one `<think>...</think>` block followed by exactly one `<action>...</action>` block.

Choose one action from the admissible action list. Do not invent actions.

## Look at object under desklamp

For goals such as "look at", "examine", or "inspect" an object with a desklamp:

The real completion condition is usually: hold the target object and toggle a desklamp at your current location.

1. Identify the target object type named in the goal, such as bowl, pillow, alarmclock, mug, or book.
2. Find the target object first. Search likely places for that object. For pillows, check all beds before side tables. For bowls/mugs/books/alarmclocks, check desks, tables, shelves, dressers, and open containers.
3. When the target object is visible, immediately take the exact visible object with its full id and source, such as `take pillow 2 from bed 2` or `take bowl 1 from desk 1`.
4. After holding the target object, find a desklamp. Check desks, tables, side tables, dressers, and shelves. If one checked surface has no desklamp, move to a different plausible surface.
5. When a desklamp is visible at your current location, use the exact admissible action for it, usually `use desklamp N`. This should be the final action once you are holding the target object.
6. Do not waste actions examining the target object or the desk. For this task type, `examine TARGET`, `examine LOCATION`, and `look` do not replace taking the target and using the desklamp.

## Avoid loops

- Do not repeat `look`, `examine LOCATION`, or `use desklamp N` more than twice when the observation does not change.
- If the target object is already visible, do not keep examining the same location. Take the target object with the exact admissible action.
- If the desklamp is not visible at the current location, move to a new plausible location instead of re-examining the same surface.
- Never omit the object id or source id. Use `take pillow 2 from bed 2`, not `take pillow`, `examine pillow 2 from bed 2`, or `take pillow from desk 1`.
