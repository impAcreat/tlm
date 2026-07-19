# ScienceWorld Procedure

Observe the current state, choose one valid action, and check the resulting observation.

## Parser Error Recovery
If an observation says `No known action matches that input`, do not repeat the same action. Try a simpler ScienceWorld command form, such as `look around`, `look in <container>`, `open <container>`, `take <object>`, `put <object> in <container>`, `put <object> on <device>`, `activate <device>`, or `focus on <substance>`. Use `look around` or inspect visible containers/devices to discover exact object names before interacting with them.
