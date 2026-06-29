import sys

# --- updater.py: attach real environment_id/task_type to each correction row ---
u = "research/efm/updater.py"
s = open(u).read()
old = (
    "        correction_rows = []\n"
    "        for correction in corrections:\n"
    "            row = correction.to_dict()\n"
    "            correction_rows.append(row)"
)
new = (
    "        scope_by_episode = {\n"
    "            str(episode.get(\"episode_id\", \"\")): (\n"
    "                str(episode.get(\"environment_id\", \"\")),\n"
    "                str(episode.get(\"task_type\", \"\")),\n"
    "            )\n"
    "            for episode in window\n"
    "        }\n"
    "        correction_rows = []\n"
    "        for correction in corrections:\n"
    "            row = correction.to_dict()\n"
    "            environment_id, task_type = scope_by_episode.get(correction.episode_id, (\"\", \"\"))\n"
    "            row[\"environment_id\"] = environment_id\n"
    "            row[\"task_type\"] = task_type\n"
    "            correction_rows.append(row)"
)
assert s.count(old) == 1, "updater correction_rows anchor not unique"
s = s.replace(old, new)
open(u, "w").write(s)
print("updater.py scope-threading patched")

# --- prompts.py: tell the proposer to use the provided scope, not invent one ---
p = "research/efm/prompts.py"
s = open(p).read()
old_scope = (
    '    "scope" MUST be a JSON object (use {} for a global rule), never a string.'
)
new_scope = (
    '    "scope" MUST be a JSON object, never a string. Set it from the\n'
    '    environment_id / task_type carried on the supporting corrections;\n'
    '    prefer {"environment_id": "<that id>"} alone so the rule applies across\n'
    '    tasks in that environment, and use {} only for a rule meant for every\n'
    '    environment. Never invent scope values not present in the input.'
)
assert s.count(old_scope) == 1, "prompts scope anchor not unique"
s = s.replace(old_scope, new_scope)
open(p, "w").write(s)
print("prompts.py scope-guidance patched")

sys.path.insert(0, ".")
import importlib
import research.efm.prompts as P
importlib.reload(P)
import research.efm.updater  # noqa
print("imports OK")
