import sys

# --- policy.py: add scope-is-object guard to add_rule / add_example ---
p = "research/efm/policy.py"
s = open(p).read()

old_rule = (
    '        if op == "add_rule":\n'
    '            instruction = str(edit.get("instruction", "")).strip()'
)
new_rule = (
    '        if op == "add_rule":\n'
    '            if not isinstance(edit.get("scope", {}), dict):\n'
    '                return "edit_scope_not_object"\n'
    '            instruction = str(edit.get("instruction", "")).strip()'
)
assert s.count(old_rule) == 1, "policy add_rule anchor not unique"
s = s.replace(old_rule, new_rule)

old_ex = (
    '        elif op == "add_example":\n'
    '            if not str(edit.get("situation", "")).strip()'
)
new_ex = (
    '        elif op == "add_example":\n'
    '            if not isinstance(edit.get("scope", {}), dict):\n'
    '                return "edit_scope_not_object"\n'
    '            if not str(edit.get("situation", "")).strip()'
)
assert s.count(old_ex) == 1, "policy add_example anchor not unique"
s = s.replace(old_ex, new_ex)
open(p, "w").write(s)
print("policy.py patched")

# --- updater.py: pass min_support into policy_user_prompt ---
u = "research/efm/updater.py"
s = open(u).read()
old_call = (
    "                policy_user_prompt(\n"
    "                    policy=policy.to_dict(),\n"
    "                    corrections=correction_rows,\n"
    "                    edit_budget=self.config.policy_max_edits,\n"
    "                ),"
)
new_call = (
    "                policy_user_prompt(\n"
    "                    policy=policy.to_dict(),\n"
    "                    corrections=correction_rows,\n"
    "                    edit_budget=self.config.policy_max_edits,\n"
    "                    min_support=self.config.policy_min_support,\n"
    "                ),"
)
assert s.count(old_call) == 1, "updater call anchor not unique"
s = s.replace(old_call, new_call)
open(u, "w").write(s)
print("updater.py patched")

sys.path.insert(0, ".")
import research.efm.prompts, research.efm.policy, research.efm.updater  # noqa
print("imports OK")
