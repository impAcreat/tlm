from research.steering.experiments.reflexion_T.scripts.make_split import expand, split_seen_tasks


def _manifest():
    tasks = {}
    ids = []
    for prefix in ("val", "test"):
        for kind in ("a", "b"):
            for index in range(5):
                task_id = f"{prefix}:{kind}{index}"
                ids.append(task_id)
                tasks[task_id] = {"v0000": {"hard": 0, "task_type": kind}}
    return {"ids": ids, "tasks": tasks}


def test_task_disjoint_split_and_group_expansion():
    manifest = _manifest()
    train, dev = split_seen_tasks(manifest, seed=42, dev_fraction=0.2)
    test = [task_id for task_id in manifest["ids"] if task_id.startswith("test:")]
    assert set(train).isdisjoint(dev)
    assert (set(train) | set(dev)).isdisjoint(test)
    groups = expand(manifest, "train", train, seeds=3)
    assert len(groups) == len(train) * 3
    assert len({group["group_id"] for group in groups}) == len(groups)
