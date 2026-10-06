"""Policy view and prompt bytes. Real level-1 data is read only, never rewritten.

The level-1 dataset stores policy states and menus but no request ``user`` text;
training rebuilt it as ``decision_prompt(row["state"], row["menu"])``
(tools/train-level1-24b.py). The checks below therefore pin ``policy_state``
against the collector's own implementation and ``prompt_v1`` against that text.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import unittest

from alpharush_rl import policy_view
from alpharush_rl.journal import canonical_bytes, sha256_data
from alpharush_rl.menus import build_menu, decision_prompt, validate_menu

WORKSPACE = Path(__file__).resolve().parents[1]
DATASET = WORKSPACE / "runtime/rl/level1-24b-phase1/data/dataset.json"
BRANCHES = DATASET.with_name("branches.json")
COLLECTOR = WORKSPACE / "tools/collect-level1-rewards.py"


# Verbatim copy of ``policy_state`` from tools/collect-level1-rewards.py, lines
# 48-52. The tool is not imported: its module level edits sys.path and imports
# alpharush_rl.env/engine/validation (and numpy). test_collector_copy_is_current
# parses the tool's source (without running it) to keep this copy honest.
def collector_policy_state(raw):
    """Do not disclose all future wave_db group counts to the option policy."""
    result = copy.deepcopy(raw)
    result.pop("level_path_wave_counts", None)
    return result


PRIVILEGED = [{"future_wave": 7, "hidden_group_count": 99}]


def state_fixture(holders=2):
    native = {"gold": 200, "lives": 20, "wave": 0, "wave_total": 6, "tick": 0,
              "level_path_wave_counts": copy.deepcopy(PRIVILEGED), "level_name": "测试",
              "holders": [{"id": index + 1, "x": 100 + index, "y": 200, "path_score": 10,
                           "blocked": False} for index in range(holders)],
              "towers": [], "enemies": [], "heroes": [], "wave_ready": True, "action_catalog": []}
    for holder in native["holders"]:
        native["action_catalog"].extend([
            {"action": "build_tower", "holder_id": holder["id"], "tower_type": "archer", "cost": 70, "available": True},
            {"action": "build_tower", "holder_id": holder["id"], "tower_type": "mage", "cost": 100, "available": True},
        ])
    return native


def _function_node(source: str, name: str) -> ast.FunctionDef:
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


class PolicyViewUnitTests(unittest.TestCase):
    def test_policy_state_removes_privileged_without_mutation(self):
        raw = state_fixture()
        before = copy.deepcopy(raw)
        view = policy_view.policy_state(raw)
        self.assertEqual(policy_view.PRIVILEGED_KEYS, ("level_path_wave_counts",))
        self.assertNotIn("level_path_wave_counts", view)
        self.assertEqual(raw, before)
        view["holders"][0]["path_score"] = -1  # deep copy, not a view onto raw
        self.assertEqual(raw, before)
        self.assertEqual(policy_view.policy_state(view), view)  # idempotent

    def test_policy_state_bytes_match_collector_copy(self):
        for raw in (state_fixture(), state_fixture(5), {"tick": 1}, {}):
            ours, theirs = policy_view.policy_state(raw), collector_policy_state(raw)
            self.assertEqual(canonical_bytes(ours), canonical_bytes(theirs))
            # Same insertion order too, so even non-canonical dumps agree.
            self.assertEqual(json.dumps(ours, ensure_ascii=False), json.dumps(theirs, ensure_ascii=False))

    @unittest.skipUnless(COLLECTOR.exists(), "collector tool not present")
    def test_collector_copy_is_current(self):
        tool = _function_node(COLLECTOR.read_text(encoding="utf-8"), "policy_state")
        copied = _function_node(textwrap.dedent(inspect.getsource(collector_policy_state)), "collector_policy_state")
        self.assertEqual((tool.lineno, tool.end_lineno), (48, 52))
        self.assertEqual(ast.dump(tool.args), ast.dump(copied.args))
        self.assertEqual([ast.dump(node) for node in tool.body], [ast.dump(node) for node in copied.body])

    def test_prompt_v1_is_decision_prompt_of_policy_state(self):
        raw = state_fixture()
        menu = build_menu(raw)
        prompt = policy_view.prompt_v1(raw, menu)
        self.assertEqual(prompt, decision_prompt(collector_policy_state(raw), menu))
        self.assertNotIn("level_path_wave_counts", prompt)
        self.assertIn("action_catalog", prompt)
        self.assertEqual(prompt, policy_view.prompt_v1(policy_view.policy_state(raw), menu))

    def test_prompt_v2_compact_and_complete(self):
        raw = state_fixture(3)
        before = copy.deepcopy(raw)
        menu = build_menu(raw)
        prompt = policy_view.prompt_v2(raw, menu)
        self.assertEqual(raw, before)
        decoded = json.loads(prompt)
        self.assertEqual(decoded["schema"], "alpharush-option-v2")
        self.assertEqual(decoded["state"], policy_view.compact_state(raw))
        self.assertNotIn("action_catalog", decoded["state"])
        self.assertNotIn("level_path_wave_counts", prompt)
        self.assertEqual(decoded["menu"], [{"label": m["label"], "text": m["text"]} for m in menu])
        self.assertIn("测试", prompt)  # canonical JSON keeps UTF-8, not \\u escapes

    def test_prompt_v2_validates_menu(self):
        raw = state_fixture()
        menu = build_menu(raw)
        with self.assertRaises(ValueError):
            policy_view.prompt_v2(raw, menu + [dict(menu[1])])
        with self.assertRaises(ValueError):
            policy_view.prompt_v2(raw, [])

    def test_prompt_sha256_hashes_utf8_text(self):
        text = policy_view.prompt_v2(state_fixture(), build_menu(state_fixture()))
        self.assertEqual(policy_view.prompt_sha256(text), hashlib.sha256(text.encode("utf-8")).hexdigest())
        self.assertEqual(policy_view.prompt_sha256("中"), hashlib.sha256("中".encode("utf-8")).hexdigest())


@unittest.skipUnless(DATASET.exists(), "level-1 phase-1 dataset not present")
class Level1DatasetPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = json.loads(DATASET.read_text(encoding="utf-8"))
        cls.branches = json.loads(BRANCHES.read_text(encoding="utf-8")) if BRANCHES.exists() else None

    def rows(self):
        """(name, stored policy state, menu, raw native state) for fork/anchors/validation."""
        group = self.dataset["groups"][0]
        rows = [("fork", group, group["state"], group["menu"])]
        for role, items in sorted(self.dataset["anchors"].items()):
            rows.extend((f"anchor:{role}", item, item["state"], item["menu"]) for item in items)
        rows.extend((f"validation:{item['id']}", item, item["state"], item["menu"])
                    for item in self.dataset["validation_forks"])
        result = []
        for name, row, state, menu in rows:
            if name == "fork" and self.branches is not None:
                raw = self.branches["fork_state"]
                # The exact native fork the collector projected into this row.
                self.assertEqual(sha256_data(raw), group["native_fork_state_sha256"])
            else:
                raw = {**state, "level_path_wave_counts": copy.deepcopy(PRIVILEGED)}
            result.append((name, row, state, menu, raw))
        self.assertGreaterEqual(len(result), 4)
        return result

    def test_stored_states_are_policy_states(self):
        for name, _, state, _, _ in self.rows():
            with self.subTest(name):
                self.assertNotIn("level_path_wave_counts", state)
                self.assertEqual(canonical_bytes(policy_view.policy_state(state)), canonical_bytes(state))

    def test_policy_state_byte_equivalent_to_collector(self):
        for name, _, state, _, raw in self.rows():
            with self.subTest(name):
                self.assertIn("level_path_wave_counts", raw)
                ours = policy_view.policy_state(raw)
                self.assertEqual(canonical_bytes(ours), canonical_bytes(collector_policy_state(raw)))
                self.assertEqual(json.dumps(ours, ensure_ascii=False),
                                 json.dumps(collector_policy_state(raw), ensure_ascii=False))
                self.assertEqual(canonical_bytes(ours), canonical_bytes(state))

    def test_prompt_v1_reproduces_level1_user_bytes(self):
        for name, row, state, menu, raw in self.rows():
            with self.subTest(name):
                # A saved request text wins if present; this dataset stores none.
                expected = row["user"] if isinstance(row.get("user"), str) else decision_prompt(state, menu)
                self.assertEqual(policy_view.prompt_v1(raw, menu).encode("utf-8"), expected.encode("utf-8"))
                self.assertEqual(policy_view.prompt_v1(state, menu).encode("utf-8"), expected.encode("utf-8"))

    def test_prompt_v2_is_shorter_and_covers_every_label(self):
        ratios = {}
        for name, _, state, menu, raw in self.rows():
            with self.subTest(name):
                v1, v2 = policy_view.prompt_v1(raw, menu), policy_view.prompt_v2(raw, menu)
                self.assertNotIn("action_catalog", v2)
                self.assertNotIn("level_path_wave_counts", v2)
                decoded = json.loads(v2)
                self.assertEqual([m["label"] for m in decoded["menu"]], validate_menu(menu))
                self.assertEqual([m["text"] for m in decoded["menu"]], [m["text"] for m in menu])
                ratios[name] = len(v2.encode("utf-8")) / len(v1.encode("utf-8"))
                self.assertLess(ratios[name], 0.6, f"v2/v1 byte ratio {ratios[name]:.3f}")
        self.assertTrue(ratios)


if __name__ == "__main__":
    unittest.main()
