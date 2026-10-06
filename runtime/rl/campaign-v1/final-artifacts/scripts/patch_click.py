"""v2 action click_entity: the GUI click a game script waits for (PickView sets e.ui.clicked = true).

Offered only for (a) a boss knocked down and waiting for the finishing tap (J.T.: e.dying with a tap decal)
and (b) a click-to-break modifier on a tower (J.T.'s ice: e.modifier with required_clicks > 0).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PATCH_ROOT", "C:/Users/<user>/Documents/AlphaRush"))


def patch(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:80], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8", newline="\n")


patch("alpharush_rl/assets/host.lua", [
    ('''local V2_ACTIONS = {build_tower=true, send_wave=true, upgrade_tower=true,
    upgrade_power=true, sell_tower=true, use_power=true, point_tower=true}
local V2_MATCH_KEYS = {"action", "holder_id", "tower_id", "tower_type", "target", "power", "x", "y", "anchor_id"}''',
     '''local V2_ACTIONS = {build_tower=true, send_wave=true, upgrade_tower=true,
    upgrade_power=true, sell_tower=true, use_power=true, point_tower=true, click_entity=true}
local V2_MATCH_KEYS = {"action", "holder_id", "tower_id", "tower_type", "target", "power", "x", "y", "anchor_id",
    "entity_id"}'''),
    ('''local function clickable(e)
    return not (type(e.ui) == "table" and e.ui.can_click == false)
end''',
     '''local function clickable(e)
    return not (type(e.ui) == "table" and e.ui.can_click == false)
end

-- Clicks a game script is waiting for (the GUI's PickView sets e.ui.clicked on a clickable entity):
-- a boss knocked down awaiting the finishing tap (J.T.: dying, tap decal shown) and a click-to-break
-- modifier on a tower (J.T.'s ice; its script counts required_clicks itself). Returns the kind, or nil.
local function click_need(e)
    local ui = type(e) == "table" and e.ui
    if type(ui) ~= "table" or ui.can_click ~= true or ui.clicked then return nil end
    if type(e.pos) ~= "table" or not finite(e.pos.x) or not finite(e.pos.y) then return nil end
    if e.dying == true and type(e.tap_decal) == "string" and type(e.health) == "table" and not e.health.dead then
        return "downed_boss"
    end
    if type(e.modifier) == "table" and finite(e.required_clicks) and e.required_clicks > 0 then
        return "tower_trap"
    end
    return nil
end'''),
    ('''    local anchors = power_anchors(s.enemies)
    for p = 1, 2 do''',
     '''    for id, e in pairs(store.entities or {}) do
        local need = click_need(e)
        if need then
            add({action="click_entity", entity_id=id, x=math.floor(e.pos.x + 0.5), y=math.floor(e.pos.y + 0.5),
                 template=type(e.template_name) == "string" and e.template_name or "", kind=need, cost=0})
        end
    end
    local anchors = power_anchors(s.enemies)
    for p = 1, 2 do'''),
    ('''        keys[a] = {a.action or "", a.holder_id or a.tower_id or 0, a.tower_type or a.target or named or "",''',
     '''        keys[a] = {a.action or "", a.holder_id or a.tower_id or a.entity_id or 0, a.tower_type or a.target or named or "",'''),
    ('''    elseif cmd.action == "point_tower" then
        -- As the GUI's tw_point click: the tower's own script picks the enemy at this point and fires.''',
     '''    elseif cmd.action == "click_entity" then
        -- As the GUI's PickView click on this entity: its script consumes ui.clicked.
        local e = game.store.entities[cmd.entity_id]
        e.ui.clicked = true
        return {native_wire=native.codec.encode({type="ok", action="click_entity", entity_id=cmd.entity_id,
                                                 x=cmd.x, y=cmd.y})}
    elseif cmd.action == "point_tower" then
        -- As the GUI's tw_point click: the tower's own script picks the enemy at this point and fires.'''),
])

patch("alpharush_rl/env.py", [
    ('''SCOPES = {"v1": SCOPE_V1, "v2": SCOPE_V1 + ("upgrade_tower", "upgrade_power", "sell_tower", "use_power",
                                           "point_tower")}''',
     '''SCOPES = {"v1": SCOPE_V1, "v2": SCOPE_V1 + ("upgrade_tower", "upgrade_power", "sell_tower", "use_power",
                                           "point_tower", "click_entity")}'''),
    ('''ACTION_TICKS = {"build_tower": BUILD_TICKS, "upgrade_tower": UPGRADE_TICKS, "sell_tower": SELL_TICKS}''',
     '''CLICK_TICKS = 30
ACTION_TICKS = {"build_tower": BUILD_TICKS, "upgrade_tower": UPGRADE_TICKS, "sell_tower": SELL_TICKS,
                "click_entity": CLICK_TICKS}'''),
    ('''# v2 completion receipts: each returns''',
     '''def _click_entity_receipt(before, after, action, cost):
    """A GUI click only sets the entity's ui.clicked flag (the host replied ok) and its script counts the
    clicks itself, so the evidence is that the click was offered before, and whether it is offered after."""
    def offered(state):
        return any(isinstance(item, dict) and item.get("action") == "click_entity"
                   and item.get("entity_id") == action["entity_id"] for item in state.get("action_catalog") or [])
    return {"executed": offered(before), "offered_after": offered(after)}


# v2 completion receipts: each returns'''),
    ('''               "point_tower": _point_tower_receipt}''',
     '''               "point_tower": _point_tower_receipt, "click_entity": _click_entity_receipt}'''),
])

patch("alpharush_rl/menus.py", [
    ('''SUPPORTED_ACTIONS = frozenset(("wait", "build_tower", "upgrade_tower", "send_wave",
                               "upgrade_power", "sell_tower", "use_power", "point_tower"))''',
     '''SUPPORTED_ACTIONS = frozenset(("wait", "build_tower", "upgrade_tower", "send_wave",
                               "upgrade_power", "sell_tower", "use_power", "point_tower", "click_entity"))'''),
    ('''        elif action_name == "send_wave":
            if state.get("wave_ready") is not True:''',
     '''        elif action_name == "click_entity":
            entity, x, y = (native.get(key) for key in ("entity_id", "x", "y"))
            if not (_integer(entity) and _integer(x) and _integer(y)):
                continue
            action, cost = {"action": action_name, "entity_id": entity, "x": x, "y": y}, 0.0
            what = native.get("template") if isinstance(native.get("template"), str) else "entity"
            why = {"downed_boss": " to finish the downed boss", "tower_trap": " to free the trapped tower"}
            text = f"Click {what} {entity} at ({x},{y})" + why.get(native.get("kind"), "")
        elif action_name == "send_wave":
            if state.get("wave_ready") is not True:'''),
])

patch("alpharush_rl/search.py", [
    ('''    def choose(self, state: dict, menu: list[dict], context: dict) -> dict:
        validate_menu(menu)
        aim = aim_choice(self, state, menu)''',
     '''    def choose(self, state: dict, menu: list[dict], context: dict) -> dict:
        validate_menu(menu)
        clicks = [item for item in menu if item["action"].get("action") == "click_entity"]
        if clicks:
            # A click a script waits for (finish a downed boss, break ice off a tower) is always taken first.
            item = min(clicks, key=lambda i: (_number(i["action"].get("entity_id")), i["label"]))
            return _choice(self, item["label"], rule="click", entity_id=item["action"].get("entity_id"),
                           plan_cursor=self.cursor)
        aim = aim_choice(self, state, menu)'''),
])

patch("alpharush_rl/operator_net.py", [
    ('''ACTIONS = ("wait", "build_tower", "upgrade_tower", "upgrade_power", "sell_tower", "use_power", "send_wave",
           "point_tower")''',
     '''ACTIONS = ("wait", "build_tower", "upgrade_tower", "upgrade_power", "sell_tower", "use_power", "send_wave",
           "point_tower", "click_entity")
# Rows recorded before click_entity existed lack its one-hot column (the last action column).
LEGACY_ACTIONS = len(ACTIONS) - 1'''),
    ('''    elif name == "use_power":
        power = action.get("power")''',
     '''    elif name == "click_entity":
        where = {"x": action.get("x"), "y": action.get("y")}
    elif name == "use_power":
        power = action.get("power")'''),
    ('''        if data["g_dim"] != G_DIM or data["o_dim"] != O_DIM:
            raise ValueError("operator feature dimensions changed since these weights were trained")
        net = cls(tuple(data["hidden"]))
        net.params = [np.asarray(p, dtype=np.float32) for p in data["params"]]
        return net''',
     '''        legacy = data["g_dim"] == G_DIM and data["o_dim"] == O_DIM - 1
        if data["g_dim"] != G_DIM or (data["o_dim"] != O_DIM and not legacy):
            raise ValueError("operator feature dimensions changed since these weights were trained")
        net = cls(tuple(data["hidden"]))
        net.params = [np.asarray(p, dtype=np.float32) for p in data["params"]]
        if legacy:
            # Weights trained before click_entity: a zero input weight for its one-hot column.
            net.params[0] = np.insert(net.params[0], G_DIM + LEGACY_ACTIONS, 0.0, axis=0)
        return net'''),
    ('''    g, o, offsets, target = data["g"], data["o"], data["offsets"], data["target"]''',
     '''    g, o, offsets, target = data["g"], data["o"], data["offsets"], data["target"]
    if o.shape[1] == O_DIM - 1:
        o = np.insert(o, LEGACY_ACTIONS, 0.0, axis=1)  # recorded before click_entity existed'''),
])
print("click_entity patched")
