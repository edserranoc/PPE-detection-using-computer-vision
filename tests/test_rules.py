from pathlib import Path

import yaml

from ppe.rules.association import PersonPPE
from ppe.rules.events import EventManager, PersonObs, condition_key, missing_ppe, resolve_event

RULES = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "rules.yaml").read_text(encoding="utf-8"))
EV = RULES["events"]
BOX = (100.0, 100.0, 200.0, 400.0)


def test_severity_table_from_config():
    assert resolve_event("sin_chaleco", EV) == ("persona_sin_chaleco", "media")
    assert resolve_event("sin_casco", EV) == ("persona_sin_casco", "alta")
    assert resolve_event("sin_casco_y_chaleco", EV) == ("persona_sin_casco_y_chaleco", "critica")


def test_condition_key_combinations():
    assert condition_key(False, False) is None
    assert condition_key(False, True) == "sin_chaleco"
    assert condition_key(True, False) == "sin_casco"
    assert condition_key(True, True) == "sin_casco_y_chaleco"


def test_strategy_association():
    assert missing_ppe(PersonPPE(), "association") == (True, True)
    assert missing_ppe(PersonPPE(helmet=0.8, vest=0.6), "association") == (False, False)
    assert missing_ppe(PersonPPE(no_helmet=0.9, helmet=0.5), "association") == (False, True)  # ignora no_*


def test_strategy_explicit():
    assert missing_ppe(PersonPPE(), "explicit") == (False, False)   # sin evidencia explícita
    assert missing_ppe(PersonPPE(no_helmet=0.8), "explicit") == (True, False)
    assert missing_ppe(PersonPPE(no_helmet=0.6, helmet=0.9), "explicit") == (False, False)


def test_strategy_hybrid():
    assert missing_ppe(PersonPPE(), "hybrid") == (True, True)
    assert missing_ppe(PersonPPE(helmet=0.9, vest=0.8), "hybrid") == (False, False)
    assert missing_ppe(PersonPPE(helmet=0.5, no_helmet=0.8, vest=0.9), "hybrid") == (True, False)
    assert missing_ppe(PersonPPE(helmet=0.8, no_helmet=0.5, vest=0.9), "hybrid") == (False, False)


def obs(cond="sin_casco", inside=True, key=1, bbox=BOX):
    return PersonObs(key, key, bbox, 0.9, f"pred-{key}", cond, inside, "Z")


def manager(min_frames=3, min_seconds=0.0, gap=1, cooldown=30.0):
    return EventManager({"min_frames": min_frames, "min_seconds": min_seconds, "max_gap_frames": gap},
                        {**EV, "cooldown_seconds": cooldown}, "1.0.0", "run")


def feed(m, frames, source="v1"):
    out = []
    for f, persons in frames:
        out += m.update(source, "CAM_01", f, f / 1.0, persons)
    return out


def test_person_outside_roi_generates_no_event():
    m = manager(min_frames=1)
    assert feed(m, [(0, [obs(inside=False)])]) == []


def test_no_condition_no_event():
    m = manager(min_frames=1)
    assert feed(m, [(0, [obs(cond=None)])]) == []


def test_persistence_requires_min_frames_and_emits_once():
    m = manager(min_frames=3)
    events = feed(m, [(i, [obs()]) for i in range(10)])
    assert len(events) == 1
    assert events[0]["frame_id"] == 2 and events[0]["start_frame_id"] == 0
    assert events[0]["event_type"] == "persona_sin_casco" and events[0]["severity"] == "alta"
    assert events[0]["latency_seconds"] == 2.0


def test_not_enough_frames_no_event():
    m = manager(min_frames=3)
    assert feed(m, [(0, [obs()]), (1, [obs()])]) == []


def test_min_seconds_is_enforced():
    m = manager(min_frames=1, min_seconds=2.0)
    events = feed(m, [(i, [obs()]) for i in range(5)])
    assert len(events) == 1 and events[0]["frame_id"] == 2


def test_gap_tolerance_keeps_streak_and_long_gap_resets():
    m = manager(min_frames=3, gap=1)
    ev = feed(m, [(0, [obs()]), (1, [obs()]), (3, [obs()])])   # hueco de 1 frame: tolerado
    assert len(ev) == 1
    m = manager(min_frames=3, gap=1)
    ev = feed(m, [(0, [obs()]), (1, [obs()]), (5, [obs()])])   # hueco largo: reinicia
    assert ev == []


def test_cooldown_blocks_repeat_for_same_track_and_type():
    m = manager(min_frames=1, cooldown=30.0)
    ev = feed(m, [(0, [obs()]), (10, [obs()])])                # reaparece tras >gap, dentro del cooldown
    assert len(ev) == 1 and m.stats["suppressed_cooldown"] == 1


def test_reappearance_after_cooldown_generates_new_event():
    m = manager(min_frames=1, cooldown=5.0)
    ev = feed(m, [(0, [obs()]), (10, [obs()])])
    assert len(ev) == 2


def test_track_id_switch_is_merged():
    m = manager(min_frames=1)
    ev = feed(m, [(0, [obs(key=1)]), (1, [obs(key=2, bbox=(102, 101, 201, 401))])])
    assert len(ev) == 1 and m.stats["suppressed_track_switch"] == 1


def test_different_people_same_type_are_not_merged():
    m = manager(min_frames=1)
    ev = feed(m, [(0, [obs(key=1), obs(key=2, bbox=(600, 100, 700, 400))])])
    assert len(ev) == 2


def test_escalation_to_critical_is_a_new_event_type():
    m = manager(min_frames=1, cooldown=0.0)
    ev = feed(m, [(0, [obs(cond="sin_casco")]), (1, [obs(cond="sin_casco_y_chaleco")])])
    assert [e["severity"] for e in ev] == ["alta", "critica"]


def test_end_source_releases_state():
    m = manager(min_frames=1)
    feed(m, [(0, [obs()])], source="a")
    m.end_source("a")
    assert not m._states and not m._last_emit and not m._recent
