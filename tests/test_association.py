from ppe.rules.association import associate
from ppe.types import Detection

ROLES = {"helmet": "helmet", "no_helmet": "no_helmet", "vest": "vest", "no_vest": "no_vest"}
CFG = {"min_ioa": 0.5, "zones": {"head": [0.0, 0.4], "torso": [0.15, 0.8]}}


def det(cls, box, conf=0.9):
    return Detection(cls, conf, box)


PERSON = det("person", (100, 100, 200, 400))   # alto 300


def test_helmet_in_head_zone_is_associated():
    res, assigned = associate([PERSON], [det("helmet", (130, 100, 170, 140))], ROLES, CFG)
    assert res[0].helmet == 0.9 and assigned == {0: 0}


def test_helmet_at_feet_is_not_associated():
    res, assigned = associate([PERSON], [det("helmet", (130, 360, 170, 395))], ROLES, CFG)
    assert res[0].helmet == 0.0 and assigned == {}


def test_vest_in_torso_zone_and_outside_person_rejected():
    res, _ = associate([PERSON], [det("vest", (110, 180, 190, 300))], ROLES, CFG)
    assert res[0].vest > 0
    res, _ = associate([PERSON], [det("vest", (300, 180, 380, 300))], ROLES, CFG)
    assert res[0].vest == 0


def test_low_ioa_rejected():
    # casco apenas rozando el borde de la persona: < 50 % de su área dentro
    res, _ = associate([PERSON], [det("helmet", (180, 100, 240, 140))], ROLES, CFG)
    assert res[0].helmet == 0.0


def test_each_ppe_goes_to_one_person_only():
    p1, p2 = det("person", (100, 100, 200, 400)), det("person", (120, 100, 220, 400))
    res, assigned = associate([p1, p2], [det("helmet", (135, 100, 185, 140))], ROLES, CFG)
    assert len(assigned) == 1
    assert sum(r.helmet > 0 for r in res) == 1


def test_negative_classes_use_same_zones():
    res, _ = associate([PERSON], [det("no_helmet", (130, 100, 170, 140), 0.7)], ROLES, CFG)
    assert res[0].no_helmet == 0.7 and res[0].helmet == 0.0


def test_keeps_max_confidence():
    items = [det("helmet", (130, 100, 170, 140), 0.5), det("helmet", (131, 101, 171, 141), 0.8)]
    res, _ = associate([PERSON], items, ROLES, CFG)
    assert res[0].helmet == 0.8
