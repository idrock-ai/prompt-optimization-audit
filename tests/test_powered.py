import collections

from src.data import replication_all, replication_onatili


def test_replication_all_covers_four_subjects():
    """E4 used only the ona_tili slice. The powered CROSS-subject test needs the other
    three as well, or the differential odds ratio has no comparison stratum."""
    ex = replication_all()
    by = collections.Counter(e.subject for e in ex)
    assert set(by) == {"ona_tili", "tarix", "matematika", "fizika"}
    assert by["ona_tili"] == len(replication_onatili())
    assert len(ex) == sum(by.values()) > 1500


def test_capping_never_thins_the_native_subject():
    """The primary endpoint is measured on the native subject, so a runtime cap must
    leave it whole no matter how aggressive it is."""
    full = collections.Counter(e.subject for e in replication_all())
    capped = collections.Counter(e.subject for e in replication_all(cap_nonnative=50))
    assert capped["ona_tili"] == full["ona_tili"]
    for s in ("tarix", "matematika", "fizika"):
        assert capped[s] == 50


def test_cap_above_pool_size_is_a_no_op():
    full = collections.Counter(e.subject for e in replication_all())
    capped = collections.Counter(e.subject for e in replication_all(cap_nonnative=10_000))
    assert capped == full


def test_capping_is_deterministic_and_seed_dependent():
    a = [e.qid for e in replication_all(cap_nonnative=40, seed=1)]
    b = [e.qid for e in replication_all(cap_nonnative=40, seed=1)]
    c = [e.qid for e in replication_all(cap_nonnative=40, seed=2)]
    assert a == b
    assert a != c


def test_replication_items_carry_the_fields_the_analyses_need():
    """analysis/interaction.py keys on qid and subject; a missing qid would silently
    collapse items together and fabricate flips."""
    for e in replication_all(cap_nonnative=5)[:20]:
        assert e.qid and e.qid.startswith("pub")
        assert e.subject
        assert str(e.answer_letter).strip().upper() in "ABCD"
        assert e.question and e.options
