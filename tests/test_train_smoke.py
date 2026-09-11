"""End-to-end smoke: each agent variant trains, evaluates, checkpoints, resumes."""
import os

from selfplay_lab import train as T


def _run(agent, ck, extra=()):
    argv = ["--game", "leduc_poker(players=3)", "--agent", agent,
            "--episodes", "60", "--eval-every", "30", "--eval-games", "5",
            "--min-buffer", "50", "--checkpoint", ck, "--quiet", *extra]
    return T.main(argv)


def test_stock_distil_search_all_run(tmp_path):
    for agent, extra in (("stock", ()), ("distil", ()),
                         ("search", ("--search-sims", "4"))):
        ck = str(tmp_path / f"{agent}.pt")
        hist = _run(agent, ck, extra)
        assert os.path.exists(ck)
        assert [h["episode"] for h in hist] == [30, 60]
        for h in hist:
            assert 0.0 <= h["vs_random_pooled"] <= 1.0
            assert abs(h["fair_line"] - 1 / 3) < 1e-3


def test_resume_continues_episode_count(tmp_path):
    ck = str(tmp_path / "r.pt")
    _run("stock", ck)
    argv = ["--game", "leduc_poker(players=3)", "--agent", "stock",
            "--episodes", "90", "--eval-every", "30", "--eval-games", "5",
            "--min-buffer", "50", "--checkpoint", ck, "--quiet", "--resume"]
    hist = T.main(argv)
    assert [h["episode"] for h in hist] == [30, 60, 90]
