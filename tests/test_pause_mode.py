"""--sglang-pause-generation-mode: pause payload and cache flush around weight updates."""

from types import SimpleNamespace

import pytest


@pytest.mark.unit
@pytest.mark.parametrize("mode,payload", [("abort", {}), ("in_place", {"mode": "in_place"}), ("retract", {"mode": "retract"})])
def test_pause_generation_sends_mode(monkeypatch, mode, payload):
    from slime.backends.sglang_utils import sglang_engine

    sent = {}

    class _Response:
        def raise_for_status(self):
            return None

    def fake_post(url, json):
        sent["url"], sent["json"] = url, json
        return _Response()

    monkeypatch.setattr(sglang_engine.requests, "post", fake_post)
    engine = SimpleNamespace(args=SimpleNamespace(sglang_pause_generation_mode=mode), server_host="h", server_port=1)
    sglang_engine.SGLangEngine.pause_generation(engine)
    assert sent == {"url": "http://h:1/pause_generation", "json": payload}


@pytest.mark.unit
def test_flush_cache_skipped_only_in_place():
    from slime.backends.megatron_utils.update_weight.pause_mode import flush_cache_after_pause

    assert flush_cache_after_pause(SimpleNamespace())
    assert flush_cache_after_pause(SimpleNamespace(sglang_pause_generation_mode="abort"))
    assert flush_cache_after_pause(SimpleNamespace(sglang_pause_generation_mode="retract"))
    assert not flush_cache_after_pause(SimpleNamespace(sglang_pause_generation_mode="in_place"))
