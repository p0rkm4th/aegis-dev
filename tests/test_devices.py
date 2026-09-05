import json
from datetime import datetime, timezone

import pytest

from aegis.devices import (
    HomeAssistantAdapter,
    HomeAssistantRestControlGateway,
    HomeAssistantRestGateway,
)


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, _limit):
        return self.payload


def test_home_assistant_rest_gateway_is_bounded_and_read_only(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["authorization"] = request.get_header("Authorization")
        seen["timeout"] = timeout
        return _Response([{"entity_id": "light.lamp", "state": "on", "attributes": {}}])

    monkeypatch.setattr("aegis.devices.urlopen", fake_urlopen)
    gateway = HomeAssistantRestGateway("http://ha.test:8123", "secret", timeout=2)
    states = HomeAssistantAdapter(gateway, policy=lambda _: False).read_states(
        datetime.now(timezone.utc)
    )

    assert states[0].entity_id == "light.lamp"
    assert seen == {
        "url": "http://ha.test:8123/api/states",
        "authorization": "Bearer secret",
        "timeout": 2,
    }
    with pytest.raises(PermissionError):
        gateway.call_service({"entity_id": "light.lamp"})


def test_home_assistant_rest_gateway_rejects_non_http_urls():
    with pytest.raises(ValueError):
        HomeAssistantRestGateway("file:///tmp/ha", "secret")


def test_home_assistant_control_gateway_posts_only_allowlisted_service(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["method"] = request.method
        seen["body"] = json.loads(request.data)
        seen["authorization"] = request.get_header("Authorization")
        seen["timeout"] = timeout
        return _Response([])

    monkeypatch.setattr("aegis.devices.urlopen", fake_urlopen)
    gateway = HomeAssistantRestControlGateway("https://ha.test:8123", "secret", timeout=2)
    gateway.call_service({"entity_id": "light.lamp", "service": "turn_on"})
    assert seen == {
        "url": "https://ha.test:8123/api/services/light/turn_on",
        "method": "POST",
        "body": {"entity_id": "light.lamp"},
        "authorization": "Bearer secret",
        "timeout": 2,
    }
