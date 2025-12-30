from datetime import datetime

from strands.types.state_serializers import JsonStateSerializer, PickleStateSerializer


def test_json_state_serializer_drops_non_serializable():
    serializer = JsonStateSerializer()
    state = {"a": 1, "dt": datetime.utcnow()}

    serialized = serializer.serialize(state)

    assert serialized == {"a": 1}


def test_pickle_state_serializer_round_trip():
    serializer = PickleStateSerializer()
    marker = object()
    state = {"a": 1, "obj": marker}

    serialized = serializer.serialize(state)
    restored = serializer.deserialize(serialized)

    assert restored["a"] == 1
    assert isinstance(restored["obj"], object)
