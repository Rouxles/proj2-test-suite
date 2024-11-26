import unittest

def _test_id_to_name(test: unittest.TestCase) -> str:
    parts = test.id().split(".")
    assert len(parts) == 3, f"Unexpected test id: {test.id()}"
    return parts[2]