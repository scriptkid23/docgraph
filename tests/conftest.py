import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path / "boostmcp_data"
