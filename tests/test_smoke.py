def test_import():
    import jkb
    assert jkb.__version__ == "0.1.0"


def test_cli_loads():
    from jkb.cli import app
    assert app is not None
