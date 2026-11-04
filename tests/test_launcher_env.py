from soccer_launcher.env import load_dotenv, parse_dotenv


def test_parse_dotenv_handles_comments_quotes_and_exports():
    text = """
    # comment
    PLAIN=value
    export EXPORTED=two words
    QUOTED="two words"
    SINGLE='literal # value'
    HASH=value # trailing comment
    WEIRD=not a comment#kept
    EMPTY=
    INVALID KEY=nope
    NOVALUE
    """
    assert parse_dotenv(text) == {
        "PLAIN": "value",
        "EXPORTED": "two words",
        "QUOTED": "two words",
        "SINGLE": "literal # value",
        "HASH": "value",
        "WEIRD": "not a comment#kept",
        "EMPTY": "",
    }


def test_load_dotenv_does_not_override_existing_values(tmp_path):
    path = tmp_path / ".env"
    path.write_text("A=from-file\nB=from-file\n")
    env = {"A": "already-set"}
    load_dotenv(path, env)
    assert env == {"A": "already-set", "B": "from-file"}


def test_load_dotenv_ignores_missing_file(tmp_path):
    env = {}
    load_dotenv(tmp_path / "missing.env", env)
    assert env == {}
