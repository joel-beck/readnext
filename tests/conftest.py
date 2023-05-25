from pathlib import Path


def filepath_to_modules(filepath: Path) -> str:
    return str(filepath).replace("/", ".").replace(".py", "")


def list_fixture_modules() -> list[str]:
    """
    List the relative paths (`tests/fixtures/*.py`) of all fixture files and display the
    filepaths as module paths.

    Example: - relative filepath: `tests/fixtures/test_data.py` - module path:
    `tests.fixtures.test_data`
    """
    root_path = Path(__file__).parent.parent
    tests_dirpath = root_path / "tests"
    fixtures_dirpath = tests_dirpath / "fixtures"

    return [
        filepath_to_modules(fixture_filepath.relative_to(root_path))
        for fixture_filepath in fixtures_dirpath.glob("*.py")
        # exclude `__init__.py` files
        if "__" not in str(fixture_filepath)
    ]


pytest_plugins = list_fixture_modules()
