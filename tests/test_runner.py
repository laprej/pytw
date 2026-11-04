import importlib.metadata

import pytest

from pytw.runner import main


def test_version(capsys):
    with pytest.raises(SystemExit) as exit:
        main(["--version"])
    assert exit.value.code == 0
    assert importlib.metadata.version("pytw") in capsys.readouterr().out
