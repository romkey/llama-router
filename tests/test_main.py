from __future__ import annotations

import pytest

from llama_router import __version__
from llama_router.__main__ import main


def test_version_prints_and_exits(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"llama-router {__version__}"
