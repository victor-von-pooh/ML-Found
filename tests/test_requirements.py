import json
import pickle
from typing import Callable

import pytest

from ml_found import describe_requirements, environment_packages, require
from ml_found.utils import requirements
from ml_found.utils.requirements import (
    MissingRequirementError, register_requirement, unregister_requirement
)


def hide_modules(
    monkeypatch: pytest.MonkeyPatch, hidden: set[str]
) -> None:
    """
    決めたモジュールが入っていないように見せる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        差し替えに使う
    hidden: set[str]
        入っていないように見せるモジュールの名前

    Returns
    ----------
    None
    """
    # 元の確かめ方を残し, 隠す名前だけ見つからない扱い
    original = requirements.module_found
    monkeypatch.setattr(
        requirements, "module_found",
        HiddenModules(original, hidden)
    )


class HiddenModules:
    def __init__(self, original: Callable[[str], bool], hidden: set[str]):
        # 元の関数と隠す名前を保存
        self.original = original
        self.hidden = hidden

    def __call__(self, name: str) -> bool:
        """
        隠す名前なら見つからないとし, それ以外は元の関数で確かめる関数

        Parameters
        ----------
        name: str
            モジュールの名前

        Returns
        ----------
        found: bool
            見つかった場合は True
        """
        # 隠す名前の判定
        if name in self.hidden:
            found = False
            return found
        found = self.original(name)

        return found


def test_describe_lists_seeded_features() -> None:
    """
    登録簿に rl, mujoco, ros2, cuda があり, JSON にできることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    entries = {entry["name"]: entry for entry in describe_requirements()}

    # 4 つの機能と, pip か公式の手順の入れ方
    assert {"rl", "mujoco", "ros2", "cuda"} <= set(entries)
    assert entries["rl"]["install"]["kind"] == "pip"
    assert entries["rl"]["install"]["extra"] == "rl"
    command = entries["rl"]["install"]["command"]
    assert command == 'pip install "gymnasium>=1.0"'
    assert entries["mujoco"]["install"]["extra"] == "mujoco"
    assert entries["mujoco"]["note"]
    assert entries["ros2"]["install"]["kind"] == "guide"
    assert entries["ros2"]["install"]["url"].startswith("https://")
    assert entries["ros2"]["match"] == "any"
    assert entries["cuda"]["install"]["kind"] == "guide"
    assert entries["cuda"]["checks"][0]["kind"] == "function"

    # 確かめた結果は真偽値で, 関数そのものは含まず JSON にできる
    assert all(
        isinstance(entry["available"], bool) for entry in entries.values()
    )
    json.dumps(describe_requirements())


def test_describe_without_check() -> None:
    """
    確かめない場合は結果が None になることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    entries = describe_requirements(["rl"], check=False)

    # 指定した機能だけで, 確かめた結果は無い
    assert [entry["name"] for entry in entries] == ["rl"]
    assert entries[0]["available"] is None
    assert entries[0]["checks"][0]["ok"] is None


def test_unknown_feature() -> None:
    """
    知らない機能の名前は登録済みの名前を添えてエラーになることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    with pytest.raises(ValueError, match="登録済み"):
        require("unknown")
    with pytest.raises(ValueError, match="unknown"):
        describe_requirements(["unknown"])


def test_require_missing_pip_package(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    pip で入るものが無い場合に, コマンドを添えて止まることを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        gymnasium を隠すのに使う

    Returns
    ----------
    None
    """
    hide_modules(monkeypatch, {"gymnasium"})

    # 足りないものと, pip と追加の依存のコマンドをメッセージに含む
    with pytest.raises(MissingRequirementError) as info:
        require("rl")
    message = str(info.value)
    assert "gymnasium" in message
    assert 'pip install "gymnasium>=1.0"' in message
    assert 'pip install -e "./ML-Found[rl]"' in message
    assert info.value.feature == "rl"
    assert info.value.missing == ["gymnasium"]
    assert isinstance(info.value, ImportError)

    # 一覧でも入っていない判定
    entry = describe_requirements(["rl"])[0]
    assert entry["available"] is False


def test_require_missing_guide(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    pip で入らないものが無い場合に, 公式の手順へ案内することを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        rclpy と ros2 のコマンドを隠すのに使う

    Returns
    ----------
    None
    """
    hide_modules(monkeypatch, {"rclpy"})
    monkeypatch.setattr(requirements.shutil, "which", NoCommand())

    # pip のコマンドではなく URL と補足
    with pytest.raises(MissingRequirementError) as info:
        require("ros2")
    message = str(info.value)
    assert "https://docs.ros.org/" in message
    assert "pip install" not in message
    assert "補足" in message


def test_ros2_any_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    ros2 はコマンドか rclpy のどちらかがあれば入っているとすることを
    確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        rclpy を隠し, コマンドがあることにするのに使う

    Returns
    ----------
    None
    """
    hide_modules(monkeypatch, {"rclpy"})
    monkeypatch.setattr(requirements.shutil, "which", SomeCommand())

    # コマンドだけでも入っている判定
    require("ros2")
    entry = describe_requirements(["ros2"])[0]
    assert entry["available"] is True
    assert [item["ok"] for item in entry["checks"]] == [False, True]


class NoCommand:
    def __call__(self, name: str) -> None:
        """
        どのコマンドも無いことにする関数

        Parameters
        ----------
        name: str
            コマンドの名前

        Returns
        ----------
        path: None
            無いことを表す None
        """
        # コマンドが無い印
        path = None

        return path


class SomeCommand:
    def __call__(self, name: str) -> str:
        """
        どのコマンドもあることにする関数

        Parameters
        ----------
        name: str
            コマンドの名前

        Returns
        ----------
        path: str
            仮のパス
        """
        # コマンドがある印の仮のパス
        path = f"/bin/{name}"

        return path


def test_cuda_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    cuda の確かめ方が torch.cuda.is_available に従うことを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        torch.cuda.is_available を差し替えるのに使う

    Returns
    ----------
    None
    """
    # 使えない場合は公式の手順を添えたエラー
    monkeypatch.setattr(requirements.torch.cuda, "is_available", NoCuda())
    with pytest.raises(MissingRequirementError, match="pytorch.org"):
        require("cuda")

    # 確かめる関数のエラーは使えないとして記録
    monkeypatch.setattr(requirements.torch.cuda, "is_available", Broken())
    entry = describe_requirements(["cuda"])[0]
    assert entry["available"] is False
    assert "RuntimeError" in entry["checks"][0]["error"]


class NoCuda:
    def __call__(self) -> bool:
        """
        CUDA を使えないことにする関数

        Parameters
        ----------
        None

        Returns
        ----------
        available: bool
            いつも False
        """
        # 使えない印
        available = False

        return available


class Broken:
    def __call__(self) -> bool:
        """
        いつもエラーを出す関数

        Parameters
        ----------
        None

        Returns
        ----------
        result: bool
            返らない
        """
        # 確かめ方のエラーの再現
        raise RuntimeError("壊れた確かめ方")


def test_register_requirement() -> None:
    """
    プロジェクトの機能を登録して一覧と require に使えることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    register_requirement(
        "fake", "入っていないもの",
        [{"kind": "module", "name": "no_such_module_for_ml_found"}],
        {"kind": "pip", "packages": ["no-such-package"]}
    )
    try:
        entry = describe_requirements(["fake"])[0]
        assert entry["available"] is False
        with pytest.raises(MissingRequirementError, match="no-such-package"):
            require("fake")
    finally:
        unregister_requirement("fake")
    assert "fake" not in {entry["name"] for entry in describe_requirements()}


def test_register_requirement_errors() -> None:
    """
    登録の書き方の誤りと, ML-Found の機能の上書きを止めることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    module = [{"kind": "module", "name": "x"}]
    with pytest.raises(ValueError, match="上書き"):
        register_requirement("rl", "", module, {"kind": "pip",
                                                 "packages": ["x"]})
    with pytest.raises(ValueError, match="1 つ以上"):
        register_requirement("a", "", [], {"kind": "pip", "packages": ["x"]})
    with pytest.raises(ValueError, match="packages"):
        register_requirement("a", "", module, {"kind": "pip"})
    with pytest.raises(ValueError, match="url"):
        register_requirement("a", "", module, {"kind": "guide"})
    with pytest.raises(ValueError, match="function"):
        register_requirement(
            "a", "", [{"kind": "function", "name": "f"}],
            {"kind": "guide", "url": "https://example.com/"}
        )
    with pytest.raises(ValueError, match="除けません"):
        unregister_requirement("cuda")


def test_missing_requirement_error_pickles() -> None:
    """
    MissingRequirementError が pickle で送れることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    error = MissingRequirementError(
        "rl", ["gymnasium"],
        {"kind": "pip", "extra": "rl", "packages": ["gymnasium>=1.0"],
         "url": None},
        "補足"
    )

    # 作り直しても属性とメッセージが同じ
    copied = pickle.loads(pickle.dumps(error))
    assert str(copied) == str(error)
    assert copied.feature == "rl"
    assert copied.missing == ["gymnasium"]
    assert copied.note == "補足"


@pytest.mark.parametrize(
    "feature, modules, package",
    [
        ("eda", ["ml_eda", "pyarrow"], "ml-eda[parquet] @ git+https://"
         "github.com/victor-von-pooh/ML-EDA.git"),
        ("modelcontrol", ["ml_modelcontrol"], "ml-modelcontrol @ git+"
         "https://github.com/victor-von-pooh/ML-ModelControl.git"),
        ("inference", ["ml_inference"], "ml-inference @ git+https://"
         "github.com/victor-von-pooh/ML-Inference.git")
    ]
)
def test_ml_library_features(
    monkeypatch: pytest.MonkeyPatch, feature: str, modules: list[str],
    package: str
) -> None:
    """
    段階ごとのライブラリの機能が, import の名前で確かめ, git から pip で
    入れる案内になることを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        モジュールを隠すのに使う
    feature: str
        機能の名前
    modules: list[str]
        確かめる import の名前 (ML-EDA は parquet を読む pyarrow も)
    package: str
        pip に渡す指定

    Returns
    ----------
    None
    """
    entry = describe_requirements([feature], check=False)[0]

    # import の名前の確かめ方と, git の URL の pip の入れ方
    assert entry["checks"] == [
        {"kind": "module", "name": module, "ok": None, "error": None}
        for module in modules
    ]
    assert entry["install"]["kind"] == "pip"
    assert entry["install"]["packages"] == [package]
    assert entry["install"]["command"] == f'pip install "{package}"'
    assert entry["install"]["extra"] is None
    assert entry["install"]["extra_command"] is None

    # 無ければ pip のコマンドを添えて止まる
    hide_modules(monkeypatch, set(modules))
    with pytest.raises(MissingRequirementError) as error:
        require(feature)
    assert error.value.missing == modules
    assert f'pip install "{package}"' in str(error.value)


def test_environment_packages() -> None:
    """
    環境に入れる pip の指定が, ML-Found 自身と pip で入る機能の順になる
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    packages = environment_packages(["eda", "rl", "cuda", "mujoco"])

    # ML-Found が先頭, 同じ指定は 1 回, pip で入らない cuda は含めない
    assert packages == [
        "ml-found @ git+https://github.com/victor-von-pooh/ML-Found.git",
        "ml-eda[parquet] @ git+https://github.com/victor-von-pooh/ML-EDA.git",
        "gymnasium>=1.0", "mujoco"
    ]

    # ML-Found 自身を除く場合と, 何も要らない場合
    assert environment_packages(["inference"], include_found=False) == [
        "ml-inference @ git+https://github.com/victor-von-pooh/"
        "ML-Inference.git"
    ]
    assert environment_packages([], include_found=False) == []

    # 登録簿に無い名前はエラー
    with pytest.raises(ValueError, match="登録済み"):
        environment_packages(["eda", "unknown"])
