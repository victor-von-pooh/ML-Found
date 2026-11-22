import json
import pickle
from typing import Callable, Optional

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
        "補足", ["mujoco (2.0. 3.0 以上が要る)"]
    )

    # 作り直しても属性とメッセージが同じ
    copied = pickle.loads(pickle.dumps(error))
    assert str(copied) == str(error)
    assert copied.feature == "rl"
    assert copied.missing == ["gymnasium"]
    assert copied.note == "補足"
    assert copied.outdated == ["mujoco (2.0. 3.0 以上が要る)"]


@pytest.mark.parametrize(
    "feature, modules, package",
    [
        ("eda", ["ml_eda", "pyarrow"], "ml-eda[parquet] @ git+https://"
         "github.com/victor-von-pooh/ML-EDA.git"),
        ("modelcontrol", ["ml_modelcontrol"], "ml-modelcontrol @ git+"
         "https://github.com/victor-von-pooh/ML-ModelControl.git"),
        ("inference", ["ml_inference", "pyarrow"], "ml-inference @ git+"
         "https://github.com/victor-von-pooh/ML-Inference.git")
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
        {"kind": "module", "name": module, "ok": None, "error": None,
         "version": None, "min_version": None}
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


class FixedVersion:
    def __init__(self, version: Optional[str]):
        # 返す版の文字を保存
        self.version = version

    def __call__(self, distribution: str) -> Optional[str]:
        """
        どのパッケージにも決めた版を返す関数

        Parameters
        ----------
        distribution: str
            パッケージの名前

        Returns
        ----------
        version: Optional[str]
            決めた版の文字
        """
        # 決めた版
        version = self.version

        return version


@pytest.mark.parametrize(
    "version, minimum, expected",
    [
        ("1.0", "1.0", True),
        ("1.0.0", "1.0", True),
        ("1.2.3", "1.0", True),
        ("10.0", "9.9", True),
        ("0.29.1", "1.0", False),
        ("2.11.0+cpu", "2.2", True),
        ("2.1.0+cu121", "2.2", False),
        ("1.0.0rc1", "1.0", False),
        ("1.0.0a1", "1.0.0.dev1", True),
        ("1.0b2", "1.0rc1", False),
        ("1.0.dev3", "1.0", False),
        ("1.1.dev0", "1.0", True),
        ("1.0.post1", "1.0", True),
        ("1.0.post1.dev2", "1.0.post1", False),
        ("v1.0", "1.0", True),
        ("abc", "1.0", None)
    ]
)
def test_version_at_least(
    version: str, minimum: str, expected: Optional[bool]
) -> None:
    """
    版の比べ方が PEP 440 の順に従い, ローカルの版を比べないことを
    確かめる関数

    Parameters
    ----------
    version: str
        入っている版の文字
    minimum: str
        版の下限
    expected: Optional[bool]
        期待する結果 (読めない版は None)

    Returns
    ----------
    None
    """
    assert requirements.version_at_least(version, minimum) is expected


def test_require_outdated_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    版が下限より古い場合に, 入れ方を添えて止まることを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        gymnasium があり, 版が古いことにするのに使う

    Returns
    ----------
    None
    """
    monkeypatch.setattr(requirements, "module_found", FoundAll())
    monkeypatch.setattr(
        requirements, "module_version", FixedVersion("0.29.1")
    )

    # 版が足りないことと, 入っていない時と同じ入れ方の案内
    with pytest.raises(MissingRequirementError) as info:
        require("rl")
    message = str(info.value)
    assert "版が足りません: gymnasium (0.29.1. 1.0 以上が要る)" in message
    assert "入っていません" not in message
    assert 'pip install "gymnasium>=1.0"' in message
    assert 'pip install -e "./ML-Found[rl]"' in message
    assert info.value.missing == []
    assert info.value.outdated == ["gymnasium (0.29.1. 1.0 以上が要る)"]

    # 一覧では入っていない判定で, 入っている版と下限と説明を返す
    entry = describe_requirements(["rl"])[0]
    assert entry["available"] is False
    check = entry["checks"][0]
    assert check["ok"] is False
    assert check["version"] == "0.29.1"
    assert check["min_version"] == "1.0"
    assert "1.0 以上が要る" in check["error"]


def test_require_missing_and_outdated(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    無いものと版が足りないものが両方ある場合に, 分けて案内することを
    確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        mujoco を隠し, gymnasium の版が古いことにするのに使う

    Returns
    ----------
    None
    """
    monkeypatch.setattr(
        requirements, "module_found", HiddenModules(FoundAll(), {"mujoco"})
    )
    monkeypatch.setattr(requirements, "module_version", FixedVersion("0.28"))

    # 2 つの行で, 入れ方は 1 回
    with pytest.raises(MissingRequirementError) as info:
        require("mujoco")
    message = str(info.value)
    assert "入っていません: mujoco" in message
    assert "版が足りません: gymnasium (0.28. 1.0 以上が要る)" in message
    assert message.count("次の一手") == 1


@pytest.mark.parametrize("version", [None, "unknown-version"])
def test_unreadable_version_is_available(
    monkeypatch: pytest.MonkeyPatch, version: Optional[str]
) -> None:
    """
    版を読めない場合は止めず, 確かめていないことを残すことを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        gymnasium があり, 版を読めないことにするのに使う
    version: Optional[str]
        読めた版の文字 (パッケージの情報が無い場合は None)

    Returns
    ----------
    None
    """
    monkeypatch.setattr(requirements, "module_found", FoundAll())
    monkeypatch.setattr(requirements, "module_version", FixedVersion(version))

    # 使えるものとして扱い, 説明だけ残す
    require("rl")
    check = describe_requirements(["rl"])[0]["checks"][0]
    assert check["ok"] is True
    assert "確かめていない" in check["error"]


def test_register_requirement_min_version(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    プロジェクトの機能にも版の下限と版を読むパッケージの名前を書け,
    誤った書き方は止めることを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        版を決めるのに使う

    Returns
    ----------
    None
    """
    monkeypatch.setattr(requirements, "module_found", FoundAll())
    calls = []
    monkeypatch.setattr(requirements, "module_version", RecordVersion(calls))
    register_requirement(
        "fake", "版の要るもの",
        [{"kind": "module", "name": "fake_module", "min_version": "2.0",
          "distribution": "fake-package"}],
        {"kind": "pip", "packages": ["fake-package>=2.0"]}
    )
    try:
        with pytest.raises(MissingRequirementError, match="版が足りません"):
            require("fake")
        assert calls == ["fake-package"]
    finally:
        unregister_requirement("fake")

    # module 以外の版の下限と, 読めない版の文字はエラー
    pip = {"kind": "pip", "packages": ["x"]}
    with pytest.raises(ValueError, match="module の確かめ方だけ"):
        register_requirement(
            "a", "", [{"kind": "command", "name": "x", "min_version": "1"}],
            pip
        )
    with pytest.raises(ValueError, match="版の文字"):
        register_requirement(
            "a", "", [{"kind": "module", "name": "x", "min_version": "new"}],
            pip
        )


class FoundAll:
    def __call__(self, name: str) -> bool:
        """
        どのモジュールもあることにする関数

        Parameters
        ----------
        name: str
            モジュールの名前

        Returns
        ----------
        found: bool
            いつも True
        """
        # ある印
        found = True

        return found


class RecordVersion:
    def __init__(self, calls: list[str]):
        # 呼ばれたパッケージの名前を貯めるリストを保存
        self.calls = calls

    def __call__(self, distribution: str) -> str:
        """
        呼ばれたパッケージの名前を残し, 古い版を返す関数

        Parameters
        ----------
        distribution: str
            パッケージの名前

        Returns
        ----------
        version: str
            古い版の文字
        """
        # 名前の記録と古い版
        self.calls.append(distribution)
        version = "1.5"

        return version
