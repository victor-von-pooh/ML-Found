import copy
import importlib.util
import shutil
from typing import Any, Optional

import torch


class MissingRequirementError(ImportError):
    def __init__(
        self, feature: str, missing: list[str], install: dict,
        note: Optional[str] = None
    ):
        # 親クラスのコンストラクタを呼び出す
        super(MissingRequirementError, self).__init__(
            format_missing(feature, missing, install, note)
        )

        # 機能の名前, 足りないもの, 入れ方, 補足を保存
        self.feature = feature
        self.missing = missing
        self.install = install
        self.note = note


def cuda_available() -> bool:
    """
    PyTorch から CUDA の GPU を使えるかを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    available: bool
        使える場合は True
    """
    # CUDA 付きの PyTorch と, 見える GPU の確認
    available = bool(torch.cuda.is_available())

    return available


# 機能ごとの要るもの, 確かめ方, 入れ方の登録簿
# checks の kind は module (import できるか), command (コマンドがあるか),
# function (関数が True を返すか). match は all (全て要る) か any (どれか 1 つ)
REQUIREMENTS: dict[str, dict[str, Any]] = {
    "rl": {
        "description": "強化学習の環境 (Gymnasium の形の環境を作る)",
        "checks": [{"kind": "module", "name": "gymnasium"}],
        "match": "all",
        "install": {
            "kind": "pip",
            "extra": "rl",
            "packages": ["gymnasium>=1.0"],
            "url": "https://gymnasium.farama.org/introduction/basic_usage/"
        },
        "note": None
    },
    "mujoco": {
        "description": "MuJoCo の物理シミュレータ (Gymnasium の MuJoCo の環境)",
        "checks": [
            {"kind": "module", "name": "gymnasium"},
            {"kind": "module", "name": "mujoco"}
        ],
        "match": "all",
        "install": {
            "kind": "pip",
            "extra": "mujoco",
            "packages": ["gymnasium>=1.0", "mujoco"],
            "url": "https://mujoco.readthedocs.io/en/stable/python.html"
        },
        "note": (
            "pip で入るが, 画面への描画には OpenGL が要る. 画面の無い計算機で"
            "描画する場合は, 公式の手順の MUJOCO_GL (egl, osmesa) の設定を見る"
        )
    },
    "ros2": {
        "description": "ROS 2 (ロボットの仕組みとのやり取り. rclpy)",
        "checks": [
            {"kind": "module", "name": "rclpy"},
            {"kind": "command", "name": "ros2"}
        ],
        "match": "any",
        "install": {
            "kind": "guide",
            "extra": None,
            "packages": [],
            "url": "https://docs.ros.org/"
        },
        "note": (
            "pip では入らない. 公式の手順で版 (distribution) を選んで入れ, "
            "使う端末で setup のスクリプトを source してから Python を起動する"
        )
    },
    "cuda": {
        "description": "CUDA の GPU で学習する",
        "checks": [
            {"kind": "function", "name": "torch.cuda.is_available",
             "function": cuda_available}
        ],
        "match": "all",
        "install": {
            "kind": "guide",
            "extra": None,
            "packages": [],
            "url": "https://pytorch.org/get-started/locally/"
        },
        "note": (
            "NVIDIA のドライバを入れ, 公式の手順で CUDA 付きの PyTorch を"
            "入れ直す. Mac の GPU は CUDA ではなく mps (experiment.device) を使う"
        )
    }
}

# 確かめ方の種類
CHECK_KINDS = {"module", "command", "function"}

# 入れ方の種類
INSTALL_KINDS = {"pip", "guide"}


def module_found(name: str) -> bool:
    """
    モジュールを import せずに, 入っているかを確かめる関数

    Parameters
    ----------
    name: str
        モジュールの名前 (a.b のような名前も可)

    Returns
    ----------
    found: bool
        見つかった場合は True
    """
    # 親のパッケージが無い場合の例外も, 見つからないものとして判定
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        found = False

    return found


def run_check(check: dict) -> tuple[bool, Optional[str]]:
    """
    1 つの確かめ方を実行する関数

    Parameters
    ----------
    check: dict
        kind と name (function の場合は function も) を持つ辞書

    Returns
    ----------
    ok: bool
        満たしている場合は True
    error: Optional[str]
        確かめる途中のエラー, 無ければ None
    """
    # 種類ごとの確認, 関数の例外は満たしていないものとして記録
    error = None
    if check["kind"] == "module":
        ok = module_found(check["name"])
    elif check["kind"] == "command":
        ok = shutil.which(check["name"]) is not None
    else:
        try:
            ok = bool(check["function"]())
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"

    return ok, error


def install_command(install: dict) -> Optional[str]:
    """
    pip で入る場合のコマンドを作る関数

    Parameters
    ----------
    install: dict
        入れ方の辞書

    Returns
    ----------
    command: Optional[str]
        pip のコマンド, pip で入らない場合は None
    """
    # 版の指定を含むので, パッケージごとに引用符で囲む
    if install["kind"] != "pip":
        command = None
        return command
    packages = " ".join(f'"{package}"' for package in install["packages"])
    command = f"pip install {packages}"

    return command


def extra_command(install: dict) -> Optional[str]:
    """
    ML-Found の追加の依存として入れる場合のコマンドを作る関数

    Parameters
    ----------
    install: dict
        入れ方の辞書

    Returns
    ----------
    command: Optional[str]
        submodule の ML-Found に追加の依存を付けて入れるコマンド,
        追加の依存が無い場合は None
    """
    # README の導入と同じく, プロジェクトの直下の submodule を想定
    if not install.get("extra"):
        command = None
        return command
    command = f'pip install -e "./ML-Found[{install["extra"]}]"'

    return command


def format_missing(
    feature: str, missing: list[str], install: dict,
    note: Optional[str] = None
) -> str:
    """
    足りないものと次の一手を伝えるメッセージを作る関数

    Parameters
    ----------
    feature: str
        機能の名前
    missing: list[str]
        足りないもの (確かめ方の名前)
    install: dict
        入れ方の辞書
    note: Optional[str] = None
        補足, None の場合は書かない

    Returns
    ----------
    message: str
        エラーのメッセージ
    """
    # 足りないものと, pip のコマンドか公式の手順の案内
    lines = [f"機能 {feature} に要るものが入っていません: {', '.join(missing)}"]
    if install["kind"] == "pip":
        lines.append(f"次の一手: {install_command(install)}")
        extra = extra_command(install)
        if extra is not None:
            lines.append(
                f"(ML-Found の追加の依存 {install['extra']} でも入る: {extra})"
            )
        if install.get("url"):
            lines.append(f"説明: {install['url']}")
    else:
        lines.append(
            f"次の一手: pip では入らないので, 公式の手順 {install['url']} に"
            "従って入れる"
        )
    if note:
        lines.append(f"補足: {note}")
    message = "\n".join(lines)

    return message


def describe_one(name: str, spec: dict, check: bool) -> dict[str, Any]:
    """
    1 つの機能の要るもの, 確かめた結果, 入れ方を JSON にできる形にする関数

    Parameters
    ----------
    name: str
        機能の名前
    spec: dict
        登録簿の定義
    check: bool
        入っているかを確かめる場合は True

    Returns
    ----------
    entry: dict[str, Any]
        機能の説明 (関数そのものは含めない)
    """
    # 確かめ方ごとの結果 (確かめない場合は null)
    checks = []
    for item in spec["checks"]:
        ok, error = run_check(item) if check else (None, None)
        checks.append(
            {"kind": item["kind"], "name": item["name"], "ok": ok,
             "error": error}
        )

    # all か any で全体の判定
    if not check:
        available = None
    elif spec["match"] == "any":
        available = any(item["ok"] for item in checks)
    else:
        available = all(item["ok"] for item in checks)

    # 入れ方にコマンドを添えて集約
    install = copy.deepcopy(spec["install"])
    install["command"] = install_command(install)
    install["extra_command"] = extra_command(install)
    entry = {
        "name": name,
        "description": spec["description"],
        "checks": checks,
        "match": spec["match"],
        "available": available,
        "install": install,
        "note": spec["note"]
    }

    return entry


def describe_requirements(
    names: Optional[list[str]] = None, check: bool = True
) -> list[dict[str, Any]]:
    """
    機能ごとの要るもの, 入っているか, 入れ方の一覧を返す関数

    画面 (ML-Dashboard) のカードや, 問題の設定の受け渡し (ML-Quest) で,
    足りないものを早めに知らせるために使う. 戻り値は JSON にできる

    Parameters
    ----------
    names: Optional[list[str]] = None
        機能の名前, None の場合は全て
    check: bool = True
        入っているかを確かめる場合は True. False の場合は checks の ok と
        available が None

    Returns
    ----------
    entries: list[dict[str, Any]]
        機能ごとの説明 (name, description, checks, match, available,
        install, note)
    """
    # 名前の確認と, 登録の順の説明
    selected = list(REQUIREMENTS) if names is None else list(names)
    for name in selected:
        get_spec(name)
    entries = [
        describe_one(name, REQUIREMENTS[name], check) for name in selected
    ]

    return entries


def get_spec(name: str) -> dict:
    """
    登録簿から機能の定義を取り出す関数

    Parameters
    ----------
    name: str
        機能の名前

    Returns
    ----------
    spec: dict
        機能の定義
    """
    # 知らない名前は登録済みの名前を添えてエラー
    if name not in REQUIREMENTS:
        raise ValueError(
            f"機能 {name} は登録されていません. 登録済み: "
            f"{', '.join(sorted(REQUIREMENTS))}"
        )
    spec = REQUIREMENTS[name]

    return spec


def require(feature: str) -> None:
    """
    機能に要るものが入っていることを確かめ, 無ければ入れ方を示して止める関数

    Parameters
    ----------
    feature: str
        機能の名前 (rl, mujoco, ros2, cuda など)

    Returns
    ----------
    None
    """
    # 確かめて, 足りなければ入れ方を添えたエラー
    entry = describe_one(feature, get_spec(feature), check=True)
    if entry["available"]:
        return
    missing = [item["name"] for item in entry["checks"] if not item["ok"]]
    raise MissingRequirementError(
        feature, missing, entry["install"], entry["note"]
    )


def check_definition(
    checks: list[dict], match: str, install: dict
) -> None:
    """
    登録する機能の確かめ方と入れ方の書き方を確かめる関数

    Parameters
    ----------
    checks: list[dict]
        確かめ方のリスト
    match: str
        all か any
    install: dict
        入れ方の辞書

    Returns
    ----------
    None
    """
    # 確かめ方は 1 つ以上で, 種類ごとに要る項目を持つこと
    if not checks:
        raise ValueError("checks に確かめ方を 1 つ以上書いてください")
    for item in checks:
        if item.get("kind") not in CHECK_KINDS or not item.get("name"):
            raise ValueError(
                f"確かめ方は kind ({sorted(CHECK_KINDS)}) と name を持つ辞書に"
                f"してください: {item!r}"
            )
        if item["kind"] == "function" and not callable(item.get("function")):
            raise ValueError(
                f"function の確かめ方には呼べる function が要ります: {item!r}"
            )
    if match not in ("all", "any"):
        raise ValueError(f"match は all か any にしてください: {match!r}")

    # pip はパッケージを, 公式の手順は URL を持つこと
    if install.get("kind") not in INSTALL_KINDS:
        raise ValueError(
            f"install の kind は {sorted(INSTALL_KINDS)} のどれかにしてください"
        )
    if install["kind"] == "pip" and not install.get("packages"):
        raise ValueError("pip で入れる場合は install に packages を書いてください")
    if install["kind"] == "guide" and not install.get("url"):
        raise ValueError("公式の手順に案内する場合は install に url を書いてください")


def register_requirement(
    name: str, description: str, checks: list[dict], install: dict,
    match: str = "all", note: Optional[str] = None
) -> None:
    """
    プロジェクトの機能の要るものと入れ方を登録簿に加える関数

    Parameters
    ----------
    name: str
        機能の名前
    description: str
        登録簿に出す説明
    checks: list[dict]
        確かめ方 (kind が module, command, function の辞書. function の場合は
        引数の無い関数を function に入れる)
    install: dict
        入れ方 (kind が pip なら packages, guide なら url. extra は ML-Found
        の追加の依存の名前で, 無ければ None)
    match: str = "all"
        all なら全ての確かめ方, any ならどれか 1 つを満たせば入っている
    note: Optional[str] = None
        補足

    Returns
    ----------
    None
    """
    # ML-Found が持つ機能の名前は上書きさせず, 書き方を確かめて登録
    if name in BUILTIN_NAMES:
        raise ValueError(f"機能 {name} は ML-Found が持つので上書きできません")
    check_definition(checks, match, install)
    full_install = {"extra": None, "packages": [], "url": None}
    full_install.update(install)
    REQUIREMENTS[name] = {
        "description": description,
        "checks": [dict(item) for item in checks],
        "match": match,
        "install": full_install,
        "note": note
    }


def unregister_requirement(name: str) -> None:
    """
    register_requirement で登録した機能を登録簿から除く関数

    Parameters
    ----------
    name: str
        機能の名前

    Returns
    ----------
    None
    """
    # ML-Found が持つ機能はエラー
    if name in BUILTIN_NAMES:
        raise ValueError(f"機能 {name} は ML-Found が持つので除けません")
    REQUIREMENTS.pop(name, None)


# ML-Found が持つ機能の名前
BUILTIN_NAMES = frozenset(REQUIREMENTS)

