import copy
import importlib.metadata
import importlib.util
import re
import shutil
from typing import Any, Optional

import torch


class MissingRequirementError(ImportError):
    def __init__(
        self, feature: str, missing: list[str], install: dict,
        note: Optional[str] = None, outdated: Optional[list[str]] = None
    ):
        # 親クラスのコンストラクタを呼び出す
        super(MissingRequirementError, self).__init__(
            format_missing(feature, missing, install, note, outdated)
        )

        # 機能の名前, 足りないもの, 入れ方, 補足, 版が足りないものを保存
        self.feature = feature
        self.missing = missing
        self.install = install
        self.note = note
        self.outdated = list(outdated or [])

    def __reduce__(self) -> tuple:
        """
        pickle で送れるように, 作り直すための引数を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        state: tuple
            クラスと, コンストラクタの引数
        """
        # 別のプロセス (DataLoader の worker など) で作り直すための引数
        state = (
            type(self),
            (self.feature, self.missing, self.install, self.note,
             self.outdated)
        )

        return state


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


# 「機械学習を全て GUI 上で行う」ためのライブラリを置く場所
GITHUB = "https://github.com/victor-von-pooh"

# ML-Found 自身を pip で入れるときの指定
FOUND_PACKAGE = f"ml-found @ git+{GITHUB}/ML-Found.git"

# 機能ごとの要るもの, 確かめ方, 入れ方の登録簿
# checks の kind は module (import できるか), command (コマンドがあるか),
# function (関数が True を返すか). match は all (全て要る) か any (どれか 1 つ)
# module の min_version は版の下限 (pyproject.toml の任意の依存の下限と同じ),
# distribution は版を読むパッケージの名前 (無ければ name と同じ)
REQUIREMENTS: dict[str, dict[str, Any]] = {
    "rl": {
        "description": "強化学習の環境 (Gymnasium の形の環境を作る)",
        "checks": [
            {"kind": "module", "name": "gymnasium", "min_version": "1.0"}
        ],
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
            {"kind": "module", "name": "gymnasium", "min_version": "1.0"},
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
    },
    "eda": {
        "description": "データの分析 (ML-EDA. データセットの読み込みと前処理)",
        # 書き出したデータセットは parquet なので, 読む pyarrow の確かめも
        "checks": [
            {"kind": "module", "name": "ml_eda"},
            {"kind": "module", "name": "pyarrow"}
        ],
        "match": "all",
        "install": {
            "kind": "pip",
            "extra": None,
            "packages": [
                f"ml-eda[parquet] @ git+{GITHUB}/ML-EDA.git"
            ],
            "url": f"{GITHUB}/ML-EDA"
        },
        "note": (
            "ML-EDA (ML-Fathom) が書き出すデータセットは parquet なので, "
            "読むための pyarrow (追加の依存 parquet) も入れる"
        )
    },
    "modelcontrol": {
        "description": "モデルの組み立て (ML-ModelControl. モデルの定義から作る)",
        "checks": [{"kind": "module", "name": "ml_modelcontrol"}],
        "match": "all",
        "install": {
            "kind": "pip",
            "extra": None,
            "packages": [
                f"ml-modelcontrol @ git+{GITHUB}/ML-ModelControl.git"
            ],
            "url": f"{GITHUB}/ML-ModelControl"
        },
        "note": None
    },
    "inference": {
        "description": "推論 (ML-Inference. 学習したモデルで予測し, 判断に使う)",
        "checks": [
            {"kind": "module", "name": "ml_inference"},
            {"kind": "module", "name": "pyarrow"}
        ],
        "match": "all",
        "install": {
            "kind": "pip",
            "extra": None,
            "packages": [
                f"ml-inference @ git+{GITHUB}/ML-Inference.git"
            ],
            "url": f"{GITHUB}/ML-Inference"
        },
        "note": (
            "ML-Inference は ML-EDA (parquet を読む pyarrow を含む) と "
            "ML-ModelControl も一緒に入れる"
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

    importlib.util.find_spec で探す (版は module_version で別に確かめる).
    a.b のような名前では, 親のパッケージ a は import される

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


def module_version(distribution: str) -> Optional[str]:
    """
    パッケージを import せずに, 入っている版の文字を読む関数

    Parameters
    ----------
    distribution: str
        パッケージの名前 (pip で入れるときの名前)

    Returns
    ----------
    version: Optional[str]
        版の文字, 読めない場合は None
    """
    # パッケージの情報 (dist-info) が無い場合は読めないものとして判定
    try:
        version = importlib.metadata.version(distribution)
    except (importlib.metadata.PackageNotFoundError, ValueError):
        version = None

    return version


# 版の文字の形 (PEP 440 の公開の版と, + の後のローカルの版)
VERSION_PATTERN = re.compile(
    r"^v?(?:\d+!)?(?P<release>\d+(?:\.\d+)*)"
    r"(?:[-_.]?(?P<pre>a|alpha|b|beta|c|rc|pre|preview)[-_.]?(?P<pre_n>\d*))?"
    r"(?:-(?P<post_implicit>\d+)"
    r"|[-_.]?(?:post|rev|r)[-_.]?(?P<post_n>\d*))?"
    r"(?:[-_.]?(?P<dev>dev)[-_.]?(?P<dev_n>\d*))?"
    r"(?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?$",
    re.IGNORECASE
)

# pip で入れるときのパッケージの名前の形 (PEP 508)
DISTRIBUTION_PATTERN = re.compile(
    r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$", re.IGNORECASE
)

# 先行版の印の順 (alpha, beta, release candidate)
PRE_ORDER = {
    "a": 0, "alpha": 0, "b": 1, "beta": 1, "c": 2, "rc": 2, "pre": 2,
    "preview": 2
}


def version_key(text: str) -> Optional[tuple]:
    """
    版の文字を, 大小を比べられる組に直す関数

    PEP 440 の順 (1.0.dev1 < 1.0a1 < 1.0rc1 < 1.0 < 1.0.post1) に並べる.
    1.0 と 1.0.0 は同じ. + の後のローカルの版 (2.11.0+cpu の cpu) と
    エポック (1!) は比べない. そのため 1!0.1 は 0.1 として比べ, 1.0 以上
    とは見ない (PEP 440 の順より古く見る側に倒れ, 入れ方を案内して止まる).
    packaging に依存しないための, 下限の比較に足りるだけの実装

    Parameters
    ----------
    text: str
        版の文字

    Returns
    ----------
    key: Optional[tuple]
        比べるための組, 読めない形の場合は None
    """
    # 形の確認
    match = VERSION_PATTERN.match(text.strip())
    if match is None:
        key = None
        return key

    # 末尾の 0 を除いた数の並び
    release = [int(part) for part in match["release"].split(".")]
    while len(release) > 1 and release[-1] == 0:
        release.pop()

    # 先行版, 後の版, 開発版の順位
    post = match["post_implicit"] or match["post_n"]
    has_post = match["post_implicit"] is not None or (
        match["post_n"] is not None
    )
    if match["pre"]:
        pre = (0, PRE_ORDER[match["pre"].lower()], int(match["pre_n"] or 0))
    elif match["dev"] and not has_post:
        pre = (-1, 0, 0)
    else:
        pre = (1, 0, 0)
    post_rank = int(post or 0) if has_post else -1
    dev = (0, int(match["dev_n"] or 0)) if match["dev"] else (1, 0)
    key = (tuple(release), pre, post_rank, dev)

    return key


def version_at_least(version: str, minimum: str) -> Optional[bool]:
    """
    入っている版が下限以上かを確かめる関数

    Parameters
    ----------
    version: str
        入っている版の文字
    minimum: str
        版の下限

    Returns
    ----------
    enough: Optional[bool]
        下限以上なら True, 古ければ False, 版の文字が読めない場合は None
    """
    # 末尾の 0 を除いた組どうしの比較
    found, needed = version_key(version), version_key(minimum)
    if found is None or needed is None:
        enough = None
        return enough
    enough = found >= needed

    return enough


def check_module(check: dict) -> tuple[bool, Optional[str], Optional[str]]:
    """
    モジュールがあるかと, 版の下限を満たすかを確かめる関数

    Parameters
    ----------
    check: dict
        name と, 任意の min_version と distribution を持つ辞書

    Returns
    ----------
    ok: bool
        あって版の下限を満たす場合は True
    error: Optional[str]
        版が足りない場合と版を確かめられない場合の説明, 無ければ None
    version: Optional[str]
        入っている版, 無いか読めない場合は None
    """
    # モジュールが無ければ版は読まない
    if not module_found(check["name"]):
        return False, None, None

    # 版の下限の確認, 版が分からない場合は使えるものとして説明だけ残す
    minimum = check.get("min_version")
    version = module_version(check.get("distribution") or check["name"])
    if minimum is None:
        return True, None, version
    if version is None:
        return True, f"版を読めないので {minimum} 以上かを確かめていない", None
    enough = version_at_least(version, minimum)
    if enough is None:
        error = f"版 {version} を読めないので {minimum} 以上かを確かめていない"
        return True, error, version
    if not enough:
        error = f"版 {version} は古い. {minimum} 以上が要る"
        return False, error, version

    return True, None, version


def run_check(check: dict) -> tuple[bool, Optional[str], Optional[str]]:
    """
    1 つの確かめ方を実行する関数

    Parameters
    ----------
    check: dict
        kind と name (function の場合は function, module の場合は任意の
        min_version と distribution も) を持つ辞書

    Returns
    ----------
    ok: bool
        満たしている場合は True
    error: Optional[str]
        確かめる途中のエラーか版が足りない説明, 無ければ None
    version: Optional[str]
        module の場合の入っている版, それ以外か読めない場合は None
    """
    # 種類ごとの確認, 関数の例外は満たしていないものとして記録
    error = None
    version = None
    if check["kind"] == "module":
        ok, error, version = check_module(check)
    elif check["kind"] == "command":
        ok = shutil.which(check["name"]) is not None
    else:
        try:
            ok = bool(check["function"]())
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"

    return ok, error, version


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
    note: Optional[str] = None, outdated: Optional[list[str]] = None
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
    outdated: Optional[list[str]] = None
        版が足りないもの (名前と版の説明), None の場合は無し

    Returns
    ----------
    message: str
        エラーのメッセージ
    """
    # 足りないものと版が足りないもの
    lines = []
    if missing:
        lines.append(
            f"機能 {feature} に要るものが入っていません: {', '.join(missing)}"
        )
    if outdated:
        lines.append(
            f"機能 {feature} に要るものの版が足りません: {', '.join(outdated)}"
        )

    # pip のコマンドか公式の手順の案内
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
        ok, error, version = run_check(item) if check else (None, None, None)
        checks.append(
            {"kind": item["kind"], "name": item["name"], "ok": ok,
             "error": error, "version": version,
             "min_version": item.get("min_version")}
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

    # 無いものと, あるが版が足りないものに分けて案内
    missing, outdated = [], []
    for item in entry["checks"]:
        if item["ok"]:
            continue
        if item["version"] is not None and item["min_version"] is not None:
            outdated.append(
                f"{item['name']} ({item['version']}. "
                f"{item['min_version']} 以上が要る)"
            )
        else:
            missing.append(item["name"])
    raise MissingRequirementError(
        feature, missing, entry["install"], entry["note"], outdated
    )


def environment_packages(
    names: list[str], include_found: bool = True
) -> list[str]:
    """
    プロジェクトの環境に pip で入れるものの指定の一覧を返す関数

    学習のプロジェクトの仮想環境を作るとき (画面の ML-Dashboard など) に,
    ML-Found 自身と, 要る機能のうち pip で入るものを 1 回の pip install で
    入れられる形にする. pip で入らない機能 (install の kind が guide) は
    含めないので, describe_requirements の入れ方で案内する. 同じ指定は
    文字が同じものだけを 1 回にまとめる. 登録簿に無い名前は ValueError

    Parameters
    ----------
    names: list[str]
        要る機能の名前 (config の requires など)
    include_found: bool = True
        ML-Found 自身 (git の main) を先頭に含める場合は True

    Returns
    ----------
    packages: list[str]
        pip に渡す指定 (書いた順. 同じ指定は 1 回だけ)
    """
    # 名前を先に全て確かめてから, 書いた順に pip の指定を集約
    specs = [get_spec(name) for name in names]
    packages = [FOUND_PACKAGE] if include_found else []
    for spec in specs:
        if spec["install"]["kind"] != "pip":
            continue
        for package in spec["install"]["packages"]:
            if package not in packages:
                packages.append(package)

    return packages


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
        check_version_fields(item)
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


def check_version_fields(item: dict) -> None:
    """
    確かめ方の版の下限の書き方を確かめる関数

    Parameters
    ----------
    item: dict
        1 つの確かめ方

    Returns
    ----------
    None
    """
    # 版の下限は module だけに書け, 読める版の文字とパッケージの名前であること
    fields = [key for key in ("min_version", "distribution") if key in item]
    if fields and item["kind"] != "module":
        raise ValueError(
            f"{', '.join(fields)} は module の確かめ方だけに書けます: {item!r}"
        )
    minimum = item.get("min_version")
    if minimum is not None and (
        not isinstance(minimum, str) or version_key(minimum) is None
    ):
        raise ValueError(
            f"min_version は 1.0 のような版の文字にしてください: {minimum!r}"
        )
    distribution = item.get("distribution")
    if distribution is not None and (
        not isinstance(distribution, str)
        or DISTRIBUTION_PATTERN.match(distribution) is None
    ):
        raise ValueError(
            "distribution は pip で入れるときのパッケージの名前 (版の指定や"
            f"追加の依存の [ ] を含まない) にしてください: {distribution!r}"
        )


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
        引数の無い関数を function に入れる. module の場合は版の下限
        min_version と, 版を読むパッケージの名前 distribution を書ける)
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
