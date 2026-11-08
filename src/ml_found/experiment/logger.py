import logging
from logging import FileHandler, Formatter, getLogger


def get_logger(
    log_file: str,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    name: str = "ml_found"
) -> logging.Logger:
    """
    ファイルに書き出すロガーを取得する関数

    Parameters
    ----------
    log_file: str
        ログファイルのパス
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ログの書式
    name: str = "ml_found"
        ロガーの名前

    Returns
    ----------
    logger: logging.Logger
        設定済みのロガー
    """
    # ロガーを取得してレベルを設定
    logger = getLogger(name)
    logger.setLevel(logging.INFO)

    # 同じプロセスで何度も呼ばれても二重に書き出さないように古い出力先の削除
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    # ルートロガーへ伝わって二重に出力されないように設定
    logger.propagate = False

    # ファイルへの出力先を設定
    handler = FileHandler(log_file, mode="w", encoding="utf-8")
    handler.setFormatter(Formatter(log_format))
    logger.addHandler(handler)

    return logger
