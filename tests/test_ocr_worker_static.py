"""
Статическая проверка workers/ocr_gpu_worker.py без Paddle и без GPU.

Зачем нужен: остальные тесты подменяют воркеры заглушками и никогда не
выполняют реальный файл воркера. Поэтому они не поймали ошибку, когда код
переключателя режима оказался внутри строки документации: переменная
ENABLE_ENHANCED не создавалась, и каждый трудный кадр падал с NameError.

Запуск:
    python3 tests/test_ocr_worker_static.py
"""

import ast
import os

WORKER = os.path.join(os.path.dirname(__file__), "..", "workers", "ocr_gpu_worker.py")


def module_level_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def test_mode_switch_is_real_code_not_docstring_text():
    src = open(WORKER, encoding="utf-8").read()
    tree = ast.parse(src)
    names = module_level_names(tree)
    for required in ("OCR_VARIANT_MODE", "ENABLE_ENHANCED", "OCR_EARLY_EXIT_CONF"):
        assert required in names, (
            f"{required} не является присваиванием на уровне модуля. "
            f"Скорее всего код попал внутрь строки документации."
        )
    print("[OK] переключатель режима это настоящий код, а не текст в документации")


def test_every_used_global_is_defined():
    """Каждое имя, которое читается внутри run_ocr, должно быть определено."""
    src = open(WORKER, encoding="utf-8").read()
    tree = ast.parse(src)
    defined = module_level_names(tree) | {
        n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    for n in ast.walk(tree):
        # импорты верхнего уровня и внутри try/if на уровне модуля
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                defined.add((a.asname or a.name).split(".")[0])

    run_ocr = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_ocr")
    local = {a.arg for a in run_ocr.args.args}
    for node in ast.walk(run_ocr):
        if isinstance(node, ast.FunctionDef):
            local.add(node.name)
            local |= {a.arg for a in node.args.args}
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local.add(node.id)
        elif isinstance(node, ast.Nonlocal):
            local |= set(node.names)

    import builtins
    missing = sorted({
        node.id for node in ast.walk(run_ocr)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        and node.id not in defined and node.id not in local
        and not hasattr(builtins, node.id)
    })
    assert not missing, f"run_ocr использует неопределённые имена: {missing}"
    print("[OK] все имена, которые использует run_ocr, определены")


def test_mode_parsing_for_each_mode():
    src = open(WORKER, encoding="utf-8").read()
    start = src.index("OCR_VARIANT_MODE = sys.argv")
    end = src.index('print(f"OCR VARIANT MODE')
    code = src[start:end]

    for argv, expect in [(["w", "h", "1", "k"], True),
                         (["w", "h", "1", "k", "full"], True),
                         (["w", "h", "1", "k", "no-enhanced"], False)]:
        ns = {"sys": type("S", (), {"argv": argv})}
        exec(code, ns)
        assert ns["ENABLE_ENHANCED"] is expect, (argv, ns["ENABLE_ENHANCED"])

    ns = {"sys": type("S", (), {"argv": ["w", "h", "1", "k", "no-enhancd"]})}
    try:
        exec(code, ns)
        raise AssertionError("опечатка в режиме не была поймана")
    except SystemExit:
        pass
    print("[OK] режимы full и no-enhanced разбираются верно, опечатка ловится сразу")


if __name__ == "__main__":
    test_mode_switch_is_real_code_not_docstring_text()
    test_every_used_global_is_defined()
    test_mode_parsing_for_each_mode()
    print("\nAll tests passed.")
