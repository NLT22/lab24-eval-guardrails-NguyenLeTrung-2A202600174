_SUMMARY = "0 passed"


def pytest_terminal_summary(terminalreporter):
    global _SUMMARY
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    if failed:
        _SUMMARY = f"{passed} passed, {failed} failed"
    else:
        _SUMMARY = f"{passed} passed"


def pytest_unconfigure(config):
    print(_SUMMARY)
