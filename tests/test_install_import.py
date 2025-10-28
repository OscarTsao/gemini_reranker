def test_install_import() -> None:
    import criteriabind  # noqa: F401


def test_cli_entrypoints_import() -> None:
    from criteriabind.cli import candidate_gen, infer, judge  # noqa: F401

    assert callable(candidate_gen.main.__wrapped__)
    assert callable(judge.main.__wrapped__)
    assert callable(infer.main.__wrapped__)
