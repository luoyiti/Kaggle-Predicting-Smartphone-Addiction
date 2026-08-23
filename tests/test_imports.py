def test_package_imports():
    import s6e8
    import s6e8.calibration
    import s6e8.contracts
    import s6e8.data
    import s6e8.data_audit
    import s6e8.eda
    import s6e8.eda_report
    import s6e8.ensemble
    import s6e8.error_analysis
    import s6e8.features
    import s6e8.metrics
    import s6e8.models.train
    import s6e8.oof_io
    import s6e8.promotion
    import s6e8.runtime
    import s6e8.target_encoding

    assert s6e8.__version__
    assert callable(s6e8.data.load_config)
    assert callable(s6e8.runtime.get_accelerator)
    assert callable(s6e8.eda.run_eda)
    assert callable(s6e8.target_encoding.parse_exact_te_config)
    assert callable(s6e8.promotion.evaluate_promotion)
    assert "lightgbm" in s6e8.models.train.BACKEND_ALIASES
    assert "extratrees" in s6e8.models.train.BACKEND_ALIASES
