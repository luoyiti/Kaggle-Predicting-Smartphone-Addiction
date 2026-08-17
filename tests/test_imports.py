def test_package_imports():
    import s6e8
    import s6e8.data
    import s6e8.features
    import s6e8.models.train
    import s6e8.runtime

    assert s6e8.__version__
    assert callable(s6e8.data.load_config)
    assert callable(s6e8.runtime.get_accelerator)
