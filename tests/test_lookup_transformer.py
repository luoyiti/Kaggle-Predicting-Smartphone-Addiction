import json

import numpy as np
import pandas as pd
import pytest

from s6e8.models.lookup_transformer import LookupPreprocessor


def test_fold_training_api_is_available_without_importing_torch():
    from s6e8.models.lookup_transformer import train_lookup_fold

    assert callable(train_lookup_fold)


def test_lookup_preprocessor_is_target_free_and_deterministic():
    train = pd.DataFrame(
        {
            "age": [20.0, 21.0, np.nan],
            "daily_screen_time_hours": [3.10, 3.20, 3.10],
            "screen_remainder_complete": [1.0, 1.2, np.nan],
        }
    )
    test = pd.DataFrame(
        {
            "age": [22.0],
            "daily_screen_time_hours": [3.30],
            "screen_remainder_complete": [1.1],
        }
    )
    pre = LookupPreprocessor(
        lookup_columns=["age", "daily_screen_time_hours"],
        numeric_columns=[
            "age",
            "daily_screen_time_hours",
            "screen_remainder_complete",
        ],
        decimal_places={"age": 0, "daily_screen_time_hours": 2},
    ).fit(train, test)

    first = pre.transform(train)
    second = pre.transform(train.copy())

    assert np.array_equal(first.lookup_ids, second.lookup_ids)
    assert np.allclose(first.numeric_values, second.numeric_values, equal_nan=False)
    assert first.lookup_ids.shape == (3, 2)
    assert first.numeric_values.shape == (3, 3)
    assert first.missing_mask.shape == (3, 3)
    assert pre.provenance()["transductive_predictor_preprocessing"] is True


def test_lookup_preprocessor_reserves_missing_and_oov_ids():
    train = pd.DataFrame({"grid_value": [10.0, np.nan], "numeric": [1.0, 2.0]})
    test = pd.DataFrame({"grid_value": [20.0], "numeric": [3.0]})
    pre = LookupPreprocessor(
        lookup_columns=["grid_value"],
        numeric_columns=["grid_value", "numeric"],
        decimal_places={"grid_value": 0},
    ).fit(train, test)

    transformed = pre.transform(
        pd.DataFrame(
            {
                "grid_value": [30.0, np.nan, 20.0],
                "numeric": [4.0, np.nan, 2.0],
            }
        )
    )

    assert transformed.lookup_ids[:, 0].tolist() == [1, 0, 3]
    assert transformed.numeric_values[:, 1].tolist() == [2.0, 0.0, 0.0]
    assert transformed.missing_mask[:, 1].tolist() == [False, True, False]
    assert pre.lookup_cardinalities == [4]


def test_lookup_preprocessor_handles_read_only_pandas_integer_arrays(monkeypatch):
    """Regression: pandas may return a read-only mapped integer array."""
    train = pd.DataFrame({"grid_value": [10.0, np.nan]})
    test = pd.DataFrame({"grid_value": [20.0]})
    pre = LookupPreprocessor(
        lookup_columns=["grid_value"],
        numeric_columns=["grid_value"],
        decimal_places={"grid_value": 0},
    ).fit(train, test)
    original_to_numpy = pd.Series.to_numpy

    def readonly_when_copy_not_requested(series, *args, **kwargs):
        array = original_to_numpy(series, *args, **kwargs)
        dtype = kwargs.get("dtype", args[0] if args else None)
        if dtype is not None and np.dtype(dtype) == np.dtype(np.int64):
            if not kwargs.get("copy", False):
                array.setflags(write=False)
        return array

    monkeypatch.setattr(pd.Series, "to_numpy", readonly_when_copy_not_requested)

    transformed = pre.transform(
        pd.DataFrame({"grid_value": [10.0, np.nan, 30.0]})
    )

    assert transformed.lookup_ids[:, 0].tolist() == [2, 0, 1]
    assert transformed.lookup_ids.dtype == np.int64
    assert transformed.lookup_ids.flags.owndata
    assert transformed.lookup_ids.flags.writeable


def test_lookup_preprocessor_provenance_is_serializable_and_target_free():
    train = pd.DataFrame(
        {
            "grid_value": [1.0, 2.0],
            "numeric": [1.0, 3.0],
            "addicted_label": [0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "grid_value": [3.0],
            "numeric": [5.0],
            "addicted_label": [1],
        }
    )
    pre = LookupPreprocessor(
        lookup_columns=["grid_value"],
        numeric_columns=["grid_value", "numeric"],
        decimal_places={"grid_value": 0},
    ).fit(train, test)

    provenance = pre.provenance()

    assert json.loads(json.dumps(provenance, sort_keys=True)) == provenance
    assert provenance == {
        "lookup_columns": ["grid_value"],
        "numeric_columns": ["grid_value", "numeric"],
        "decimal_places": {"grid_value": 0},
        "lookup_cardinalities": [5],
        "numeric_medians": {"grid_value": 2.0, "numeric": 3.0},
        "numeric_scales": {"grid_value": 1.0, "numeric": 2.0},
        "transductive_predictor_preprocessing": True,
    }
    assert all(
        "target" not in key.lower() and "label" not in key.lower()
        for key in provenance
    )


def test_empty_column_preprocessor_still_requires_fit():
    pre = LookupPreprocessor(
        lookup_columns=[],
        numeric_columns=[],
        decimal_places={},
    )
    frame = pd.DataFrame(index=[0, 1])

    with pytest.raises(RuntimeError, match="fitted before transform"):
        pre.transform(frame)
    with pytest.raises(RuntimeError, match="fitted before provenance"):
        pre.provenance()

    pre.fit(frame, frame.iloc[:0])
    transformed = pre.transform(frame)

    assert transformed.lookup_ids.shape == (2, 0)
    assert transformed.numeric_values.shape == (2, 0)
    assert transformed.missing_mask.shape == (2, 0)
    assert pre.provenance()["lookup_cardinalities"] == []


@pytest.mark.parametrize(
    ("lookup_columns", "numeric_columns"),
    [
        (["age", "screen"], ["screen", "age", "budget"]),
        (["age", "screen"], ["age", "budget"]),
        (["age", "screen"], ["age"]),
    ],
)
def test_lookup_preprocessor_requires_lookup_columns_as_numeric_prefix(
    lookup_columns, numeric_columns
):
    with pytest.raises(
        ValueError,
        match="numeric_columns.*start.*lookup_columns",
    ):
        LookupPreprocessor(
            lookup_columns=lookup_columns,
            numeric_columns=numeric_columns,
            decimal_places={},
        )


def test_periodic_numeric_embedding_returns_one_token_per_feature():
    torch = pytest.importorskip("torch")
    from s6e8.models.lookup_transformer import PeriodicNumericEmbedding

    embedding = PeriodicNumericEmbedding(
        n_features=3,
        n_frequencies=4,
        d_model=8,
    )
    tokens = embedding(torch.tensor([[0.1, 0.2, 0.3], [-1.0, 0.0, 1.0]]))

    assert tokens.shape == (2, 3, 8)
    assert torch.isfinite(tokens).all()


def test_lookup_transformer_returns_one_finite_logit_per_row():
    torch = pytest.importorskip("torch")
    from s6e8.models.lookup_transformer import LookupTransformer

    model = LookupTransformer(
        lookup_cardinalities=[8, 10],
        n_numeric=3,
        d_model=32,
        plr_frequencies=8,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        mask_probability=0.2,
    )
    model.train()
    logits = model(
        lookup_ids=torch.tensor([[2, 3], [4, 0], [1, 5]], dtype=torch.long),
        numeric_values=torch.tensor(
            [[0.1, 0.2, 0.3], [0.0, -0.2, 0.4], [1.0, 0.5, -0.1]]
        ),
        missing_mask=torch.tensor(
            [
                [False, False, False],
                [True, False, False],
                [False, False, False],
            ]
        ),
    )

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_lookup_transformer_exact_ids_affect_corresponding_numeric_tokens():
    torch = pytest.importorskip("torch")
    from s6e8.models.lookup_transformer import LookupTransformer

    torch.manual_seed(7)
    model = LookupTransformer(
        lookup_cardinalities=[8, 10],
        n_numeric=3,
        d_model=16,
        plr_frequencies=4,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        mask_probability=0.0,
    ).eval()
    logits = model(
        lookup_ids=torch.tensor([[2, 3], [4, 3]], dtype=torch.long),
        numeric_values=torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
        missing_mask=torch.zeros((2, 3), dtype=torch.bool),
    )

    assert not torch.isclose(logits[0], logits[1])


def test_lookup_transformer_missing_token_replaces_raw_feature_branches():
    torch = pytest.importorskip("torch")
    from s6e8.models.lookup_transformer import LookupTransformer

    torch.manual_seed(11)
    model = LookupTransformer(
        lookup_cardinalities=[8, 10],
        n_numeric=3,
        d_model=16,
        plr_frequencies=4,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        mask_probability=0.0,
    ).eval()
    logits = model(
        lookup_ids=torch.tensor([[2, 3], [7, 3]], dtype=torch.long),
        numeric_values=torch.tensor([[0.1, 0.2, 0.3], [9.9, 0.2, 0.3]]),
        missing_mask=torch.tensor(
            [[True, False, False], [True, False, False]], dtype=torch.bool
        ),
    )

    assert torch.allclose(logits[0], logits[1], atol=1e-6, rtol=0.0)


def test_lookup_transformer_masks_feature_tokens_only_during_training():
    torch = pytest.importorskip("torch")
    from s6e8.models.lookup_transformer import LookupTransformer

    torch.manual_seed(13)
    model = LookupTransformer(
        lookup_cardinalities=[8, 10],
        n_numeric=3,
        d_model=16,
        plr_frequencies=4,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        mask_probability=1.0,
    )
    lookup_ids = torch.tensor([[2, 3], [7, 9]], dtype=torch.long)
    numeric_values = torch.tensor([[0.1, 0.2, 0.3], [9.9, -4.0, 8.0]])
    missing_mask = torch.zeros((2, 3), dtype=torch.bool)

    model.train()
    masked_logits = model(lookup_ids, numeric_values, missing_mask)
    model.eval()
    eval_logits_first = model(lookup_ids, numeric_values, missing_mask)
    eval_logits_second = model(lookup_ids, numeric_values, missing_mask)

    assert torch.allclose(masked_logits[0], masked_logits[1], atol=1e-6, rtol=0.0)
    assert torch.equal(eval_logits_first, eval_logits_second)
    assert not torch.isclose(eval_logits_first[0], eval_logits_first[1])


def test_lookup_transformer_rejects_more_lookup_than_numeric_columns():
    pytest.importorskip("torch")
    from s6e8.models.lookup_transformer import LookupTransformer

    with pytest.raises(ValueError, match="n_numeric.*lookup"):
        LookupTransformer(
            lookup_cardinalities=[8, 10],
            n_numeric=1,
            d_model=16,
            plr_frequencies=4,
            n_layers=1,
            n_heads=4,
            dropout=0.0,
            mask_probability=0.0,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lookup_cardinalities": [1]}, "cardinalit"),
        ({"lookup_cardinalities": [2.5]}, "cardinalit"),
        ({"lookup_cardinalities": [True]}, "cardinalit"),
        ({"n_numeric": 0}, "n_numeric"),
        ({"n_numeric": 3.5}, "n_numeric"),
        ({"d_model": 0}, "d_model"),
        ({"d_model": 16.5}, "d_model"),
        ({"plr_frequencies": 0}, "plr_frequencies"),
        ({"plr_frequencies": 4.5}, "plr_frequencies"),
        ({"n_layers": 0}, "n_layers"),
        ({"n_layers": 1.5}, "n_layers"),
        ({"n_heads": 0}, "n_heads"),
        ({"n_heads": 2.5}, "n_heads"),
        ({"d_model": 10, "n_heads": 4}, "divisible"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.1}, "dropout"),
        ({"mask_probability": -0.1}, "mask_probability"),
        ({"mask_probability": 1.1}, "mask_probability"),
    ],
)
def test_lookup_transformer_rejects_invalid_constructor_parameters(
    overrides, message
):
    pytest.importorskip("torch")
    from s6e8.models.lookup_transformer import LookupTransformer

    parameters = {
        "lookup_cardinalities": [8, 10],
        "n_numeric": 3,
        "d_model": 16,
        "plr_frequencies": 4,
        "n_layers": 1,
        "n_heads": 4,
        "dropout": 0.1,
        "mask_probability": 0.2,
    }
    parameters.update(overrides)

    with pytest.raises(ValueError, match=message):
        LookupTransformer(**parameters)


def test_train_lookup_fold_one_epoch_cpu_smoke():
    torch = pytest.importorskip("torch")
    from s6e8.models.lookup_transformer import train_lookup_fold

    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "age": np.tile(np.arange(20.0, 28.0), 8),
            "daily_screen_time_hours": np.round(
                np.tile(np.linspace(2.0, 7.5, 16), 4), 2
            ),
            "screen_remainder_complete": rng.normal(1.5, 0.3, size=64),
        }
    )
    test = pd.DataFrame(
        {
            "age": np.arange(20.0, 28.0),
            "daily_screen_time_hours": np.round(
                np.linspace(2.25, 7.25, 8), 2
            ),
            "screen_remainder_complete": rng.normal(1.5, 0.3, size=8),
        }
    )
    labels = (
        frame["daily_screen_time_hours"].to_numpy()
        + 0.15 * rng.normal(size=len(frame))
        > 4.75
    ).astype(np.float32)
    pre = LookupPreprocessor(
        lookup_columns=["age", "daily_screen_time_hours"],
        numeric_columns=[
            "age",
            "daily_screen_time_hours",
            "screen_remainder_complete",
        ],
        decimal_places={"age": 0, "daily_screen_time_hours": 2},
    ).fit(frame, test)

    train_arrays = pre.transform(frame.iloc[:48])
    valid_arrays = pre.transform(frame.iloc[48:])
    test_arrays = pre.transform(test)
    va_pred, te_pred, best_epoch, diagnostics = train_lookup_fold(
        train_arrays=train_arrays,
        train_y=labels[:48],
        valid_arrays=valid_arrays,
        valid_y=labels[48:],
        test_arrays=test_arrays,
        lookup_cardinalities=pre.lookup_cardinalities,
        params={
            "d_model": 16,
            "plr_frequencies": 4,
            "n_layers": 1,
            "n_heads": 2,
            "dropout": 0.0,
            "mask_probability": 0.0,
            "batch_size": 16,
            "epochs": 1,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "embedding_weight_decay": 1.0e-3,
            "ema_decay": 0.99,
            "patience": 1,
            "max_grad_norm": 1.0,
            "num_workers": 0,
        },
        seed=42,
        device="cpu",
    )

    assert va_pred.shape == (len(labels[48:]),)
    assert te_pred.shape == (len(test),)
    assert np.isfinite(va_pred).all()
    assert np.isfinite(te_pred).all()
    assert np.logical_and(va_pred >= 0.0, va_pred <= 1.0).all()
    assert best_epoch == 1
    assert diagnostics["best_auc"] >= 0.0
    assert diagnostics["epochs_trained"] == 1
    assert json.loads(json.dumps(diagnostics)) == diagnostics


def test_train_lookup_fold_single_class_validation_is_stable():
    from s6e8.models.lookup_transformer import _validation_auc

    assert _validation_auc(np.zeros(4), np.linspace(0.1, 0.9, 4)) == 0.5
