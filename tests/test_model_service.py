import sys
import types

import pytest

from app.model_service import ModelService
from app.settings import Settings


def make_settings(tmp_path=None, root=None, checkpoint=None):
    return Settings(
        environment="test",
        log_json=False,
        simpleclick_root=root or (str(tmp_path) if tmp_path else "/does/not/exist"),
        checkpoint_path=checkpoint
        or (str(tmp_path / "model.pth") if tmp_path else "/does/not/exist/model.pth"),
        model_device="cpu",
    )


def install_fakes(monkeypatch):
    """Inject fake `torch` and `isegm.*` modules so the model can be loaded."""

    safe_added = []

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.device = lambda name: name
    torch.add_safe_globals = lambda globals_list: safe_added.append(globals_list)
    torch.inference_mode = lambda: types.SimpleNamespace(
        __enter__=lambda s: None, __exit__=lambda *a: None
    )
    serialization = types.ModuleType("torch.serialization")
    serialization.add_safe_globals = torch.add_safe_globals
    torch.serialization = serialization
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.serialization", serialization)

    CrossEntropyLoss = type("CrossEntropyLoss", (), {})

    click = types.SimpleNamespace()
    clicker = lambda init_clicks=None: types.SimpleNamespace(init_clicks=init_clicks)
    predictor = types.SimpleNamespace(
        set_input_image=lambda image: None,
        get_prediction=lambda clicker: None,
    )
    model = types.SimpleNamespace()

    inference = types.ModuleType("isegm.inference")
    modeling = types.ModuleType("isegm.model.modeling")
    transformer_helper = types.ModuleType("isegm.model.modeling.transformer_helper")

    utils = types.ModuleType("isegm.inference.utils")
    utils.load_is_model = lambda checkpoint, device, **kwargs: model
    inference.utils = utils
    clicker_module = types.ModuleType("isegm.inference.clicker")
    clicker_module.Click = click
    clicker_module.Clicker = clicker
    inference.clicker = clicker_module
    predictors = types.ModuleType("isegm.inference.predictors")
    predictors.get_predictor = lambda *args, **kwargs: predictor
    inference.predictors = predictors
    cross_entropy = types.ModuleType(
        "isegm.model.modeling.transformer_helper.cross_entropy_loss"
    )
    cross_entropy.CrossEntropyLoss = CrossEntropyLoss
    transformer_helper.cross_entropy_loss = cross_entropy

    isegm = types.ModuleType("isegm")
    isegm.inference = inference
    isegm.model = types.ModuleType("isegm.model")
    isegm.model.modeling = modeling
    modeling.transformer_helper = transformer_helper

    monkeypatch.setitem(sys.modules, "isegm", isegm)
    monkeypatch.setitem(sys.modules, "isegm.inference", inference)
    monkeypatch.setitem(sys.modules, "isegm.inference.utils", utils)
    monkeypatch.setitem(sys.modules, "isegm.inference.clicker", clicker_module)
    monkeypatch.setitem(sys.modules, "isegm.inference.predictors", predictors)
    monkeypatch.setitem(
        sys.modules,
        "isegm.model.modeling.transformer_helper.cross_entropy_loss",
        cross_entropy,
    )

    return {"safe_added": safe_added}


def test_load_registers_cross_entropy_loss_as_safe_global(monkeypatch, tmp_path):
    (tmp_path / "model.pth").write_bytes(b"fake")
    fakes = install_fakes(monkeypatch)
    settings = make_settings(tmp_path)

    service = ModelService(settings)
    service.load()

    assert service.state == "ready"
    assert fakes["safe_added"], "add_safe_globals was never called"
    registered = {cls.__name__ for group in fakes["safe_added"] for cls in group}
    assert "CrossEntropyLoss" in registered


def test_load_failure_sets_failed_state(monkeypatch, tmp_path):
    install_fakes(monkeypatch)
    settings = make_settings()

    service = ModelService(settings)

    with pytest.raises(RuntimeError):
        service.load()
    assert service.state == "failed"
    assert service.failure == "RuntimeError"
