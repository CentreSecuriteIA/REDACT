"""Load measurement, against a fake ``torch``. No GPU and no real torch needed."""

import builtins
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from redact.llms.resources import measure

GIB = 1024 ** 3


class _FakeCuda:
    """Per-device free memory and allocator counters, as ``torch.cuda`` reports them."""

    def __init__(self, devices=1, free_gib=20, total_gib=24, allocated_gib=0):
        self.free = [free_gib * GIB] * devices
        self.total = [total_gib * GIB] * devices
        self.allocated = [allocated_gib * GIB] * devices
        self.peak = list(self.allocated)
        self.available = True

    def is_available(self):
        return self.available

    def device_count(self):
        return len(self.free)

    def mem_get_info(self, device=0):
        return self.free[device], self.total[device]

    def memory_allocated(self, device=0):
        return self.allocated[device]

    def max_memory_allocated(self, device=0):
        return self.peak[device]

    def reset_peak_memory_stats(self, device=0):
        self.peak[device] = self.allocated[device]

    def load(self, device, gib, in_process=True):
        """Take ``gib`` on a device, through this process's allocator or not."""
        self.free[device] -= gib * GIB
        if in_process:
            self.allocated[device] += gib * GIB
            self.peak[device] = max(self.peak[device], self.allocated[device])


def _fake_torch(**kwargs):
    cuda = _FakeCuda(**kwargs)
    return SimpleNamespace(cuda=cuda), cuda


def _measuring(torch):
    return patch.object(measure, "_torch", return_value=torch)


def _import_raising(exc):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise exc
        return real_import(name, *args, **kwargs)

    return patch.object(builtins, "__import__", fake_import)


class TestTorchAccess:
    def test_missing_torch_is_none(self):
        with _import_raising(ImportError("No module named 'torch'")):
            assert measure._torch() is None
            assert measure.free_total_gib() is None

    def test_broken_torch_install_is_none_not_an_error(self):
        """A torch whose DLLs fail to load raises OSError on import."""
        with _import_raising(OSError("[WinError 126] fbgemm.dll")):
            assert measure._torch() is None
            assert measure.free_total_gib() is None
            with measure.Measurement() as m:
                pass
        assert m.claimed_gib is None and m.weights_gib is None

    def test_no_cuda_is_none(self):
        torch, cuda = _fake_torch()
        cuda.available = False
        with patch.dict("sys.modules", {"torch": torch}):
            assert measure._torch() is None
            with measure.Measurement() as m:
                pass
        assert (m.claimed_gib, m.weights_gib) == (None, None)

    def test_failing_cuda_check_is_none(self):
        torch, cuda = _fake_torch()
        cuda.is_available = lambda: 1 / 0
        with patch.dict("sys.modules", {"torch": torch}):
            assert measure._torch() is None

    def test_free_total_reports_the_current_device(self):
        torch, _ = _fake_torch(devices=2)
        with patch.dict("sys.modules", {"torch": torch}):
            assert measure.free_total_gib() == (20.0, 24.0)


class TestMeasurement:
    def test_one_gpu(self):
        torch, cuda = _fake_torch()
        with _measuring(torch), measure.Measurement() as m:
            cuda.load(0, 6)
        assert m.claimed_gib == 6.0
        assert m.weights_gib == 6.0
        assert m.before_free_gib == 20.0

    def test_two_gpus_are_summed(self):
        torch, cuda = _fake_torch(devices=2)
        with _measuring(torch), measure.Measurement() as m:
            cuda.load(0, 10)
            cuda.load(1, 7)
        assert m.claimed_gib == 17.0
        assert m.weights_gib == 17.0

    def test_a_load_on_the_second_gpu_only_is_seen(self):
        torch, cuda = _fake_torch(devices=2)
        with _measuring(torch), measure.Measurement() as m:
            cuda.load(1, 5)
        assert (m.claimed_gib, m.weights_gib) == (5.0, 5.0)

    def test_weights_exclude_what_was_already_allocated(self):
        torch, cuda = _fake_torch(devices=2, allocated_gib=3)
        with _measuring(torch), measure.Measurement() as m:
            cuda.load(0, 4)
        assert m.before_allocated_gib == 6.0
        assert m.weights_gib == 4.0

    def test_out_of_process_load_has_no_weights_figure(self):
        """vLLM's engine process takes the memory; this allocator sees nothing."""
        torch, cuda = _fake_torch()
        with _measuring(torch), measure.Measurement() as m:
            cuda.load(0, 18, in_process=False)
        assert m.claimed_gib == 18.0
        assert m.weights_gib is None

    def test_memory_freed_elsewhere_does_not_go_negative(self):
        torch, cuda = _fake_torch()
        with _measuring(torch), measure.Measurement() as m:
            cuda.free[0] += 2 * GIB
        assert m.claimed_gib == 0.0

    def test_exception_in_the_block_propagates(self):
        torch, cuda = _fake_torch()
        failure = pytest.raises(RuntimeError, match="load blew up")
        with _measuring(torch), failure, measure.Measurement() as m:
            cuda.load(0, 2)
            raise RuntimeError("load blew up")
        assert m.claimed_gib == 2.0

    def test_keyboard_interrupt_propagates(self):
        torch, _ = _fake_torch()
        interrupt = pytest.raises(KeyboardInterrupt)
        with _measuring(torch), interrupt, measure.Measurement():
            raise KeyboardInterrupt

    def test_a_failing_probe_never_fails_the_load(self):
        def boom(*_args):
            raise RuntimeError("CUDA error")

        torch, cuda = _fake_torch()
        cuda.max_memory_allocated = boom
        with _measuring(torch), measure.Measurement() as m:
            cuda.load(0, 1)
        assert (m.claimed_gib, m.weights_gib) == (1.0, None)

        torch, cuda = _fake_torch()
        cuda.mem_get_info = boom
        with _measuring(torch), measure.Measurement() as m:
            cuda.load(0, 1)
        assert (m.claimed_gib, m.weights_gib) == (None, None)

    def test_uncountable_devices_fall_back_to_the_current_one(self):
        torch, cuda = _fake_torch(devices=2)
        cuda.device_count = lambda: 1 / 0
        with _measuring(torch), measure.Measurement() as m:
            cuda.load(0, 3)
            cuda.load(1, 3)
        assert m.claimed_gib == 3.0
