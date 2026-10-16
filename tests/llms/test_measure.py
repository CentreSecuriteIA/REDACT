"""Load measurement against a fake ``torch``, card detection against a fake
``nvidia-smi``. No GPU and no real torch needed."""

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


def _nvidia_smi(monkeypatch, out):
    """Make ``nvidia-smi`` answer every query with ``out``."""
    monkeypatch.setattr(measure.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(measure.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(stdout=out))


class TestCards:
    """The machine's cards, and the ones ``CUDA_VISIBLE_DEVICES`` leaves visible."""

    TWO = "NVIDIA RTX A5000\nNVIDIA RTX A5000\n"

    def test_visible_count_follows_cuda_visible_devices(self, monkeypatch):
        _nvidia_smi(monkeypatch, self.TWO)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        assert measure.detect_gpus(visible=True) == ("NVIDIA RTX A5000", 1)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert measure.detect_gpus(visible=True) == ("NVIDIA RTX A5000", 0)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
        assert measure.detect_gpus(visible=True) == ("NVIDIA RTX A5000", 2)

    def test_billing_count_ignores_cuda_visible_devices(self, monkeypatch):
        _nvidia_smi(monkeypatch, self.TWO)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        assert measure.detect_gpus() == ("NVIDIA RTX A5000", 2)

    def test_no_nvidia_smi_is_no_card_whatever_is_visible(self, monkeypatch):
        monkeypatch.setattr(measure.shutil, "which", lambda _: None)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        assert measure.detect_gpus(visible=True) == (None, 0)

    def test_visible_ids_are_the_variable_split_or_none(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert measure.visible_gpu_ids() is None
        for value, ids in (("", []), ("1", ["1"]), ("2, 0", ["2", "0"]),
                           ("GPU-8bce,GPU-0182", ["GPU-8bce", "GPU-0182"])):
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
            assert measure.visible_gpu_ids() == ids

    def test_memory_is_the_smallest_card(self, monkeypatch):
        _nvidia_smi(monkeypatch, "32768\n24576\n")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert measure.detect_gpu_memory_gib() == 24.0

    @pytest.mark.parametrize("visible,gib", [
        ("0", 24.0), ("1", 12.0), ("1,0", 12.0),
        # Nothing to narrow to: no card, or cards nvidia-smi's list cannot name.
        ("", 12.0), ("GPU-8bce", 12.0),
        ("5", None),
    ])
    def test_memory_is_the_smallest_visible_card(self, monkeypatch, visible, gib):
        """The planner counts the visible cards: a hidden 12 GiB card must
        not size the 24 GiB one."""
        _nvidia_smi(monkeypatch, "24576\n12288\n")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
        assert measure.detect_gpu_memory_gib() == gib

    @pytest.mark.parametrize("out", ["24576\n[N/A]\n", ""])
    def test_memory_that_cannot_be_read_is_none(self, monkeypatch, out):
        _nvidia_smi(monkeypatch, out)
        assert measure.detect_gpu_memory_gib() is None


# The suite's runs see no card (conftest); these tests read the real probe.
_card_use = measure.card_use


def _nvidia_smi_answers(monkeypatch, cards, apps):
    """Make ``nvidia-smi`` answer the card query with ``cards`` and the
    compute-process query with ``apps``; ``None`` makes that query fail."""
    def run(args, **kwargs):
        out = apps if any("compute-apps" in a for a in args) else cards
        if out is None:
            raise measure.subprocess.CalledProcessError(9, args)
        return SimpleNamespace(stdout=out)

    monkeypatch.setattr(measure.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(measure.subprocess, "run", run)


class TestCardUse:
    """What each card holds, and which processes hold it."""

    CARDS = ("0, GPU-aaaa, 29070, 49140\n"
             "1, GPU-bbbb, 2, 49140\n")

    def test_each_card_with_the_processes_on_it(self, monkeypatch):
        _nvidia_smi_answers(monkeypatch, self.CARDS,
                            "21131, 29068, GPU-aaaa, VLLM::EngineCore\n")
        assert _card_use() == [
            measure.CardUse(index=0, uuid="GPU-aaaa", used_gib=28.4, total_gib=48.0,
                            processes=((21131, "VLLM::EngineCore", 28.4),)),
            measure.CardUse(index=1, uuid="GPU-bbbb", used_gib=0.0, total_gib=48.0),
        ]

    def test_a_process_without_a_memory_figure_is_still_listed(self, monkeypatch):
        """Where nvidia-smi cannot say what a process holds, it says [N/A]."""
        _nvidia_smi_answers(monkeypatch, self.CARDS,
                            "4242, [N/A], GPU-bbbb, /usr/bin/python3, with a comma\n")
        (_, second) = _card_use()
        assert second.processes == ((4242, "/usr/bin/python3, with a comma", None),)

    @pytest.mark.parametrize("cards,apps", [
        (None, ""), (CARDS, None), ("0, GPU-aaaa, [N/A], 49140\n", ""), ("", "")])
    def test_an_answer_that_cannot_be_read_is_none(self, monkeypatch, cards, apps):
        _nvidia_smi_answers(monkeypatch, cards, apps)
        assert _card_use() is None

    def test_no_nvidia_smi_is_none(self, monkeypatch):
        monkeypatch.setattr(measure.shutil, "which", lambda _: None)
        assert _card_use() is None
