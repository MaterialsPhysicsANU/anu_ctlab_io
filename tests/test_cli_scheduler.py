import importlib.util
from pathlib import Path

CLI_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "cli" / "src" / "anu_ctlab_io_cli" / "_cli.py"
)
SPEC = importlib.util.spec_from_file_location("anu_ctlab_io_cli._cli", CLI_MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
CLI_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI_MODULE)


def test_resolve_scheduler_auto_defaults_to_threads():
    assert (
        CLI_MODULE._resolve_scheduler(CLI_MODULE.Scheduler.auto)
        == CLI_MODULE.Scheduler.threads
    )


def test_is_running_under_mpi_detects_openmpi_env():
    assert CLI_MODULE._is_running_under_mpi({"OMPI_COMM_WORLD_SIZE": "4"})


def test_is_running_under_mpi_detects_pmi_env():
    assert CLI_MODULE._is_running_under_mpi({"PMI_RANK": "1"})


def test_resolve_scheduler_preserves_explicit_selection():
    assert (
        CLI_MODULE._resolve_scheduler(CLI_MODULE.Scheduler.processes)
        == CLI_MODULE.Scheduler.processes
    )
