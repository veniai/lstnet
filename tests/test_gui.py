"""Offscreen smoke test for the PySide6 GUI (no display needed)."""
from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _drain_worker(app, win, timeout=5.0):
    """Wait for the active GUI worker thread to finish and flush its signals.

    ``_compute``/``_validate`` run on a :class:`QThread`; the table/stats update
    happens in the ``finished`` slot, which only runs once the main-thread event
    loop processes the queued signal. Tests must drain the loop after kicking the
    worker, or they race the async result (and a Validate-then-Export sequence
    would hit the "nothing to export" modal dialog while the worker is mid-flight).
    """
    deadline = time.monotonic() + timeout
    while win._worker is not None:
        worker = win._worker
        worker.wait(100)  # block up to 100ms for run() to return
        app.processEvents()  # deliver the queued finished signal -> _worker=None
        if time.monotonic() > deadline:
            raise AssertionError("GUI worker did not finish within timeout")



def test_gui_window_constructs_with_25_sites():
    pytest = __import__("pytest")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from lstnet.gui import LSTNetWindow

    app = QApplication.instance() or QApplication([])
    win = LSTNetWindow()
    # all three networks checked by default -> all 25 sites listed
    assert win.site_list.count() == 25
    # emissivity combo offers the source choices
    assert win.emiss_combo.count() >= 3
    # results table + embedded plot canvas exist
    assert win.table.columnCount() == 5
    assert win.canvas is not None


def test_selecting_one_network_filters_sites():
    pytest = __import__("pytest")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from lstnet.gui import LSTNetWindow

    app = QApplication.instance() or QApplication([])
    win = LSTNetWindow()
    win.net_checks["SURFRAD"].setChecked(False)
    win.net_checks["PKULSTNet"].setChecked(False)
    # only HiWATER checked -> 11 sites
    assert win.site_list.count() == 11


def test_gui_compute_drives_compute_through_widgets(monkeypatch):
    """Exercise _compute through the real widgets (CI-safe: compute is faked,
    so no station data / network needed). Guards the processEvents + wiring path
    that the construction smoke test doesn't reach."""
    pytest = __import__("pytest")
    pytest.importorskip("PySide6")
    from datetime import datetime, timezone

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    import lstnet.gui
    from lstnet.gui import LSTNetWindow
    from lstnet.models import GroundLST

    def _fake(site, t, emiss, reader, **kw):
        return GroundLST(t, site, 280.0, 0.95, "Day", "OK")

    monkeypatch.setattr(lstnet.gui, "compute_ground_lst", _fake)

    app = QApplication.instance() or QApplication([])
    win = LSTNetWindow()
    for i in range(win.site_list.count()):
        if win.site_list.item(i).data(Qt.UserRole).name == "psu":
            win.site_list.item(i).setCheckState(Qt.Checked)
    win.times_edit.setPlainText("202007141413")
    win.emiss_combo.setCurrentIndex(0)  # Fixed 0.95
    win._compute()
    _drain_worker(app, win)
    assert win.table.rowCount() == 1
    assert win.table.item(0, 0).text() == "202007141413"
    assert win.table.item(0, 2).text() == "280.000"
    assert win.table.item(0, 4).text() == "OK"


def test_data_folder_defaults_to_project_data(monkeypatch):
    pytest = __import__("pytest")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from lstnet import config
    from lstnet.gui import LSTNetWindow

    app = QApplication.instance() or QApplication([])
    win = LSTNetWindow()
    expected = str(config.project_root() / "data")
    assert win.data_folder == expected
    assert win.data_folder_edit.text() == expected
    assert win.data_folder_edit.isReadOnly()


def test_compute_passes_data_dir_to_reader(monkeypatch, tmp_path):
    """The chosen data folder must be threaded through to each reader as the
    network-specific subdir, so users can relocate their data off the repo."""
    pytest = __import__("pytest")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    import lstnet.gui
    from lstnet.gui import LSTNetWindow
    from lstnet.models import GroundLST

    seen = {}

    class _SpyReader:
        def __init__(self, data_dir=None, **kw):
            seen["data_dir"] = str(data_dir)

        def read_radiation(self, *a, **kw):
            return None

    def _fake(site, t, emiss, reader, **kw):
        return GroundLST(t, site, 280.0, 0.95, "Day", "OK")

    monkeypatch.setattr(lstnet.gui, "compute_ground_lst", _fake)
    monkeypatch.setitem(lstnet.gui._READERS, "SURFRAD", _SpyReader)

    app = QApplication.instance() or QApplication([])
    win = LSTNetWindow()
    custom = str(tmp_path / "mydata")
    win.data_folder = custom
    for i in range(win.site_list.count()):
        if win.site_list.item(i).data(Qt.UserRole).name == "psu":
            win.site_list.item(i).setCheckState(Qt.Checked)
    win.times_edit.setPlainText("202007141413")
    win._compute()
    _drain_worker(app, win)
    assert seen["data_dir"] == str(tmp_path / "mydata" / "SURFRAD")


def test_earthdata_credentials_roundtrip(tmp_path, monkeypatch):
    """Settings dialog saves JSON (chmod 600) + env vars; startup re-reads them."""
    pytest = __import__("pytest")
    pytest.importorskip("PySide6")
    import json
    import os

    from PySide6.QtWidgets import QApplication

    import lstnet.gui

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    cred_path = fake_home / ".lstnet" / "earthdata.json"
    monkeypatch.setattr(lstnet.gui, "_EARTHDATA_FILE", cred_path)
    monkeypatch.setattr(
        lstnet.gui.Path, "home", staticmethod(lambda: fake_home)
    )
    os.environ.pop("EARTHDATA_USERNAME", None)
    os.environ.pop("EARTHDATA_PASSWORD", None)

    app = QApplication.instance() or QApplication([])
    win = lstnet.gui.LSTNetWindow()
    assert "EARTHDATA_USERNAME" not in os.environ  # no file yet

    # Write creds directly through the same path the dialog uses, then reload.
    cred_path.parent.mkdir(parents=True, exist_ok=True)
    cred_path.write_text(json.dumps({"username": "alice", "password": "s3cr3t"}))
    import os as _os
    _os.chmod(cred_path, 0o600)
    lstnet.gui.LSTNetWindow._load_earthdata_env()
    assert os.environ["EARTHDATA_USERNAME"] == "alice"
    assert os.environ["EARTHDATA_PASSWORD"] == "s3cr3t"


def test_load_retrieved_rejects_csv_missing_required_columns(tmp_path, monkeypatch):
    """A CSV without the 3 required columns must not load; the user is warned
    and the validate button stays disabled."""
    pytest = __import__("pytest")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    import lstnet.gui
    from lstnet.gui import LSTNetWindow

    bad = tmp_path / "bad.csv"
    bad.write_text("site,time_only,lst\nbon,202007141413,280\n")  # wrong names

    rejected = {}
    monkeypatch.setattr(
        lstnet.gui.QMessageBox, "warning",
        lambda *a, **k: rejected.setdefault("called", True),
    )
    monkeypatch.setattr(
        lstnet.gui.QFileDialog, "getOpenFileName",
        lambda *a, **k: (str(bad), ""),
    )

    app = QApplication.instance() or QApplication([])
    win = LSTNetWindow()
    win._load_retrieved()
    assert rejected.get("called") is True
    assert win._retr_path is None  # not loaded
    assert not win.validate_btn.isEnabled()


def test_load_retrieved_accepts_csv_with_required_columns(tmp_path, monkeypatch):
    """A CSV with the 3 required columns (extra columns OK) loads, labels the
    file, and enables the Validate button."""
    pytest = __import__("pytest")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    import lstnet.gui
    from lstnet.gui import LSTNetWindow

    good = tmp_path / "good.csv"
    good.write_text(
        "site,overpass_time_utc,lst_k,extra_note\n"
        "bon,201102121430,272.5,hello\n"
    )

    monkeypatch.setattr(
        lstnet.gui.QFileDialog, "getOpenFileName",
        lambda *a, **k: (str(good), ""),
    )
    app = QApplication.instance() or QApplication([])
    win = LSTNetWindow()
    win._load_retrieved()
    assert win._retr_path == str(good)
    assert win.validate_btn.isEnabled()


def test_load_retrieved_previews_csv_in_table(tmp_path, monkeypatch):
    """After a successful load the CSV rows are shown immediately in the results
    table (3-column preview), so the user can verify format before validating."""
    pytest = __import__("pytest")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    import lstnet.gui
    from lstnet.gui import LSTNetWindow

    good = tmp_path / "good.csv"
    good.write_text(
        "site,overpass_time_utc,lst_k\n"
        "bon,201102121430,272.5\n"
        "bon,201102121600,278.0\n"
    )
    monkeypatch.setattr(
        lstnet.gui.QFileDialog, "getOpenFileName",
        lambda *a, **k: (str(good), ""),
    )

    app = QApplication.instance() or QApplication([])
    win = LSTNetWindow()
    win._load_retrieved()
    # table switched to 3-column preview
    assert win.table.columnCount() == 3
    headers = [win.table.horizontalHeaderItem(i).text() for i in range(3)]
    assert headers == ["Site", "Time", "Retrieved (K)"]
    assert win.table.rowCount() == 2
    assert win.table.item(0, 0).text() == "bon"
    assert win.table.item(0, 1).text() == "201102121430"
    assert win.table.item(0, 2).text() == "272.500"
    assert win._last_mode == "loaded"
    assert "Loaded 2 rows" in win.status.currentMessage()


def test_validate_shows_progress_feedback_then_restores_button(tmp_path, monkeypatch):
    """Validate disables + relabels its button during computation and restores
    it after — the user must see obvious progress feedback."""
    pytest = __import__("pytest")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    import lstnet.gui
    from lstnet.gui import LSTNetWindow
    from lstnet.models import GroundLST

    csv_path = tmp_path / "retr.csv"
    csv_path.write_text("site,overpass_time_utc,lst_k\nbon,201102121430,275.0\n")

    seen = {}
    win_ref = {}

    def _fake(site, t, emiss, reader, **kw):
        # capture button + status state mid-computation
        win = win_ref.get("win")
        if win is not None:
            seen["mid_text"] = win.validate_btn.text()
            seen["mid_enabled"] = win.validate_btn.isEnabled()
            seen["mid_status"] = win.status.currentMessage()
        return GroundLST(t, site, 273.0, 0.95, "Day", "OK")

    monkeypatch.setattr(lstnet.gui, "compute_ground_lst", _fake)

    app = QApplication.instance() or QApplication([])
    win = LSTNetWindow()
    win_ref["win"] = win
    win._retr_path = str(csv_path)
    win.validate_btn.setEnabled(True)
    win._validate()
    _drain_worker(app, win)
    # mid-computation: button said "Validating…" and was disabled, status showed progress
    assert seen.get("mid_text") == "Validating…"
    assert seen.get("mid_enabled") is False
    assert "Validating — computing ground LST for 1 sites" in seen.get("mid_status", "")
    # after completion: button text + enabled state restored
    assert win.validate_btn.text() == "Validate"
    assert win.validate_btn.isEnabled() is True
    # final status reflects the validation summary (overrides the progress message)
    assert "Validated 1 pairs" in win.status.currentMessage()


def test_validate_is_self_contained_and_enriches_table(tmp_path, monkeypatch):
    """Right-panel Validate reads the CSV, computes ground per row (faked),
    and switches the results table to 6 paired columns — no left Compute needed."""
    pytest = __import__("pytest")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    import lstnet.gui
    from lstnet.gui import LSTNetWindow
    from lstnet.models import GroundLST

    csv_path = tmp_path / "retr.csv"
    csv_path.write_text(
        "site,overpass_time_utc,lst_k\nbon,201102121430,275.0\nbon,201102121600,278.0\n"
    )

    def _fake(site, t, emiss, reader, **kw):
        return GroundLST(t, site, 273.0, 0.95, "Day", "OK")

    monkeypatch.setattr(lstnet.gui, "compute_ground_lst", _fake)

    app = QApplication.instance() or QApplication([])
    win = LSTNetWindow()
    # No left-panel compute done — _ground is empty, table starts at 5 cols.
    assert win.table.columnCount() == 5
    win._retr_path = str(csv_path)
    win.validate_btn.setEnabled(True)
    win._validate()
    _drain_worker(app, win)

    # table reconfigured to 6 paired columns
    assert win.table.columnCount() == 7
    headers = [win.table.horizontalHeaderItem(i).text() for i in range(7)]
    assert headers == ["Site", "Time", "Retrieved (K)", "Ground (K)", "Diff (K)", "Emiss", "QC"]
    assert win.table.rowCount() == 2
    # diff = retrieved - ground = 275.0 - 273.0 = +2.00
    assert win.table.item(0, 4).text() == "+2.000"
    assert win._last_mode == "validation"
    assert win.validate_btn.isEnabled()
    # status bar reflects the validation summary
    assert "Validated 2 pairs" in win.status.currentMessage()


def test_export_after_validate_writes_paired_columns(tmp_path, monkeypatch):
    """After validation, Export writes the enriched 6-column paired CSV
    (not the 5-column ground-only layout)."""
    pytest = __import__("pytest")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    import lstnet.gui
    from lstnet.gui import LSTNetWindow
    from lstnet.models import GroundLST

    csv_path = tmp_path / "retr.csv"
    csv_path.write_text("site,overpass_time_utc,lst_k\nbon,201102121430,275.0\n")

    out = tmp_path / "out.csv"

    def _fake(site, t, emiss, reader, **kw):
        return GroundLST(t, site, 273.0, 0.95, "Day", "OK")

    monkeypatch.setattr(lstnet.gui, "compute_ground_lst", _fake)
    monkeypatch.setattr(
        lstnet.gui.QFileDialog, "getSaveFileName",
        lambda *a, **k: (str(out), ""),
    )

    app = QApplication.instance() or QApplication([])
    win = LSTNetWindow()
    win._retr_path = str(csv_path)
    win._validate()
    _drain_worker(app, win)  # finish before export, else the empty-result modal hangs
    win._export()

    text = out.read_text()
    lines = text.strip().splitlines()
    assert lines[0] == "site,overpass_time_utc,lst_retrieved_k,lst_ground_k,diff_k,emissivity,qc"
    assert lines[1].startswith("bon,201102121430,275.0,273.0,2.0,")


def test_compute_then_validate_reconfigures_table(monkeypatch, tmp_path):
    """Whichever operation ran last wins: left Compute sets 5 cols, right
    Validate then switches to 6 cols, replacing prior compute results."""
    pytest = __import__("pytest")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    import lstnet.gui
    from lstnet.gui import LSTNetWindow
    from lstnet.models import GroundLST

    def _fake(site, t, emiss, reader, **kw):
        return GroundLST(t, site, 273.0, 0.95, "Day", "OK")

    monkeypatch.setattr(lstnet.gui, "compute_ground_lst", _fake)

    csv_path = tmp_path / "retr.csv"
    csv_path.write_text("site,overpass_time_utc,lst_k\nbon,201102121430,275.0\n")

    app = QApplication.instance() or QApplication([])
    win = LSTNetWindow()
    # left compute
    for i in range(win.site_list.count()):
        if win.site_list.item(i).data(Qt.UserRole).name == "bon":
            win.site_list.item(i).setCheckState(Qt.Checked)
    win.times_edit.setPlainText("202007141413")
    win._compute()
    _drain_worker(app, win)
    assert win.table.columnCount() == 5
    assert win._last_mode == "ground"
    # right validate (independent) — table switches to 6 cols
    win._retr_path = str(csv_path)
    win._validate()
    _drain_worker(app, win)
    assert win.table.columnCount() == 7
    assert win._last_mode == "validation"


def test_sample_files_parse():
    """The two shipped sample files are well-formed for the validate demo."""
    import csv
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    times = (root / "samples" / "overpass_times.txt").read_text()
    tokens = re.findall(r"\d{12}", times)
    assert tokens == [
        "201102120400", "201102121000", "201102121430",
        "201102121600", "201102122200",
    ]
    with open(root / "samples" / "retrieved_sample.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 5
    assert len({r["site"] for r in rows}) >= 3  # multi-site
    assert all(r["source"] == "demo-algo" for r in rows)
