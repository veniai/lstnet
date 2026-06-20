"""LSTNet desktop GUI (PySide6).

A comfortable, batch-capable front-end over the :mod:`lstnet` library:
multi-select sites, paste overpass times, pick an emissivity source, compute
ground-truth LST, then validate against a retrieved-LST CSV (bias/RMSE/R +
embedded scatter plot). Run with ``lstnet-gui`` or ``python -m lstnet.gui``.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from lstnet import config
from lstnet import FixedEmissivity, __version__, compute_ground_lst, validate
from lstnet.io.hiwater import HiwaterReader
from lstnet.io.pku import PkuReader
from lstnet.io.surfrad import SurfradReader
from lstnet.sites import SITES
from lstnet.validation import TableRetrievedLST

_READERS = {"SURFRAD": SurfradReader, "PKULSTNet": PkuReader, "HiWATER": HiwaterReader}
_NETWORKS = ("SURFRAD", "PKULSTNet", "HiWATER")
# Subdirectory under the chosen data root where each reader expects its files.
# Readers default to ``project_root()/data/<subdir>``; when the user picks a
# different data root we anchor each reader at ``<data_root>/<subdir>`` so the
# on-disk layout is preserved.
_NETWORK_SUBDIR = {"SURFRAD": "SURFRAD", "PKULSTNet": "pku-sites", "HiWATER": "HiWATER"}
# One-line hint for where/how each network's data lands. Shown in the network
# checkbox tooltip, the status bar on toggle, and the folder README.
_NETWORK_DATA_HINT = {
    "SURFRAD": "auto-downloaded from NOAA (online; needs network)",
    "PKULSTNet": "place CR200 .dat files here (hbc/hnw/hnh/xzl/imb/xjf/cqb)",
    "HiWATER": "place <year>/*.xlsx workbooks here (UL_Cor/DL_Cor columns)",
}
# Per-column pixel widths keyed by table mode. Site stays narrow, Time wide;
# the last column always stretches to fill the panel.
_TABLE_WIDTHS = {
    "loaded": (95, 135, 130),
    "ground": (135, 95, 110, 90, 90),
    "validation": (95, 135, 110, 110, 90, 75, 90),
}
_EARTHDATA_FILE = Path.home() / ".lstnet" / "earthdata.json"


# ---------------------------------------------------------------------------
# Background workers — heavy computation runs off the GUI thread so the window
# never shows "Not Responding" while SURFRAD/MODIS downloads are in flight.
# Each worker emits its result via a Qt signal; the main thread updates the
# table/labels/canvas only from the connected slot (never from run()).
# ---------------------------------------------------------------------------


class _ComputeWorker(QThread):
    """Runs the ground-LST computation loop off the GUI thread.

    Captures only plain-Python inputs (no Qt widgets) and emits the resulting
    list of :class:`GroundLST`. The main thread wires ``finished`` to the
    table/stats update and ``error`` to a status-bar message.
    """

    finished = Signal(list)
    error = Signal(str)

    def __init__(self, sites, tokens, emiss_src, data_folder):
        super().__init__()
        self.site_time_pairs = [(s, None) for s in sites]
        self.tokens = tokens
        self.emiss_src = emiss_src
        self.data_folder = data_folder

    def run(self):
        try:
            results = _compute_ground_loop(
                self.site_time_pairs, self.tokens, self.emiss_src, self.data_folder
            )
            self.finished.emit(results)
        except Exception as e:  # noqa: BLE001 — surface any failure to the UI
            self.error.emit(str(e))


class _ValidateWorker(QThread):
    """Runs ground-LST computation + validation off the GUI thread.

    Emits the resulting :class:`ValidationResult`. Stats/plot updates happen
    on the main thread via the connected slot.
    """

    finished = Signal(object)
    error = Signal(str)

    def __init__(self, retr_items, emiss_src, data_folder):
        super().__init__()
        self.retr_items = retr_items
        self.emiss_src = emiss_src
        self.data_folder = data_folder

    def run(self):
        try:
            ground = _compute_ground_loop(
                [(r.site, r.overpass_time) for r in self.retr_items],
                None,
                self.emiss_src,
                self.data_folder,
            )
            result = validate(ground, self.retr_items)
            self.finished.emit(result)
        except Exception as e:  # noqa: BLE001
            self.error.emit(str(e))


def _compute_ground_loop(site_time_pairs, tokens, emiss_src, data_folder):
    """Pure (non-Qt) ground-LST computation loop shared by both workers.

    ``site_time_pairs`` is always a list of ``(site, datetime|None)`` tuples.
    When ``datetime`` is ``None`` the matching ``tokens`` are expanded — every
    (site, token) combination is computed (the left-Compute path). When a
    concrete datetime is supplied, that single (site, time) is computed and
    ``tokens`` is ignored (the right-Validate path).
    """
    results: list = []
    for site, t_fixed in site_time_pairs:
        subdir = _NETWORK_SUBDIR.get(site.network, "")
        reader_data_dir = (
            str(Path(data_folder) / subdir) if subdir else data_folder
        )
        reader = _READERS[site.network](data_dir=reader_data_dir)
        time_list = [t_fixed] if t_fixed is not None else (
            datetime.strptime(tok, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            for tok in (tokens or [])
        )
        for t in time_list:
            try:
                g = compute_ground_lst(site, t, emiss_src, reader)
            except (ValueError, Exception):
                continue
            if g is not None:
                results.append(g)
    return results


class LSTNetWindow(QMainWindow):
    """Main LSTNet validation window."""

    _update_available = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"LSTNet v{__version__} — LST validation")
        self.resize(1180, 780)
        self._ground: list = []
        self._validation_pairs: list = []
        self._last_mode: str | None = None  # "ground" | "validation"
        self._unmatched_summary: str = ""
        self._retr_path: str | None = None
        # PyInstaller bundles the package under _MEIPASS, so project_root()
        # points inside the bundle — there is no user data there. Default to
        # the current working directory instead (where the exe is launched).
        if getattr(sys, "frozen", False):
            self.data_folder: str = str(Path.cwd())
        else:
            self.data_folder = str(config.project_root() / "data")
        self._worker: QThread | None = None  # active background worker, if any
        self._load_earthdata_env()
        self._build_ui()
        self._populate_sites()
        self._ensure_data_subdirs()
        self._update_available.connect(self._on_update_available)
        threading.Thread(target=self._check_update, daemon=True).start()

    # --- PyPI update check --------------------------------------------------

    def _check_update(self):
        """Daemon-thread check of the latest lstnet version on PyPI.

        Silent on any failure (no network, timeout, parse error). Emits
        ``_update_available`` only when the remote version differs from the
        installed ``lstnet.__version__``.
        """
        try:
            import requests

            import lstnet

            resp = requests.get("https://pypi.org/pypi/lstnet/json", timeout=3)
            resp.raise_for_status()
            latest = resp.json()["info"]["version"]
            current = lstnet.__version__
            if latest != current:
                self._update_available.emit(str(latest), str(current))
        except Exception:
            pass

    def _on_update_available(self, latest: str, current: str):
        """Show a small blue notice that a newer version exists."""
        if getattr(sys, "frozen", False):
            text = f"🔄 v{latest} available — download from GitHub Releases"
        else:
            text = f"🔄 v{latest} available — pip install --upgrade lstnet"
        self.update_label.setText(text)
        self.update_label.setStyleSheet("color: #0066cc; font-size: 11px;")
        self.update_label.setHidden(False)

    # --- UI construction ----------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # left: inputs
        left = QWidget()
        lv = QVBoxLayout(left)

        self.update_label = QLabel(f"LSTNet v{__version__}")
        self.update_label.setStyleSheet("color: #555; font-size: 11px;")
        lv.addWidget(self.update_label)

        data_grp = QGroupBox("Data folder")
        dg = QHBoxLayout(data_grp)
        self.data_folder_edit = QLineEdit(self.data_folder)
        self.data_folder_edit.setReadOnly(True)
        self.data_folder_edit.setToolTip(
            "Root data directory. Readers look under <root>/SURFRAD, "
            "<root>/pku-sites, <root>/HiWATER; emissivity caches under "
            "<root>/MODIS, <root>/ASTER_GED."
        )
        dg.addWidget(self.data_folder_edit, 1)
        b_browse = QPushButton("Browse…")
        b_browse.clicked.connect(self._browse_data_folder)
        dg.addWidget(b_browse)
        lv.addWidget(data_grp)

        net_grp = QGroupBox("Networks")
        ng = QHBoxLayout(net_grp)
        self.net_checks: dict[str, QCheckBox] = {}
        for net in _NETWORKS:
            cb = QCheckBox(net)
            cb.setChecked(True)
            cb.setToolTip(
                f"Data: <root>/{_NETWORK_SUBDIR[net]} — {_NETWORK_DATA_HINT[net]}"
            )
            cb.toggled.connect(lambda _checked, n=net: self._on_network_toggled(n))
            self.net_checks[net] = cb
            ng.addWidget(cb)
        ng.addStretch()
        lv.addWidget(net_grp)

        sites_grp = QGroupBox("Sites (tick to select)")
        sg = QVBoxLayout(sites_grp)
        self.site_list = QListWidget()
        sg.addWidget(self.site_list)
        btnrow = QHBoxLayout()
        for label, on in (("All", True), ("None", False)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, on=on: self._set_all_sites(on))
            btnrow.addWidget(b)
        sg.addLayout(btnrow)
        lv.addWidget(sites_grp)

        times_grp = QGroupBox("Overpass times — 12-digit YYYYMMDDHHMM (UTC), one per line")
        tg = QVBoxLayout(times_grp)
        self.times_edit = QTextEdit()
        self.times_edit.setPlaceholderText("201102121430\n201102131430")
        self.times_edit.setMinimumHeight(120)
        tg.addWidget(self.times_edit)
        lv.addWidget(times_grp)

        em_grp = QGroupBox("Emissivity")
        eg = QHBoxLayout(em_grp)
        self.emiss_combo = QComboBox()
        self.emiss_combo.addItems(
            ["Fixed 0.98",
             "MODIS daily (C6.1, needs creds)", "ASTER GED (climatological, needs creds)"]
        )
        self.emiss_combo.setCurrentIndex(2)  # default: ASTER GED (best for validation)
        eg.addWidget(self.emiss_combo)
        eg.addWidget(QLabel("or ε:"))
        self.emiss_value = QLineEdit()
        self.emiss_value.setPlaceholderText("manual, e.g. 0.97")
        self.emiss_value.setMaximumWidth(90)
        eg.addWidget(self.emiss_value)
        lv.addWidget(em_grp)

        action_row = QHBoxLayout()
        self.settings_btn = QPushButton("Settings…")
        self.settings_btn.clicked.connect(self._open_settings)
        action_row.addWidget(self.settings_btn)
        self.compute_btn = QPushButton("Compute ground LST")
        self.compute_btn.clicked.connect(self._compute)
        action_row.addWidget(self.compute_btn)
        lv.addLayout(action_row)
        lv.addStretch()
        splitter.addWidget(left)

        # right: results + validate + plot
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(QLabel("Ground-truth LST"))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Time", "Site", "LST (K)", "Emiss", "QC"])
        # Default to Interactive; per-mode widths are applied when results load
        # (see _apply_column_widths) so Site stays narrow and Time stays wide.
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        rv.addWidget(self.table)

        row2 = QHBoxLayout()
        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.clicked.connect(self._export)
        b_clear = QPushButton("Clear")
        b_clear.clicked.connect(self._clear_ground)
        row2.addWidget(self.export_btn)
        row2.addWidget(b_clear)
        rv.addLayout(row2)

        val_grp = QGroupBox("Validate against retrieved LST")
        vg = QVBoxLayout(val_grp)
        vr = QHBoxLayout()
        b_load = QPushButton("Load retrieved CSV…")
        b_load.clicked.connect(self._load_retrieved)
        b_format = QPushButton("Format…")
        b_format.setToolTip("Show the required retrieved-LST CSV format + save a template")
        b_format.clicked.connect(self._show_csv_help)
        self.retr_label = QLabel("no file")
        self.validate_btn = QPushButton("Validate")
        self.validate_btn.setEnabled(False)  # enabled once a valid CSV is loaded
        self.validate_btn.clicked.connect(self._validate)
        vr.addWidget(b_load)
        vr.addWidget(b_format)
        vr.addWidget(self.retr_label, 1)
        vr.addWidget(self.validate_btn)
        vg.addLayout(vr)
        self.stats_label = QLabel("")
        vg.addWidget(self.stats_label)
        self.fig = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.fig)
        vg.addWidget(self.canvas)
        rv.addWidget(val_grp)
        splitter.addWidget(right)
        splitter.setSizes([400, 780])

        self.status = self.statusBar()

    # --- data folder + earthdata credentials --------------------------------

    def _browse_data_folder(self):
        """Open a native directory picker; updates ``self.data_folder``."""
        path = QFileDialog.getExistingDirectory(
            self, "Choose data root folder", self.data_folder
        )
        if path:
            self.data_folder = path
            self.data_folder_edit.setText(path)
            self._ensure_data_subdirs()
            self.status.showMessage(f"Data folder: {path}", 5000)

    @staticmethod
    def _load_earthdata_env():
        """Read ``~/.lstnet/earthdata.json`` (if present) into the environment.

        ``earthaccess`` honours ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``,
        so populating them here lets later MODIS/GED downloads authenticate
        without per-call prompts. The file is created on demand by the
        Settings dialog; absence is silent.
        """
        try:
            _EARTHDATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        if not _EARTHDATA_FILE.exists():
            return
        try:
            data = json.loads(_EARTHDATA_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        username = data.get("username")
        password = data.get("password")
        if username:
            os.environ["EARTHDATA_USERNAME"] = username
        if password:
            os.environ["EARTHDATA_PASSWORD"] = password

    def _open_settings(self):
        """Earthdata credentials dialog — saves to ~/.lstnet/earthdata.json."""
        dlg = QDialog(self)
        dlg.setWindowTitle("NASA Earthdata Login")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("NASA Earthdata Login (for MODIS/GED emissivity sources)"))

        user_edit = QLineEdit()
        user_edit.setPlaceholderText("username or email")
        pw_edit = QLineEdit()
        pw_edit.setEchoMode(QLineEdit.Password)
        # Pre-fill from environment if already loaded.
        existing_user = os.environ.get("EARTHDATA_USERNAME", "")
        if existing_user:
            user_edit.setText(existing_user)
        v.addWidget(user_edit)
        v.addWidget(pw_edit)

        link = QLabel(
            '<a href="https://urs.earthdata.nasa.gov/users/new">'
            "No account? Register at NASA Earthdata</a>"
        )
        link.setOpenExternalLinks(True)
        v.addWidget(link)

        row = QHBoxLayout()
        b_save = QPushButton("Save")
        b_cancel = QPushButton("Cancel")
        row.addStretch()
        row.addWidget(b_save)
        row.addWidget(b_cancel)
        v.addLayout(row)

        def save():
            username = user_edit.text().strip()
            password = pw_edit.text()
            if not username or not password:
                QMessageBox.warning(dlg, "注意", "用户名和密码均不能为空")
                return
            try:
                _EARTHDATA_FILE.parent.mkdir(parents=True, exist_ok=True)
                _EARTHDATA_FILE.write_text(
                    json.dumps({"username": username, "password": password}),
                    encoding="utf-8",
                )
                os.chmod(_EARTHDATA_FILE, 0o600)
            except OSError as e:
                QMessageBox.critical(dlg, "错误", f"保存凭据失败:\n{e}")
                return
            os.environ["EARTHDATA_USERNAME"] = username
            os.environ["EARTHDATA_PASSWORD"] = password
            dlg.accept()
            self.status.showMessage("Credentials saved", 5000)

        b_save.clicked.connect(save)
        b_cancel.clicked.connect(dlg.reject)
        dlg.exec()

    # --- site list ----------------------------------------------------------

    def _populate_sites(self):
        self.site_list.clear()
        for site in sorted(SITES.values(), key=lambda s: (s.network, s.name)):
            if not self.net_checks[site.network].isChecked():
                continue
            it = QListWidgetItem(f"{site.name}  ({site.lon:.2f}, {site.lat:.2f})  {site.network}")
            it.setData(Qt.UserRole, site)
            it.setCheckState(Qt.Unchecked)
            self.site_list.addItem(it)

    # --- data folders (pre-create + instructions) ---------------------------

    def _on_network_toggled(self, net):
        """Rebuild the site list and, when enabled, make the data subdir exist."""
        self._populate_sites()
        if self.net_checks[net].isChecked():
            created, path = self._ensure_data_subdir(net)
            hint = _NETWORK_DATA_HINT.get(net, "")
            if created:
                self.status.showMessage(f"Created {path} — {hint}", 8000)
            else:
                self.status.showMessage(f"{net} data folder: {path}", 5000)

    def _ensure_data_subdirs(self):
        """Create the data subdir for every checked network (mkdir -p)."""
        for net, cb in self.net_checks.items():
            if cb.isChecked():
                self._ensure_data_subdir(net)

    def _ensure_data_subdir(self, net):
        """Create ``<data_folder>/<subdir>`` and, for PKU/HiWATER, a README.

        Returns ``(created_now, path_str)``. ``created_now`` is True when the
        directory did not exist before this call (so the caller can announce it).
        """
        subdir = _NETWORK_SUBDIR.get(net)
        if not subdir:
            return False, self.data_folder
        path = Path(self.data_folder) / subdir
        created = not path.exists()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False, str(path)
        self._write_data_readme(net, path)
        return created, str(path)

    @staticmethod
    def _write_data_readme(net, path):
        """Drop a one-file hint describing what to place in PKU/HiWATER folders.

        SURFRAD is online, so it gets no README (nothing for the user to place).
        Idempotent — never overwrites an existing README.
        """
        bodies = {
            "PKULSTNet": (
                "PKULSTNet ground data\n"
                "=====================\n\n"
                "Place Campbell CR200Series .dat files here (comma-delimited,\n"
                "4-line header), one per site, named after the site code:\n"
                "  hbc.dat  hnw.dat  hnh.dat  xzl.dat  imb.dat  xjf.dat  cqb.dat\n\n"
                "Each file must contain TagTemp_C_Avg(1) and TagTemp_C_Avg(2)\n"
                "brightness-temperature columns.\n"
            ),
            "HiWATER": (
                "HiWATER ground data\n"
                "===================\n\n"
                "Place yearly .xlsx workbooks under a per-year subfolder:\n"
                "  <year>/<year>年黑河流域地表过程综合观测网<中文名>AWS.xlsx\n\n"
                "Workbooks must contain UL_Cor / DL_Cor longwave columns (W/m^2)\n"
                "at 10-minute intervals.\n"
            ),
        }
        body = bodies.get(net)
        if not body:
            return
        readme = path / "README.txt"
        if readme.exists():
            return
        try:
            readme.write_text(body, encoding="utf-8")
        except OSError:
            pass

    def _set_all_sites(self, on):
        state = Qt.Checked if on else Qt.Unchecked
        for i in range(self.site_list.count()):
            self.site_list.item(i).setCheckState(state)

    def _selected_sites(self):
        out = []
        for i in range(self.site_list.count()):
            it = self.site_list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out

    # --- compute ------------------------------------------------------------

    def _emissivity_source(self):
        # Manual input takes priority (if non-empty and valid).
        manual = self.emiss_value.text().strip()
        if manual:
            try:
                return FixedEmissivity(float(manual))
            except ValueError:
                pass  # fall through to combo
        txt = self.emiss_combo.currentText()
        if txt.startswith("Fixed"):
            return FixedEmissivity(float(txt.split()[1]))
        modis_dir = str(Path(self.data_folder) / "MODIS")
        aster_dir = str(Path(self.data_folder) / "ASTER_GED")
        if txt.startswith("MODIS"):
            from lstnet import ModisDailyEmissivity
            return ModisDailyEmissivity(data_dir=modis_dir)
        if txt.startswith("ASTER"):
            from lstnet import AsterGEDEmissivity
            return AsterGEDEmissivity(data_dir=aster_dir)
        return FixedEmissivity(0.95)

    def _compute(self):
        if self._worker is not None and self._worker.isRunning():
            self.status.showMessage("Busy — wait for the current job to finish", 3000)
            return
        sites = self._selected_sites()
        tokens = self.times_edit.toPlainText().split()
        if not sites or not tokens:
            QMessageBox.warning(self, "注意", "请先选站点并输入过境时间")
            return
        try:
            emiss = self._emissivity_source()
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "注意", f"发射率配置无效:\n{e}")
            return
        self._ground = []
        self.compute_btn.setText("Computing…")
        self.compute_btn.setEnabled(False)
        self.settings_btn.setEnabled(False)
        self.status.showMessage("Computing… (network sources may download data)")
        self._worker = _ComputeWorker(sites, tokens, emiss, self.data_folder)
        self._worker.finished.connect(self._on_compute_done)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_compute_done(self, results):
        self._show_ground_results(results)
        self.status.showMessage(
            f"Done — {len(self._ground)} ground-LST values", 5000
        )
        self._finish_worker()
        self.compute_btn.setText("Compute ground LST")
        self.compute_btn.setEnabled(True)
        self.settings_btn.setEnabled(True)

    def _on_worker_error(self, message):
        self.status.showMessage(f"Error: {message}", 8000)
        self._finish_worker()
        self.compute_btn.setText("Compute ground LST")
        self.compute_btn.setEnabled(True)
        self.validate_btn.setText("Validate")
        self.validate_btn.setEnabled(bool(self._retr_path))
        self.settings_btn.setEnabled(True)

    def _finish_worker(self):
        """Drop the worker reference so the next job can start (and let Qt GC it)."""
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None

    # --- results table (reconfigures columns per last operation) ------------

    def _show_loaded_csv(self, items):
        """Preview the just-loaded retrieved CSV (3 cols, no ground LST yet).

        Gives immediate visual feedback that the file parsed correctly before
        the user clicks Validate; Validate later switches to the enriched
        7-column paired view.
        """
        self._last_mode = "loaded"
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Site", "Time", "Retrieved (K)"])
        self.table.setRowCount(len(items))
        for r, it in enumerate(items):
            cells = [
                it.site.name,
                it.overpass_time.strftime("%Y%m%d%H%M"),
                f"{it.lst_k:.3f}" if it.lst_k == it.lst_k else "nan",
            ]
            for c, text in enumerate(cells):
                self.table.setItem(r, c, QTableWidgetItem(text))
        self._apply_column_widths(self._last_mode)

    def _show_ground_results(self, ground_list):
        """Left-Compute result: 5 columns of ground-truth LST only."""
        self._ground = list(ground_list)
        self._last_mode = "ground"
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Time", "Site", "LST (K)", "Emiss", "QC"])
        self.table.setRowCount(len(self._ground))
        for r, g in enumerate(self._ground):
            cells = [
                g.overpass_time.strftime("%Y%m%d%H%M"),
                g.site.name,
                f"{g.lst_k:.3f}" if g.lst_k == g.lst_k else "nan",
                f"{g.emissivity:.3f}" if g.emissivity == g.emissivity else "nan",
                g.qc_flag,
            ]
            for c, text in enumerate(cells):
                self.table.setItem(r, c, QTableWidgetItem(text))
        self._apply_column_widths(self._last_mode)

    def _show_validation_results(self, result):
        """Right-Validate result: 7 columns — paired retrieved/ground LST + emissivity."""
        self._validation_pairs = list(result.pairs)
        self._last_mode = "validation"
        self._unmatched_summary = (
            f"unmatched ground: {len(result.unmatched_ground)}, "
            f"unmatched retrieved: {len(result.unmatched_retrieved)}"
        )
        rows = list(result.pairs)
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["Site", "Time", "Retrieved (K)", "Ground (K)", "Diff (K)", "Emiss", "QC"]
        )
        self.table.setRowCount(len(rows))
        for r, p in enumerate(rows):
            cells = [
                p.ground.site.name,
                p.ground.overpass_time.strftime("%Y%m%d%H%M"),
                f"{p.retrieved.lst_k:.3f}" if p.retrieved.lst_k == p.retrieved.lst_k else "nan",
                f"{p.ground.lst_k:.3f}" if p.ground.lst_k == p.ground.lst_k else "nan",
                f"{p.diff:+.3f}" if p.diff == p.diff else "nan",
                f"{p.ground.emissivity:.3f}" if p.ground.emissivity == p.ground.emissivity else "nan",
                p.ground.qc_flag,
            ]
            for c, text in enumerate(cells):
                self.table.setItem(r, c, QTableWidgetItem(text))
        self._apply_column_widths(self._last_mode)

    def _apply_column_widths(self, mode):
        """Set per-column widths for the current table mode; stretch the last.

        Site is kept narrow, Time wide; the trailing column absorbs spare width
        so the table fills the panel without squashing the value columns.
        """
        widths = _TABLE_WIDTHS.get(mode)
        if not widths:
            return
        header = self.table.horizontalHeader()
        for i, w in enumerate(widths):
            self.table.setColumnWidth(i, w)
        header.setSectionResizeMode(len(widths) - 1, QHeaderView.Stretch)

    def _clear_ground(self):
        self._ground = []
        self._validation_pairs = []
        self._last_mode = None
        self._unmatched_summary = ""
        self.table.setRowCount(0)
        self.stats_label.setText("")
        self.fig.clear()
        self.canvas.draw()

    def _export(self):
        if self._last_mode == "validation" and self._validation_pairs:
            self._export_validation()
        elif self._last_mode == "ground" and self._ground:
            self._export_ground()
        else:
            QMessageBox.information(self, "注意", "没有可导出的结果")

    def _export_ground(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export ground LST", "", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["overpass_time", "site", "lst_k", "emissivity", "qc"])
            for g in self._ground:
                w.writerow([
                    g.overpass_time.strftime("%Y%m%d%H%M"), g.site.name,
                    g.lst_k, g.emissivity, g.qc_flag,
                ])
        self.status.showMessage(f"Saved {path}", 5000)

    def _export_validation(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export validation pairs", "", "CSV (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["site", "overpass_time_utc", "lst_retrieved_k",
                        "lst_ground_k", "diff_k", "emissivity", "qc"])
            for p in self._validation_pairs:
                w.writerow([
                    p.ground.site.name,
                    p.ground.overpass_time.strftime("%Y%m%d%H%M"),
                    p.retrieved.lst_k,
                    p.ground.lst_k,
                    p.diff,
                    p.ground.emissivity,
                    p.ground.qc_flag,
                ])
        self.status.showMessage(f"Saved {path}", 5000)

    # --- validate -----------------------------------------------------------

    def _load_retrieved(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load retrieved LST CSV", "", "CSV (*.csv)")
        if not path:
            return
        missing = self._missing_required_columns(path)
        if missing:
            QMessageBox.warning(
                self, "格式错误",
                f"CSV 缺少必需列。需要: site, overpass_time_utc, lst_k。"
                f"缺少: {', '.join(sorted(missing))}。多余列没问题。",
            )
            return
        self._retr_path = path
        self.retr_label.setText(Path(path).name)
        self.validate_btn.setEnabled(True)
        # Show the loaded CSV immediately so the user can verify the format
        # before clicking Validate (3-column preview; no ground LST yet).
        try:
            items = TableRetrievedLST(path).items
        except Exception:
            items = []
        self._show_loaded_csv(items)
        self.status.showMessage(
            f"Loaded {len(items)} rows from {Path(path).name}. "
            "Click Validate to compute ground LST.",
            5000,
        )

    @staticmethod
    def _missing_required_columns(path):
        """Return the subset of required columns absent from the CSV header.

        Required: ``site``, ``overpass_time_utc``, ``lst_k``. Extra columns are
        fine — the returned set only contains the missing required ones.
        """
        required = {"site", "overpass_time_utc", "lst_k"}
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, [])
        except OSError:
            return set()
        present = {h.strip() for h in header}
        return required - present

    def _show_csv_help(self):
        """Explain the retrieved-LST CSV format and offer to save a template."""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Retrieved-LST CSV format")
        msg.setText(
            "Required columns (header row, any order; extra columns ignored):\n"
            "    site, overpass_time_utc, lst_k\n\n"
            "Example:\n"
            "    site,overpass_time_utc,lst_k\n"
            "    psu,201202121430,295.32\n"
            "    tbl,201202130230,288.71\n\n"
            "• site             — matches a site code in the Sites list\n"
            "• overpass_time_utc — YYYYMMDDHHMM, UTC\n"
            "• lst_k            — retrieved LST in Kelvin"
        )
        save = msg.addButton("Save template…", QMessageBox.AcceptRole)
        msg.addButton("Close", QMessageBox.RejectRole)
        msg.exec()
        if msg.clickedButton() is save:
            self._save_template_csv()

    def _save_template_csv(self):
        """Write a minimal retrieved-LST template the user can fill in."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save retrieved-LST template", "retrieved_template.csv", "CSV (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["site", "overpass_time_utc", "lst_k"])
            w.writerow(["psu", "201202121430", "295.32"])
            w.writerow(["tbl", "201202130230", "288.71"])
        self.status.showMessage(f"Saved template: {path}", 5000)

    def _validate(self):
        if self._worker is not None and self._worker.isRunning():
            self.status.showMessage("Busy — wait for the current job to finish", 3000)
            return
        if not self._retr_path:
            QMessageBox.warning(self, "注意", "请先载入反演 CSV")
            return
        try:
            retr = TableRetrievedLST(self._retr_path).items
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取反演 CSV 失败:\n{e}")
            return
        try:
            emiss = self._emissivity_source()
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "注意", f"发射率配置无效:\n{e}")
            return
        # Obvious progress feedback: disable + relabel the button and tell the
        # status bar how many rows are being processed, so the user knows the
        # (potentially slow, network-bound) computation has started.
        self.validate_btn.setText("Validating…")
        self.validate_btn.setEnabled(False)
        self.compute_btn.setEnabled(False)
        self.status.showMessage(
            f"Validating — computing ground LST for {len(retr)} sites…"
        )
        self._worker = _ValidateWorker(retr, emiss, self.data_folder)
        self._worker.finished.connect(self._on_validate_done)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_validate_done(self, result):
        s = result.stats
        self._show_validation_results(result)
        self.stats_label.setText(
            f"n={s.n}   bias={s.bias:+.3f} K   RMSE={s.rmse:.3f} K   R={s.r:.3f}"
            f"   ({self._unmatched_summary})"
        )
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        if result.pairs:
            x = [p.ground.lst_k for p in result.pairs]
            y = [p.retrieved.lst_k for p in result.pairs]
            ax.scatter(x, y)
            lo, hi = min(min(x), min(y)), max(max(x), max(y))
            ax.plot([lo, hi], [lo, hi], "r--", label="1:1")
            ax.legend()
        ax.set_xlabel("Ground LST (K)")
        ax.set_ylabel("Retrieved LST (K)")
        ax.set_title("Retrieved vs Ground")
        self.fig.tight_layout()
        self.canvas.draw()
        self.status.showMessage(
            f"Validated {s.n} pairs (bias={s.bias:+.3f}, RMSE={s.rmse:.3f}, R={s.r:.3f})",
            5000,
        )
        self._finish_worker()
        self.validate_btn.setText("Validate")
        self.validate_btn.setEnabled(True)
        self.compute_btn.setEnabled(True)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    win = LSTNetWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
