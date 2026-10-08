import sys
import os
import json
import re

# Qt on Wayland prints harmless "qt.qpa.wayland.textinput ... Got leave event"
# warnings whenever keyboard focus moves between widgets/dialogs. Silence that
# one category (set before Qt is imported; a user-defined QT_LOGGING_RULES wins).
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.wayland.textinput=false")

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QFormLayout,
    QListWidget,
    QStackedWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QFrame,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QFileDialog,
    QScrollArea,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QInputDialog,
    QDialog
)
from PySide6.QtGui import QPixmap, QPalette, QAction, QColor
from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject, QProcess

import ligand_prep
import system_generation
import pdb_preprocessing
import analysis
import residue_utils


def hline():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


def section_header(text):
    """A styled section title (see QLabel#SectionHeader in the app stylesheet).
    Replaces the old 'plain QLabel + hline()' pattern with something that reads
    as a heading and carries its own separator rule."""
    lbl = QLabel(text)
    lbl.setObjectName("SectionHeader")
    return lbl


def hint_label(text):
    """Small, dimmed helper text shown under a field or button."""
    lbl = QLabel(text)
    lbl.setObjectName("Hint")
    lbl.setWordWrap(True)
    return lbl


def mark_primary(button):
    """Tag a button as the primary action on its page so the stylesheet gives
    it the accent treatment."""
    button.setProperty("primary", True)
    return button


def file_picker_row(placeholder="No file selected", dialog_filter="All Files (*)"):
    """Returns (row_widget, line_edit) for a browse-a-file row."""
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)

    line_edit = QLineEdit()
    line_edit.setPlaceholderText(placeholder)

    browse_btn = QPushButton("Browse...")

    def browse():
        path, _ = QFileDialog.getOpenFileName(row, "Select file", "", dialog_filter)
        if path:
            line_edit.setText(path)

    browse_btn.clicked.connect(browse)

    row_layout.addWidget(line_edit)
    row_layout.addWidget(browse_btn)
    return row, line_edit


def folder_picker_row(placeholder="Default: same folder as protein PDB"):
    """Returns (row_widget, line_edit) for a browse-a-folder row."""
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)

    line_edit = QLineEdit()
    line_edit.setPlaceholderText(placeholder)

    browse_btn = QPushButton("Browse...")

    def browse():
        path = QFileDialog.getExistingDirectory(row, "Select working directory")
        if path:
            line_edit.setText(path)

    browse_btn.clicked.connect(browse)

    row_layout.addWidget(line_edit)
    row_layout.addWidget(browse_btn)
    return row, line_edit


def wrap_scrollable(page):
    """Wraps a page widget in a QScrollArea so long option lists don't get clipped."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(page)
    return scroll


class AnalysisWorker(QObject):
    """Runs analysis.run_rest2_analysis(...) on a background QThread so the
    GUI stays responsive while plots/CSVs are being generated."""

    log_line = Signal(str)
    finished = Signal(dict)  # emits the result dict on success
    error = Signal(str)  # emits error message on failure

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(self):
        try:
            result = analysis.run_rest2_analysis(
                log_callback=self.log_line.emit, **self.kwargs
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the GUI
            import traceback

            self.log_line.emit(
                "\nTRACEBACK:\n" + traceback.format_exc()
            )
            self.error.emit(f"{exc}\n\nFull traceback printed in the log above.")


def _system_generation_task(tleap_kwargs, build_kwargs):
    """Runs system_generation.run_tleap() then build_openmm_system() as a
    single background task (used by MainWindow.run_system_generation via
    FunctionWorker). Kept at module level so it's picklable-safe for the
    thread and easy to reuse."""
    tleap_result = system_generation.run_tleap(**tleap_kwargs)
    try:
        sys_result = system_generation.build_openmm_system(
            tleap_result["prmtop"], tleap_result["inpcrd"], **build_kwargs
        )
    except Exception as exc:
        raise RuntimeError(
            f"tleap succeeded (files at {tleap_result['prmtop']}) but OpenMM "
            f"system creation failed:\n{exc}"
        ) from exc
    return {"tleap": tleap_result, "system": sys_result}


class FunctionWorker(QObject):
    """Generic background-task worker: runs any func(**kwargs) on a QThread
    and reports back via finished(prefix, result)/error(prefix, message).
    Used for the shorter steps (PDB Preprocessing, Ligand Prep, System
    Generation) that don't need live log streaming the way AnalysisWorker does.

    The task's `prefix` travels with the signal so MainWindow can handle every
    task with one pair of real slots - see MainWindow._run_task for why plain
    closures can't be used here."""

    finished = Signal(str, object)
    error = Signal(str, str)

    def __init__(self, prefix, func, kwargs):
        super().__init__()
        self.prefix = prefix
        self.func = func
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(**self.kwargs)
            self.finished.emit(self.prefix, result)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the GUI
            self.error.emit(self.prefix, str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Per-task widgets/callbacks for _run_task, keyed by task prefix.
        self._task_contexts = {}

        self.setWindowTitle("PyREST2")
        self._size_window_to_screen()

        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)

        # --- Sidebar Container ---
        # We use a QFrame and style it to look like the unified dark panel
        sidebar_container = QFrame()
        sidebar_container.setObjectName("SidebarFrame")
        sidebar_container.setStyleSheet("""
            QFrame#SidebarFrame {
                background-color: #202020;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
            }
        """)

        sidebar_layout = QVBoxLayout(sidebar_container)
        # Add 15px of padding to the bottom so the image isn't touching the border
        sidebar_layout.setContentsMargins(0, 0, 0, 15)
        sidebar_layout.setSpacing(0)

        # 1. Sidebar ListWidget setup - a numbered "wizard" step list. Base
        # titles are kept separately so we can re-render each row as
        # "1  Title" / "> 2  Title" (current) / "check  Title" (done) /
        # "-  Title  (skipped)" without losing the original label.
        self._step_titles = [
            "PDB Preprocessing",
            "Ligand Prep",
            "System Generation",
            "Output format",
            "Simulation run",
            "Analysis",
        ]
        self._step_done = [False] * len(self._step_titles)

        self.sidebar = QListWidget()
        self.sidebar.setObjectName("Sidebar")
        for i, title in enumerate(self._step_titles):
            self.sidebar.addItem(f"{i + 1}   {title}")

        # Wrap long labels onto a second line instead of truncating them with
        # an ellipsis, and give items enough horizontal room to actually show
        # their full text (a fixed stretch ratio alone isn't a reliable
        # minimum width once other widgets compete for space).
        self.sidebar.setWordWrap(True)
        sidebar_container.setMinimumWidth(210)

        # 2. Image Label setup
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)  # Center the image horizontally

        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "i.png")
        pixmap = QPixmap(image_path)

        # Scale the image to fit nicely within the sidebar
        if not pixmap.isNull():
            self.image_label.setPixmap(
                pixmap.scaled(
                    460,
                    460,
                    Qt.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self.image_label.setText("[i.png not found]")
            self.image_label.setStyleSheet("color: gray;")

        # --- Add to layout ---
        # List goes first (top), Image goes second (bottom)
        sidebar_layout.addWidget(self.sidebar)
        sidebar_layout.addWidget(self.image_label)

        # --- Pages Container ---
        self.pages = QStackedWidget()

        # Create pages (each wrapped in a scroll area since option lists are long)
        self.pages.addWidget(wrap_scrollable(self.create_pdb_preprocessing_page()))
        self._ligand_prep_page = self.create_ligand_prep_page()
        self.pages.addWidget(wrap_scrollable(self._ligand_prep_page))
        self.pages.addWidget(wrap_scrollable(self.create_system_generation_page()))
        self.pages.addWidget(wrap_scrollable(self.create_output_format_page()))
        self.pages.addWidget(wrap_scrollable(self.create_simulation_run_page()))
        self.pages.addWidget(wrap_scrollable(self.create_analysis_page()))

        # Change page when sidebar selection changes
        self.sidebar.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.sidebar.currentRowChanged.connect(self._on_page_changed)

        # Apply the initial "system type" (protein-ligand vs protein-only) state
        # now that every page exists, then render the step list.
        self._on_system_type_changed()

        self.sidebar.setCurrentRow(0)

        # Add the widgets to the main layout
        layout.addWidget(sidebar_container, 2)
        layout.addWidget(self.pages, 5)

        self._setup_menu_bar()

    LIGAND_PREP_STEP = 1  # sidebar index of the "Ligand Prep" step

    def is_protein_only(self):
        """True when the user has set System type to 'Protein only' - drives
        skipping Ligand Prep and dropping every ligand-specific field."""
        return (
            getattr(self, "system_type_combo", None) is not None
            and self.system_type_combo.currentText() == "Protein only"
        )

    def _render_sidebar_labels(self):
        """Re-paint every step row: number / done-tick / skipped marker."""
        protein_only = self.is_protein_only()
        for i, title in enumerate(self._step_titles):
            if i == self.LIGAND_PREP_STEP and protein_only:
                self.sidebar.item(i).setText(f"-   {title}   (skipped)")
            elif self._step_done[i]:
                self.sidebar.item(i).setText(f"✓   {title}")
            else:
                self.sidebar.item(i).setText(f"{i + 1}   {title}")

    def _mark_step_done(self, index):
        if 0 <= index < len(self._step_done):
            self._step_done[index] = True
            self._render_sidebar_labels()

    def _on_system_type_changed(self, *_):
        """Enable/disable everything tied to having a ligand when the user
        toggles System type on the first page."""
        protein_only = self.is_protein_only()

        # Ligand Prep step: disable the sidebar row and grey the whole page.
        lig_item = self.sidebar.item(self.LIGAND_PREP_STEP)
        if protein_only:
            lig_item.setFlags(lig_item.flags() & ~Qt.ItemIsEnabled)
        else:
            lig_item.setFlags(lig_item.flags() | Qt.ItemIsEnabled)
        if getattr(self, "_ligand_prep_page", None) is not None:
            self._ligand_prep_page.setEnabled(not protein_only)
        if getattr(self, "ligand_prep_disabled_note", None) is not None:
            self.ligand_prep_disabled_note.setVisible(protein_only)

        # System Generation: the ligand force field only matters with a ligand.
        if getattr(self, "ligand_ff_combo_sysgen", None) is not None:
            self.ligand_ff_combo_sysgen.setEnabled(not protein_only)

        # Simulation Run / Analysis: hide the ligand residue-name selectors.
        for widget_name in (
            "remd_ligand_resname_row",
            "remd_ligand_resname_label",
            "analysis_ligand_sel_row",
            "analysis_ligand_sel_label",
        ):
            w = getattr(self, widget_name, None)
            if w is not None:
                w.setVisible(not protein_only)

        if protein_only:
            # Forget any ligand files picked up from a previous protein-ligand
            # session so System Generation genuinely builds a protein-only box.
            self.ligand_mol2_path = None
            self.ligand_frcmod_path = None

        self._render_sidebar_labels()

    def _on_page_changed(self, row):
        """Light gating: keep the 'run' button on a page disabled until its
        prerequisites exist, with a one-line hint explaining why."""
        # System Generation (row 2): needs a protein PDB from somewhere.
        if row == 2 and hasattr(self, "sysgen_run_button"):
            has_pdb = bool(
                getattr(self, "fixed_pdb_path", None)
                or self.pdb_input_path.text().strip()
                or self.sysgen_protein_pdb_path.text().strip()
            )
            self.sysgen_run_button.setEnabled(has_pdb)
            if hasattr(self, "sysgen_hint"):
                self.sysgen_hint.setText(
                    ""
                    if has_pdb
                    else "Select or process a protein PDB first (step 1)."
                )
        # Simulation Run (row 4): needs a generated system.
        if row == 4 and hasattr(self, "remd_start_button"):
            has_system = bool(getattr(self, "system_prmtop_path", None))
            self.remd_start_button.setEnabled(has_system)
            if hasattr(self, "remd_hint"):
                self.remd_hint.setText(
                    ""
                    if has_system
                    else "Generate or load a system first (step 3)."
                )

    def _size_window_to_screen(self):
        """Sizes the window relative to the actual available screen space
        (85%, capped at a sensible max) and centers it - a fixed pixel size
        looks cramped on large monitors and can overflow small ones."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(1600, int(available.width() * 0.85))
            height = min(1000, int(available.height() * 0.85))
            self.resize(width, height)
            x = available.x() + (available.width() - width) // 2
            y = available.y() + (available.height() - height) // 2
            self.move(x, y)
        else:
            self.resize(1400, 900)

    def _setup_menu_bar(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")

        save_action = QAction("&Save Settings...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_settings)
        file_menu.addAction(save_action)

        load_action = QAction("&Load Settings...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_settings)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    # Settings save/load - auto-discovers every QLineEdit/QComboBox/
    # QCheckBox/QSpinBox/QDoubleSpinBox attribute on self, so new fields
    # added later are picked up automatically without touching this code.
    def _advance_to_next_page(self):
        """Marks the current step done and moves to the next *enabled* step
        (so a skipped Ligand Prep in protein-only mode is stepped over)."""
        current = self.sidebar.currentRow()
        self._mark_step_done(current)
        next_row = current + 1
        while next_row < self.sidebar.count():
            if self.sidebar.item(next_row).flags() & Qt.ItemIsEnabled:
                self.sidebar.setCurrentRow(next_row)
                return
            next_row += 1

    def _run_task(
        self,
        prefix,
        func,
        kwargs,
        button,
        progress_bar,
        status_label,
        busy_message,
        title,
        on_success,
    ):
        """
        Runs func(**kwargs) on a background QThread so the GUI doesn't freeze,
        showing an indeterminate progress bar while it runs. `prefix` is used
        to build unique instance-attribute names (self._<prefix>_thread /
        self._<prefix>_worker) so the thread/worker objects stay alive for the
        duration of the run instead of being garbage-collected mid-flight.
        """
        status_label.setText(busy_message)
        button.setEnabled(False)
        progress_bar.setVisible(True)
        progress_bar.setRange(
            0, 0
        )  # indeterminate/busy mode - these tools don't report % progress

        thread = QThread()
        worker = FunctionWorker(prefix, func, kwargs)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        # Everything the completion slots need, keyed by prefix.
        self._task_contexts[prefix] = {
            "button": button,
            "progress_bar": progress_bar,
            "status_label": status_label,
            "title": title,
            "on_success": on_success,
            "thread": thread,
        }

        # Must be bound methods of this QObject so Qt queues them onto the GUI
        # thread. A plain closure has no receiver QObject, so AutoConnection
        # degrades to a direct call in the worker thread, where touching
        # widgets raises "QObject::setParent: Cannot set parent, new parent is
        # in a different thread" and usually crashes.
        worker.finished.connect(self._on_task_success)
        worker.error.connect(self._on_task_error)

        setattr(self, f"_{prefix}_thread", thread)
        setattr(self, f"_{prefix}_worker", worker)
        thread.start()

    def _finish_task(self, prefix):
        """Common teardown for a background task; returns its context."""
        context = self._task_contexts.pop(prefix, None)
        if context is None:
            return None
        context["progress_bar"].setVisible(False)
        context["button"].setEnabled(True)
        context["thread"].quit()
        context["thread"].wait(5000)
        return context

    @Slot(str, object)
    def _on_task_success(self, prefix, result):
        context = self._finish_task(prefix)
        if context is not None:
            context["on_success"](result)

    @Slot(str, str)
    def _on_task_error(self, prefix, message):
        context = self._finish_task(prefix)
        if context is None:
            return
        context["status_label"].setText("")

        # AmberTools failures come back as a multi-section report (stdout,
        # stderr, leap.log / sqm.out). Show the headline in the dialog and put
        # the rest behind "Show Details", which is scrollable and copyable.
        headline, _, details = message.partition("\n\n")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle(context["title"])
        box.setText(headline.strip() or "The step failed.")
        if details.strip():
            box.setDetailedText(details.strip())
        box.exec()

    def _detect_and_apply_ligand_resname(
        self, topology_path, target_widget, as_selection
    ):
        """Scans topology_path for hetero/non-standard residues and fills
        target_widget with the result - auto-fills if there's exactly one
        candidate, otherwise asks the user to pick from what was found."""
        if not topology_path:
            QMessageBox.warning(
                self,
                "Detect Ligand",
                "No topology available yet - run System Generation first, "
                "or select a topology file, or just type the residue name manually.",
            )
            return
        if not os.path.isfile(topology_path):
            QMessageBox.warning(
                self, "Detect Ligand", f"File not found:\n{topology_path}"
            )
            return

        try:
            candidates = residue_utils.detect_hetero_resnames(topology_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Detect Ligand", f"Could not scan topology:\n{exc}"
            )
            return

        if not candidates:
            QMessageBox.information(
                self,
                "Detect Ligand",
                "No non-standard (hetero) residues found - this looks like a "
                "protein-only system, or your ligand happens to share a name "
                "with a standard residue/ion this scan excludes.",
            )
            return

        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            chosen, ok = QInputDialog.getItem(
                self,
                "Multiple Candidates Found",
                "More than one non-standard residue was found. Which is the ligand?\n"
                "(The others might be cofactors, modified residues, or crystallization additives.)",
                candidates,
                0,
                False,
            )
            if not ok:
                return

        target_widget.setText(f"resname {chosen}" if as_selection else chosen)

    def detect_ligand_resnames_simulation(self):
        topology_path = getattr(self, "system_prmtop_path", None)
        self._detect_and_apply_ligand_resname(
            topology_path, self.remd_ligand_resnames_edit, as_selection=False
        )

    def detect_ligand_resnames_analysis(self):
        topology_path = self.analysis_topology_path.text().strip() or getattr(
            self, "system_prmtop_path", None
        )
        self._detect_and_apply_ligand_resname(
            topology_path, self.analysis_ligand_sel_edit, as_selection=True
        )

    def collect_settings(self):
        settings = {}
        for name, widget in vars(self).items():
            if isinstance(widget, QDoubleSpinBox):
                settings[name] = {"type": "QDoubleSpinBox", "value": widget.value()}
            elif isinstance(widget, QSpinBox):
                settings[name] = {"type": "QSpinBox", "value": widget.value()}
            elif isinstance(widget, QComboBox):
                settings[name] = {"type": "QComboBox", "value": widget.currentText()}
            elif isinstance(widget, QCheckBox):
                settings[name] = {"type": "QCheckBox", "value": widget.isChecked()}
            elif isinstance(widget, QLineEdit):
                settings[name] = {"type": "QLineEdit", "value": widget.text()}
        return settings

    def apply_settings(self, settings):
        applied, skipped = 0, 0
        for name, entry in settings.items():
            widget = getattr(self, name, None)
            if widget is None:
                skipped += 1
                continue
            wtype = entry.get("type")
            value = entry.get("value")
            try:
                if wtype == "QDoubleSpinBox" and isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(value))
                elif wtype == "QSpinBox" and isinstance(widget, QSpinBox):
                    widget.setValue(int(value))
                elif wtype == "QComboBox" and isinstance(widget, QComboBox):
                    idx = widget.findText(str(value))
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                elif wtype == "QCheckBox" and isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif wtype == "QLineEdit" and isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                else:
                    skipped += 1
                    continue
                applied += 1
            except (TypeError, ValueError):
                skipped += 1
        return applied, skipped

    def save_settings(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", "remd_settings.json", "JSON Files (*.json)"
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            with open(path, "w") as f:
                json.dump(self.collect_settings(), f, indent=2)
        except OSError as exc:
            QMessageBox.critical(
                self, "Save Settings", f"Could not save settings:\n{exc}"
            )
            return
        QMessageBox.information(self, "Save Settings", f"Settings saved to:\n{path}")

    def load_settings(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(
                self, "Load Settings", f"Could not read settings file:\n{exc}"
            )
            return

        applied, skipped = self.apply_settings(data)
        message = f"Loaded {applied} setting(s) from:\n{path}"
        if skipped:
            message += (
                f"\n({skipped} entr(y/ies) skipped - not recognized in this version)"
            )
        QMessageBox.information(self, "Load Settings", message)

    def show_about(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("About PyREST2")
        dialog.resize(500, 600)
        layout = QVBoxLayout(dialog)

        photo = QLabel()
        # Resolve relative to this file, not the process CWD - the app can be
        # launched from anywhere (desktop icon, file manager, another folder).
        photo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v.png")
        pixmap = QPixmap(photo_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                450, 550, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            photo.setPixmap(pixmap)
        photo.setAlignment(Qt.AlignCenter)

        layout.addWidget(photo)

        text = QLabel(
            "<h3>REMD Setup</h3>"
            "<p>This software was developed as part of PhD research at "
            "<b>ViStA lab, BITS Pilani – K K Birla Goa Campus</b>, for preparing, "
            "running, and analyzing <b>protein–ligand Replica Exchange Solute "
            "Tempering (REST2/REMD)</b> simulations. It provides an integrated "
            "workflow covering PDB preprocessing, ligand parameterization, system "
            "generation using <b>tleap</b>, simulation setup and configuration, "
            "REST2/REMD production runs and trajectory analysis.</p>"
            "<p>Built with <b>PySide6</b>, <b>PDBFixer</b>, <b>AmberTools</b> "
            "(antechamber, parmchk2 and tleap), <b>OpenMM</b>, <b>ParmEd</b> "
            "and <b>openmmtools</b>.</p>"
            "<p>Use <b>File → Save Settings</b> / <b>Load Settings</b> to "
            "reuse a configuration across sessions.</p>",

        )
        text.setWordWrap(True)
        layout.addWidget(text)
        button = QPushButton("Close")
        button.clicked.connect(dialog.accept)
        layout.addWidget(button)

        dialog.exec()



    # Page 1: PDB Preprocessing
    def create_pdb_preprocessing_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(section_header("Workflow"))

        type_form = QFormLayout()
        self.system_type_combo = QComboBox()
        self.system_type_combo.addItems(["Protein–ligand complex", "Protein only"])
        self.system_type_combo.setToolTip(
            "Protein–ligand complex: the full 6-step pipeline, including ligand "
            "parameterisation.\n"
            "Protein only: skips Ligand Prep and drops every ligand-specific "
            "field. REST2 still heats the whole protein (the solute), so this "
            "runs a protein-only solute-tempering simulation."
        )
        self.system_type_combo.currentTextChanged.connect(self._on_system_type_changed)
        type_form.addRow("System type:", self.system_type_combo)
        layout.addLayout(type_form)
        layout.addWidget(
            hint_label(
                "Choose 'Protein only' for apo / ligand-free REST2 runs - "
                "step 2 (Ligand Prep) will be skipped automatically."
            )
        )

        layout.addWidget(section_header("Input Structure"))

        row, self.pdb_input_path = file_picker_row(
            "Select input PDB file...", "PDB Files (*.pdb)"
        )
        layout.addWidget(row)

        layout.addWidget(section_header("Cleanup Options"))

        form = QFormLayout()

        self.remove_waters_cb = QCheckBox("Remove crystallographic waters")
        self.remove_waters_cb.setChecked(True)
        form.addRow(self.remove_waters_cb)

        self.remove_hetatm_cb = QCheckBox("Remove heteroatoms / non-standard residues")
        self.remove_hetatm_cb.setChecked(True)
        form.addRow(self.remove_hetatm_cb)

        self.keep_alt_loc_combo = QComboBox()
        self.keep_alt_loc_combo.addItems(["Highest occupancy", "A", "B", "C"])
        form.addRow("Alternate locations:", self.keep_alt_loc_combo)

        self.chain_select_edit = QLineEdit()
        self.chain_select_edit.setPlaceholderText(
            "e.g. A  (leave blank for all chains)"
        )
        form.addRow("Chain selection:", self.chain_select_edit)

        self.add_missing_res_cb = QCheckBox("Model missing residues/loops")
        form.addRow(self.add_missing_res_cb)

        self.add_hydrogens_cb = QCheckBox("Add missing hydrogens")
        self.add_hydrogens_cb.setChecked(True)
        form.addRow(self.add_hydrogens_cb)

        self.ph_spin = QDoubleSpinBox()
        self.ph_spin.setRange(0.0, 14.0)
        self.ph_spin.setSingleStep(0.1)
        self.ph_spin.setValue(7.0)
        form.addRow("Protonation pH:", self.ph_spin)

        self.disulfide_cb = QCheckBox("Auto-detect disulfide bonds")
        self.disulfide_cb.setChecked(True)
        form.addRow(self.disulfide_cb)

        layout.addLayout(form)
        layout.addStretch()

        self.pdb_status_label = QLabel("")
        self.pdb_status_label.setWordWrap(True)
        layout.addWidget(self.pdb_status_label)

        self.pdb_progress_bar = QProgressBar()
        self.pdb_progress_bar.setVisible(False)
        layout.addWidget(self.pdb_progress_bar)

        self.pdb_run_button = mark_primary(QPushButton("Run PDBFixer"))
        self.pdb_run_button.clicked.connect(self.run_pdb_preprocessing)
        layout.addWidget(self.pdb_run_button)

        layout.addWidget(section_header("Already Have a Processed PDB?"))
        layout.addWidget(
            QLabel(
                "If your PDB is already cleaned/protonated, skip PDBFixer and use it as-is."
            )
        )

        preprocessed_row, self.preprocessed_pdb_path = file_picker_row(
            "Select an already-processed PDB file...", "PDB Files (*.pdb)"
        )
        layout.addWidget(preprocessed_row)

        self.use_preprocessed_button = QPushButton(
            "Use This File Directly (skip PDBFixer)"
        )
        self.use_preprocessed_button.clicked.connect(self.use_preprocessed_pdb)
        layout.addWidget(self.use_preprocessed_button)

        return page

    def use_preprocessed_pdb(self):
        """Lets the user skip PDBFixer entirely and use an already-processed
        PDB directly - System Generation will pick this up the same way it
        picks up a PDBFixer output."""
        path = self.preprocessed_pdb_path.text().strip()

        if not path:
            QMessageBox.warning(
                self, "PDB Preprocessing", "Please select a PDB file first."
            )
            return
        if not os.path.isfile(path):
            QMessageBox.warning(self, "PDB Preprocessing", f"File not found:\n{path}")
            return

        self.fixed_pdb_path = path
        self.pdb_status_label.setText(
            f"Using this PDB directly (PDBFixer skipped): {path}"
        )
        self._advance_to_next_page()

    def run_pdb_preprocessing(self):
        """Reads the PDB Preprocessing page widgets and runs PDBFixer via pdb_preprocessing.py
        on a background thread."""
        input_file = self.pdb_input_path.text().strip()

        if not input_file:
            QMessageBox.warning(
                self, "PDB Preprocessing", "Please select an input PDB file first."
            )
            return
        if not os.path.isfile(input_file):
            QMessageBox.warning(
                self, "PDB Preprocessing", f"File not found:\n{input_file}"
            )
            return

        if self.keep_alt_loc_combo.currentText() != "Highest occupancy":
            QMessageBox.information(
                self,
                "PDB Preprocessing",
                "Note: alternate-location selection isn't implemented yet "
                "(PDBFixer has no native altloc API) - proceeding without it.",
            )

        kwargs = dict(
            input_pdb=input_file,
            remove_waters=self.remove_waters_cb.isChecked(),
            remove_heterogens=self.remove_hetatm_cb.isChecked(),
            chain_select=self.chain_select_edit.text(),
            model_missing_residues=self.add_missing_res_cb.isChecked(),
            add_hydrogens=self.add_hydrogens_cb.isChecked(),
            ph=self.ph_spin.value(),
            detect_disulfides=self.disulfide_cb.isChecked(),
        )

        self._run_task(
            prefix="pdb",
            func=pdb_preprocessing.fix_pdb,
            kwargs=kwargs,
            button=self.pdb_run_button,
            progress_bar=self.pdb_progress_bar,
            status_label=self.pdb_status_label,
            busy_message="Running PDBFixer...",
            title="PDB Preprocessing",
            on_success=self._on_pdb_preprocessing_success,
        )

    def _on_pdb_preprocessing_success(self, result):
        # Remember this so later pages (System Generation) use the cleaned PDB automatically
        self.fixed_pdb_path = result["output_pdb"]

        message = (
            f"Done. Output: {result['output_pdb']}\n"
            f"Chains kept: {', '.join(result['chains_kept'])}\n"
            f"Missing residues found: {result['n_missing_residues']}\n"
            f"Missing atoms added: {result['n_missing_atoms_added']}"
        )
        if self.disulfide_cb.isChecked():
            if result["disulfide_pairs"]:
                pairs_str = "; ".join(
                    f"{c1}{r1}-{c2}{r2}"
                    for (c1, r1), (c2, r2) in result["disulfide_pairs"]
                )
                message += f"\nCandidate disulfides: {pairs_str}"
            else:
                message += "\nCandidate disulfides: none found"
        self.pdb_status_label.setText(message)
        self._advance_to_next_page()

    # Page 2: Ligand Prep
    def create_ligand_prep_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.ligand_prep_disabled_note = QLabel(
            "System type is set to “Protein only” on step 1 - this step "
            "is not needed and has been skipped. Switch it back to "
            "“Protein–ligand complex” to parameterise a ligand."
        )
        self.ligand_prep_disabled_note.setWordWrap(True)
        self.ligand_prep_disabled_note.setStyleSheet(
            "background-color: #3a2f00; color: #ffd479; border: 1px solid "
            "#6b5500; border-radius: 6px; padding: 8px;"
        )
        self.ligand_prep_disabled_note.setVisible(False)
        layout.addWidget(self.ligand_prep_disabled_note)

        layout.addWidget(section_header("Ligand File"))

        row, self.ligand_input_path = file_picker_row(
            "Select ligand file (.mol2 / .sdf / .pdb)...",
            "Ligand Files (*.mol2 *.sdf *.pdb)",
        )
        layout.addWidget(row)

        layout.addWidget(section_header("Parameterization Options"))

        form = QFormLayout()

        self.ligand_ff_combo = QComboBox()
        self.ligand_ff_combo.addItems(["gaff2", "gaff"])
        form.addRow("Ligand force field:", self.ligand_ff_combo)

        self.charge_method_combo = QComboBox()
        self.charge_method_combo.addItems(["AM1-BCC", "RESP", "Gasteiger"])
        form.addRow("Partial charge method:", self.charge_method_combo)

        self.net_charge_spin = QSpinBox()
        self.net_charge_spin.setRange(-10, 10)
        self.net_charge_spin.setValue(0)
        form.addRow("Net charge:", self.net_charge_spin)

        self.multiplicity_spin = QSpinBox()
        self.multiplicity_spin.setRange(1, 5)
        self.multiplicity_spin.setValue(1)
        form.addRow("Spin multiplicity:", self.multiplicity_spin)

        self.generate_frcmod_cb = QCheckBox("Generate frcmod (parmchk2)")
        self.generate_frcmod_cb.setChecked(True)
        form.addRow(self.generate_frcmod_cb)

        layout.addLayout(form)

        layout.addWidget(section_header("Execution"))

        form_exec = QFormLayout()

        self.ligand_conda_env_edit = QLineEdit()
        self.ligand_conda_env_edit.setPlaceholderText(
            "e.g. amber-env  (leave blank to run antechamber/parmchk2 directly, no conda)"
        )
        form_exec.addRow(
            "Conda environment name (optional):", self.ligand_conda_env_edit
        )

        ligand_conda_env_path_row, self.ligand_conda_env_path_edit = folder_picker_row(
            "e.g. /home/you/miniforge3/envs/amber-env  (recommended over name if you"
            " have more than one conda install)"
        )
        form_exec.addRow(
            "Conda environment path (recommended):", ligand_conda_env_path_row
        )

        ligand_conda_exe_row, self.ligand_conda_exe_path = file_picker_row(
            "Default: derived from environment path, else auto-detected from PATH",
            "All Files (*)",
        )
        form_exec.addRow("Conda executable (optional):", ligand_conda_exe_row)

        layout.addLayout(form_exec)
        layout.addStretch()

        self.ligand_status_label = QLabel("")
        self.ligand_status_label.setWordWrap(True)
        layout.addWidget(self.ligand_status_label)

        self.ligand_progress_bar = QProgressBar()
        self.ligand_progress_bar.setVisible(False)
        layout.addWidget(self.ligand_progress_bar)

        self.ligand_run_button = mark_primary(
            QPushButton("Run antechamber / parmchk2")
        )
        self.ligand_run_button.clicked.connect(self.run_ligand_prep)
        layout.addWidget(self.ligand_run_button)

        layout.addWidget(section_header("Already Have Prepared Ligand Files?"))
        layout.addWidget(
            QLabel(
                "If you already have a parameterized ligand.mol2 (and optionally a "
                "matching .frcmod), skip antechamber/parmchk2 and use them as-is."
            )
        )

        mol2_row, self.preprocessed_ligand_mol2_path = file_picker_row(
            "Select an already-prepared ligand .mol2 file...", "Mol2 Files (*.mol2)"
        )
        layout.addWidget(mol2_row)

        frcmod_row, self.preprocessed_ligand_frcmod_path = file_picker_row(
            "Select a matching .frcmod file (optional)...", "Frcmod Files (*.frcmod)"
        )
        layout.addWidget(frcmod_row)

        self.use_preprocessed_ligand_button = QPushButton(
            "Use These Files Directly (skip antechamber/parmchk2)"
        )
        self.use_preprocessed_ligand_button.clicked.connect(
            self.use_preprocessed_ligand
        )
        layout.addWidget(self.use_preprocessed_ligand_button)

        return page

    def use_preprocessed_ligand(self):
        """Lets the user skip antechamber/parmchk2 entirely and use an
        already-prepared ligand.mol2 (+ optional .frcmod) directly - System
        Generation will pick these up the same way it picks up
        ligand_prep.py's own output."""
        mol2_path = self.preprocessed_ligand_mol2_path.text().strip()
        frcmod_path = self.preprocessed_ligand_frcmod_path.text().strip()

        if not mol2_path:
            QMessageBox.warning(
                self, "Ligand Prep", "Please select a ligand .mol2 file first."
            )
            return
        if not os.path.isfile(mol2_path):
            QMessageBox.warning(self, "Ligand Prep", f"File not found:\n{mol2_path}")
            return
        if frcmod_path and not os.path.isfile(frcmod_path):
            QMessageBox.warning(
                self, "Ligand Prep", f".frcmod file not found:\n{frcmod_path}"
            )
            return

        self.ligand_mol2_path = mol2_path
        self.ligand_frcmod_path = frcmod_path or None

        message = (
            f"Using this mol2 directly (antechamber/parmchk2 skipped): {mol2_path}"
        )
        if frcmod_path:
            message += f"\nfrcmod: {frcmod_path}"
        else:
            message += (
                "\nNo frcmod provided - System Generation will proceed without one."
            )
        self.ligand_status_label.setText(message)
        self._advance_to_next_page()

    def run_ligand_prep(self):
        """Reads the Ligand Prep page widgets and runs antechamber/parmchk2 via
        ligand_prep.py on a background thread."""
        input_file = self.ligand_input_path.text().strip()

        if not input_file:
            QMessageBox.warning(
                self, "Ligand Prep", "Please select a ligand file first."
            )
            return
        if not os.path.isfile(input_file):
            QMessageBox.warning(self, "Ligand Prep", f"File not found:\n{input_file}")
            return

        atom_type = self.ligand_ff_combo.currentText()
        if atom_type not in ligand_prep.ATOM_TYPE_MAP:
            QMessageBox.warning(
                self,
                "Ligand Prep",
                f"'{atom_type}' isn't supported by antechamber directly.\n"
                "Please choose 'gaff2' or 'gaff' for this step "
                "(OpenFF/SMIRNOFF uses a separate, non-antechamber workflow).",
            )
            return

        output_name = os.path.splitext(os.path.basename(input_file))[0]

        kwargs = dict(
            input_file=input_file,
            charge_method=self.charge_method_combo.currentText(),
            atom_type=atom_type,
            net_charge=self.net_charge_spin.value(),
            multiplicity=self.multiplicity_spin.value(),
            output_name=output_name,
            generate_frcmod=self.generate_frcmod_cb.isChecked(),
            conda_env=self.ligand_conda_env_edit.text().strip() or None,
            conda_env_path=self.ligand_conda_env_path_edit.text().strip() or None,
            conda_exe=self.ligand_conda_exe_path.text().strip() or None,
        )

        self._run_task(
            prefix="ligand",
            func=ligand_prep.prepare_ligand,
            kwargs=kwargs,
            button=self.ligand_run_button,
            progress_bar=self.ligand_progress_bar,
            status_label=self.ligand_status_label,
            busy_message="Running antechamber... this can take a minute.",
            title="Ligand Prep",
            on_success=self._on_ligand_prep_success,
        )

    def _on_ligand_prep_success(self, result):
        mol2_path, frcmod_path = result

        message = f"Done. mol2: {mol2_path}"
        if frcmod_path:
            message += f"\nfrcmod: {frcmod_path}"
        self.ligand_status_label.setText(message)

        # Remember these so the System Generation page can use them automatically
        self.ligand_mol2_path = mol2_path
        self.ligand_frcmod_path = frcmod_path
        self._advance_to_next_page()

    # Page 3: System Generation (tleap)
    def create_system_generation_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(section_header("Force Fields"))

        form = QFormLayout()

        self.protein_ff_combo = QComboBox()
        self.protein_ff_combo.addItems(
            [
                "leaprc.protein.ff19SB",
                "leaprc.protein.ff14SB",
                "leaprc.protein.ff99SBildn",
            ]
        )
        form.addRow("Protein force field:", self.protein_ff_combo)

        self.ligand_ff_combo_sysgen = QComboBox()
        self.ligand_ff_combo_sysgen.addItems(["leaprc.gaff2", "leaprc.gaff"])
        form.addRow("Ligand force field:", self.ligand_ff_combo_sysgen)

        self.water_model_combo = QComboBox()
        self.water_model_combo.addItems(
            [
                "leaprc.water.opc",
                "leaprc.water.tip3p",
                "leaprc.water.tip4pew",
                "leaprc.water.spce",
                "leaprc.water.opc3",
            ]
        )
        form.addRow("Water model:", self.water_model_combo)

        layout.addLayout(form)

        layout.addWidget(section_header("Solvation"))

        form2 = QFormLayout()

        self.box_shape_combo = QComboBox()
        self.box_shape_combo.addItems(
            ["Truncated octahedron (solvateoct)", "Cubic (solvatebox)"]
        )
        form2.addRow("Box shape:", self.box_shape_combo)

        self.box_padding_spin = QDoubleSpinBox()
        self.box_padding_spin.setRange(5.0, 30.0)
        self.box_padding_spin.setSingleStep(0.5)
        self.box_padding_spin.setValue(10.0)
        self.box_padding_spin.setSuffix(" \u00c5")
        form2.addRow("Box padding:", self.box_padding_spin)

        layout.addLayout(form2)

        layout.addWidget(section_header("Ions"))

        form3 = QFormLayout()

        self.neutralize_cb = QCheckBox(
            "Neutralize system (addions Na+/Cl- to 0 net charge)"
        )
        self.neutralize_cb.setChecked(True)
        form3.addRow(self.neutralize_cb)

        self.cation_combo = QComboBox()
        self.cation_combo.addItems(["Na+", "K+"])
        form3.addRow("Cation:", self.cation_combo)

        self.anion_combo = QComboBox()
        self.anion_combo.addItems(["Cl-"])
        form3.addRow("Anion:", self.anion_combo)

        self.ion_conc_spin = QDoubleSpinBox()
        self.ion_conc_spin.setRange(0.0, 1.0)
        self.ion_conc_spin.setSingleStep(0.01)
        self.ion_conc_spin.setValue(0.15)
        self.ion_conc_spin.setSuffix(" M")
        form3.addRow("Additional ionic strength (addionsrand):", self.ion_conc_spin)

        layout.addLayout(form3)

        layout.addWidget(section_header("Output"))

        form4 = QFormLayout()
        self.topology_prefix_edit = QLineEdit("complex")
        form4.addRow("Output file prefix:", self.topology_prefix_edit)
        layout.addLayout(form4)

        layout.addWidget(section_header("Execution"))

        form5 = QFormLayout()

        protein_row, self.sysgen_protein_pdb_path = file_picker_row(
            "Default: use the fixed PDB from 'PDB Preprocessing' (or its raw input)",
            "PDB Files (*.pdb)",
        )
        form5.addRow("Protein PDB:", protein_row)

        tleap_row, self.tleap_path_edit = file_picker_row(
            "Default: use 'tleap' on PATH", "All Files (*)"
        )
        form5.addRow("tleap executable (optional):", tleap_row)

        self.conda_env_edit = QLineEdit()
        self.conda_env_edit.setPlaceholderText(
            "e.g. amber-env  (leave blank to run tleap directly, no conda)"
        )
        form5.addRow("Conda environment name (optional):", self.conda_env_edit)

        conda_env_path_row, self.conda_env_path_edit = folder_picker_row(
            "e.g. /home/you/miniforge3/envs/amber-env  (recommended over name if you"
            " have more than one conda install)"
        )
        form5.addRow("Conda environment path (recommended):", conda_env_path_row)

        conda_exe_row, self.conda_exe_path = file_picker_row(
            "Default: derived from environment path, else auto-detected from PATH",
            "All Files (*)",
        )
        form5.addRow("Conda executable (optional):", conda_exe_row)

        workdir_row, self.sysgen_workdir_edit = folder_picker_row()
        form5.addRow("Working directory:", workdir_row)

        layout.addLayout(form5)

        layout.addStretch()

        self.sysgen_status_label = QLabel("")
        self.sysgen_status_label.setWordWrap(True)
        layout.addWidget(self.sysgen_status_label)

        self.sysgen_progress_bar = QProgressBar()
        self.sysgen_progress_bar.setVisible(False)
        layout.addWidget(self.sysgen_progress_bar)

        self.sysgen_hint = hint_label("")
        layout.addWidget(self.sysgen_hint)

        self.sysgen_run_button = mark_primary(QPushButton("Generate System (tleap)"))
        self.sysgen_run_button.clicked.connect(self.run_system_generation)
        layout.addWidget(self.sysgen_run_button)

        layout.addWidget(section_header("Already Have a Generated System?"))
        layout.addWidget(
            QLabel(
                "If you already have a complex.prmtop/complex.inpcrd (built elsewhere, "
                "or from a previous run), skip tleap and OpenMM system creation entirely."
            )
        )

        preprocessed_prmtop_row, self.preprocessed_prmtop_path = file_picker_row(
            "Select an already-generated .prmtop file...", "Topology Files (*.prmtop)"
        )
        layout.addWidget(preprocessed_prmtop_row)

        preprocessed_inpcrd_row, self.preprocessed_inpcrd_path = file_picker_row(
            "Select the matching .inpcrd file...", "Coordinate Files (*.inpcrd)"
        )
        layout.addWidget(preprocessed_inpcrd_row)

        self.use_preprocessed_system_button = QPushButton(
            "Use These Files Directly (skip tleap)"
        )
        self.use_preprocessed_system_button.clicked.connect(
            self.use_preprocessed_system
        )
        layout.addWidget(self.use_preprocessed_system_button)

        return page

    def use_preprocessed_system(self):
        """Lets the user skip tleap + OpenMM system creation entirely and use
        an already-generated prmtop/inpcrd directly - the Simulation Run page
        will pick these up the same way it picks up this page's own output."""
        prmtop_path = self.preprocessed_prmtop_path.text().strip()
        inpcrd_path = self.preprocessed_inpcrd_path.text().strip()

        if not prmtop_path or not inpcrd_path:
            QMessageBox.warning(
                self,
                "System Generation",
                "Please select both a .prmtop and a .inpcrd file.",
            )
            return
        if not os.path.isfile(prmtop_path):
            QMessageBox.warning(
                self, "System Generation", f"File not found:\n{prmtop_path}"
            )
            return
        if not os.path.isfile(inpcrd_path):
            QMessageBox.warning(
                self, "System Generation", f"File not found:\n{inpcrd_path}"
            )
            return

        self.system_prmtop_path = prmtop_path
        self.system_inpcrd_path = inpcrd_path

        self.sysgen_status_label.setText(
            f"Using this system directly (tleap skipped):\nprmtop: {prmtop_path}\ninpcrd: {inpcrd_path}"
        )
        self._advance_to_next_page()

    def run_system_generation(self):
        """Runs tleap (system_generation.run_tleap) then builds the OpenMM
        System (system_generation.build_openmm_system) on a background thread,
        using the settings from this page and the Output Format page."""

        protein_pdb = (
            self.sysgen_protein_pdb_path.text().strip()
            or getattr(self, "fixed_pdb_path", None)
            or self.pdb_input_path.text().strip()
        )
        if not protein_pdb:
            QMessageBox.warning(
                self,
                "System Generation",
                "Please select a protein PDB (either here or on the PDB Preprocessing page).",
            )
            return
        if not os.path.isfile(protein_pdb):
            QMessageBox.warning(
                self, "System Generation", f"Protein PDB not found:\n{protein_pdb}"
            )
            return

        # Ligand files: prefer whatever Ligand Prep just produced; otherwise
        # unset. In protein-only mode they're always unset so tleap builds a
        # protein-in-water box (system_generation.build_tleap_script handles
        # the no-ligand path).
        if self.is_protein_only():
            ligand_mol2 = None
            ligand_frcmod = None
        else:
            ligand_mol2 = getattr(self, "ligand_mol2_path", None)
            ligand_frcmod = getattr(self, "ligand_frcmod_path", None)

        workdir = self.sysgen_workdir_edit.text().strip() or None
        tleap_path = self.tleap_path_edit.text().strip() or None
        output_prefix = self.topology_prefix_edit.text().strip() or "complex"

        tleap_kwargs = dict(
            protein_pdb=protein_pdb,
            protein_ff=self.protein_ff_combo.currentText(),
            ligand_ff=self.ligand_ff_combo_sysgen.currentText(),
            water_model=self.water_model_combo.currentText(),
            ligand_mol2=ligand_mol2,
            ligand_frcmod=ligand_frcmod,
            box_shape=self.box_shape_combo.currentText(),
            box_padding=self.box_padding_spin.value(),
            neutralize=self.neutralize_cb.isChecked(),
            cation=self.cation_combo.currentText(),
            anion=self.anion_combo.currentText(),
            ion_conc=self.ion_conc_spin.value(),
            output_prefix=output_prefix,
            workdir=workdir,
            tleap_path=tleap_path,
            conda_env=self.conda_env_edit.text().strip() or None,
            conda_env_path=self.conda_env_path_edit.text().strip() or None,
            conda_exe=self.conda_exe_path.text().strip() or None,
        )

        build_kwargs = dict(
            nonbonded_method=self.nonbonded_method_combo.currentText(),
            nonbonded_cutoff=self.nonbonded_cutoff_spin.value(),
            constraints=self.constraints_combo.currentText(),
            rigid_water=self.rigid_water_cb.isChecked(),
            hmr_enabled=self.hmr_cb.isChecked(),
            hydrogen_mass=self.hydrogen_mass_spin.value(),
            ewald_error_tolerance=self.ewald_tol_spin.value(),
        )

        self._run_task(
            prefix="sysgen",
            func=_system_generation_task,
            kwargs={"tleap_kwargs": tleap_kwargs, "build_kwargs": build_kwargs},
            button=self.sysgen_run_button,
            progress_bar=self.sysgen_progress_bar,
            status_label=self.sysgen_status_label,
            busy_message="Running tleap, then building OpenMM system... this can take a while.",
            title="System Generation",
            on_success=self._on_system_generation_success,
        )

    def _on_system_generation_success(self, result):
        sys_result = result["system"]

        # Remember these for the Simulation Run / Analysis pages
        self.system_prmtop_path = sys_result["prmtop"]
        self.system_inpcrd_path = sys_result["inpcrd"]
        self.system_pdb_path = result["tleap"].get("pdb")

        tleap_result = result["tleap"]
        fixes = ""
        if tleap_result.get("pdb_fixes"):
            fixes = "\nPDB fixed for tleap: " + "; ".join(tleap_result["pdb_fixes"])
        salt = ""
        if tleap_result.get("n_ion_pairs"):
            salt = (
                f"\nSalt: {tleap_result['n_ion_pairs']} ion pairs for "
                f"{tleap_result['ion_conc']} M among "
                f"{tleap_result['n_waters']} waters"
            )
        self.sysgen_status_label.setText(
            f"System built successfully.\n"
            f"Atoms: {sys_result['natom']}   Residues: {sys_result['nres']}{fixes}{salt}\n"
            f"prmtop: {sys_result['prmtop']}\n"
            f"inpcrd: {sys_result['inpcrd']}"
        )
        self._advance_to_next_page()

    # Page 4: Output format (OpenMM system-creation settings)
    def create_output_format_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(section_header("Nonbonded Settings"))

        form = QFormLayout()

        self.nonbonded_method_combo = QComboBox()
        self.nonbonded_method_combo.addItems(
            ["PME", "CutoffPeriodic", "CutoffNonPeriodic", "NoCutoff", "Ewald"]
        )
        form.addRow("Nonbonded method:", self.nonbonded_method_combo)

        self.nonbonded_cutoff_spin = QDoubleSpinBox()
        self.nonbonded_cutoff_spin.setRange(0.6, 2.0)
        self.nonbonded_cutoff_spin.setSingleStep(0.1)
        self.nonbonded_cutoff_spin.setValue(1.0)
        self.nonbonded_cutoff_spin.setSuffix(" nm")
        form.addRow("Nonbonded cutoff:", self.nonbonded_cutoff_spin)

        self.ewald_tol_spin = QDoubleSpinBox()
        self.ewald_tol_spin.setDecimals(6)
        self.ewald_tol_spin.setRange(0.000001, 0.01)
        self.ewald_tol_spin.setSingleStep(0.0001)
        self.ewald_tol_spin.setValue(0.0005)
        form.addRow("Ewald error tolerance:", self.ewald_tol_spin)

        layout.addLayout(form)

        layout.addWidget(section_header("Constraints"))

        form2 = QFormLayout()

        self.constraints_combo = QComboBox()
        self.constraints_combo.addItems(["HBonds", "AllBonds", "HAngles", "None"])
        form2.addRow("Constraints:", self.constraints_combo)

        self.rigid_water_cb = QCheckBox("Rigid water")
        self.rigid_water_cb.setChecked(True)
        form2.addRow(self.rigid_water_cb)

        layout.addLayout(form2)

        layout.addWidget(section_header("Hydrogen Mass Repartitioning"))

        form3 = QFormLayout()

        self.hmr_cb = QCheckBox("Enable hydrogen mass repartitioning")
        self.hmr_cb.setChecked(True)
        form3.addRow(self.hmr_cb)

        self.hydrogen_mass_spin = QDoubleSpinBox()
        self.hydrogen_mass_spin.setRange(1.0, 4.0)
        self.hydrogen_mass_spin.setSingleStep(0.1)
        self.hydrogen_mass_spin.setValue(1.5)
        self.hydrogen_mass_spin.setSuffix(" amu")
        form3.addRow("Hydrogen mass:", self.hydrogen_mass_spin)

        layout.addLayout(form3)

        layout.addStretch()

        button = mark_primary(QPushButton("Next  →"))
        button.clicked.connect(self._advance_to_next_page)
        layout.addWidget(button)

        return page

    # Page 5: Simulation run (EM / NVT / NPT / REMD)
    def create_simulation_run_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # --- Energy Minimization ---
        em_group = QGroupBox("Energy Minimization")
        em_form = QFormLayout(em_group)

        self.em_max_iter_spin = QSpinBox()
        self.em_max_iter_spin.setRange(0, 1000000)
        self.em_max_iter_spin.setValue(5000)
        self.em_max_iter_spin.setSpecialValueText("Until converged (0)")
        em_form.addRow("Max iterations:", self.em_max_iter_spin)

        self.em_tolerance_spin = QDoubleSpinBox()
        self.em_tolerance_spin.setDecimals(3)
        self.em_tolerance_spin.setRange(0.001, 100.0)
        self.em_tolerance_spin.setValue(10.0)
        self.em_tolerance_spin.setSuffix(" kJ/mol/nm")
        em_form.addRow("Energy tolerance:", self.em_tolerance_spin)

        layout.addWidget(em_group)

        # --- NVT Equilibration ---
        nvt_group = QGroupBox("NVT Equilibration")
        nvt_form = QFormLayout(nvt_group)

        self.nvt_steps_spin = QSpinBox()
        self.nvt_steps_spin.setRange(0, 100000000)
        self.nvt_steps_spin.setValue(500000)
        self.nvt_steps_spin.setSuffix(" steps")
        nvt_form.addRow("Number of steps:", self.nvt_steps_spin)

        self.nvt_temp_spin = QDoubleSpinBox()
        self.nvt_temp_spin.setRange(0.0, 500.0)
        self.nvt_temp_spin.setValue(300.0)
        self.nvt_temp_spin.setSuffix(" K")
        nvt_form.addRow("Target temperature:", self.nvt_temp_spin)

        self.nvt_timestep_spin = QDoubleSpinBox()
        self.nvt_timestep_spin.setRange(0.5, 5.0)
        self.nvt_timestep_spin.setSingleStep(0.5)
        self.nvt_timestep_spin.setValue(2.0)
        self.nvt_timestep_spin.setSuffix(" fs")
        nvt_form.addRow("Timestep:", self.nvt_timestep_spin)

        self.nvt_thermostat_combo = QComboBox()
        self.nvt_thermostat_combo.addItems(
            ["Langevin middle integrator", "Nose-Hoover", "Andersen"]
        )
        nvt_form.addRow("Thermostat:", self.nvt_thermostat_combo)

        self.nvt_restraint_cb = QCheckBox("Apply positional restraints on heavy atoms")
        self.nvt_restraint_cb.setChecked(True)
        nvt_form.addRow(self.nvt_restraint_cb)

        layout.addWidget(nvt_group)

        # --- NPT Equilibration ---
        npt_group = QGroupBox("NPT Equilibration")
        npt_form = QFormLayout(npt_group)

        self.npt_steps_spin = QSpinBox()
        self.npt_steps_spin.setRange(0, 100000000)
        self.npt_steps_spin.setValue(500000)
        self.npt_steps_spin.setSuffix(" steps")
        npt_form.addRow("Number of steps:", self.npt_steps_spin)

        self.npt_pressure_spin = QDoubleSpinBox()
        self.npt_pressure_spin.setRange(0.5, 10.0)
        self.npt_pressure_spin.setSingleStep(0.1)
        self.npt_pressure_spin.setValue(1.0)
        self.npt_pressure_spin.setSuffix(" atm")  # simulation_run.py uses atmospheres
        npt_form.addRow("Target pressure:", self.npt_pressure_spin)

        self.npt_barostat_combo = QComboBox()
        self.npt_barostat_combo.addItems(
            ["Monte Carlo barostat", "Monte Carlo membrane barostat"]
        )
        npt_form.addRow("Barostat:", self.npt_barostat_combo)

        self.npt_barostat_interval_spin = QSpinBox()
        self.npt_barostat_interval_spin.setRange(1, 1000)
        self.npt_barostat_interval_spin.setValue(25)
        self.npt_barostat_interval_spin.setSuffix(" steps")
        npt_form.addRow("Barostat update interval:", self.npt_barostat_interval_spin)

        layout.addWidget(npt_group)

        # --- REMD Production ---
        remd_group = QGroupBox("REMD Production")
        remd_form = QFormLayout(remd_group)

        self.remd_tmin_spin = QDoubleSpinBox()
        # The REAL simulation temperature every replica runs at, so a
        # physical range is correct here.
        self.remd_tmin_spin.setRange(100.0, 500.0)
        self.remd_tmin_spin.setValue(300.0)
        self.remd_tmin_spin.setSuffix(" K")
        self.remd_tmin_spin.setToolTip(
            "The real temperature of the simulation - every replica runs at "
            "this temperature. State 0 is the unscaled reference ensemble."
        )
        remd_form.addRow("T min (real):", self.remd_tmin_spin)

        self.remd_tmax_spin = QDoubleSpinBox()
        # EFFECTIVE temperature of the hottest replica, not a real one: REST2
        # scales the solute Hamiltonian and every replica still runs at T min.
        # The OpenMM Cookbook REST tutorial uses 600 K, and a small hot region
        # can justify far more, so this must not be capped at a physical value.
        self.remd_tmax_spin.setRange(250.0, 2000.0)
        self.remd_tmax_spin.setValue(400.0)
        self.remd_tmax_spin.setSuffix(" K")
        self.remd_tmax_spin.setToolTip(
            "Effective temperature of the hottest replica (lambda = sqrt(T min / "
            "T max)). Nothing is physically heated above T min - REST2 only "
            "scales the solute's interactions. 500-700 K is typical; raise it "
            "until neighbour acceptance falls to roughly 20-40%."
        )
        remd_form.addRow("T max (effective):", self.remd_tmax_spin)

        self.remd_n_replicas_spin = QSpinBox()
        self.remd_n_replicas_spin.setRange(2, 256)
        self.remd_n_replicas_spin.setValue(16)
        remd_form.addRow("Number of replicas:", self.remd_n_replicas_spin)

        self.remd_temp_dist_combo = QComboBox()
        self.remd_temp_dist_combo.addItems(
            ["Exponential spacing", "Linear spacing", "Custom list"]
        )
        self.remd_temp_dist_combo.currentTextChanged.connect(
            self._on_temp_distribution_changed
        )
        remd_form.addRow("Temperature distribution:", self.remd_temp_dist_combo)

        self.remd_custom_temps_label = QLabel("Custom temperatures (K):")
        self.remd_custom_temps_edit = QLineEdit()
        self.remd_custom_temps_edit.setPlaceholderText(
            "Comma-separated, one per replica, ascending, first value = T min "
            "e.g. 300, 308, 317, 327, 340"
        )
        remd_form.addRow(self.remd_custom_temps_label, self.remd_custom_temps_edit)
        self.remd_custom_temps_label.setVisible(False)
        self.remd_custom_temps_edit.setVisible(False)

        self.remd_exchange_interval_spin = QSpinBox()
        self.remd_exchange_interval_spin.setRange(1, 100000)
        self.remd_exchange_interval_spin.setValue(1000)
        self.remd_exchange_interval_spin.setSuffix(" steps")
        remd_form.addRow("Exchange attempt interval:", self.remd_exchange_interval_spin)

        self.remd_total_steps_spin = QSpinBox()
        self.remd_total_steps_spin.setRange(0, 1000000000)
        self.remd_total_steps_spin.setValue(50000000)
        self.remd_total_steps_spin.setSuffix(" steps/replica")
        remd_form.addRow("Total production steps:", self.remd_total_steps_spin)

        self.remd_exchange_scheme_combo = QComboBox()
        self.remd_exchange_scheme_combo.addItems(
            ["Neighbor swap (Metropolis)", "Gibbs sampling"]
        )
        remd_form.addRow("Exchange scheme:", self.remd_exchange_scheme_combo)

        self.remd_ligand_resname_row = QWidget()
        ligand_resname_row_layout = QHBoxLayout(self.remd_ligand_resname_row)
        ligand_resname_row_layout.setContentsMargins(0, 0, 0, 0)
        self.remd_ligand_resnames_edit = QLineEdit("MOL")
        self.remd_ligand_resnames_edit.setPlaceholderText(
            "e.g. MOL  (comma-separated if several)"
        )
        self.remd_detect_ligand_button = QPushButton("Detect...")
        self.remd_detect_ligand_button.setToolTip(
            "Scan the generated system for residues that aren't standard amino "
            "acids, water, or ions - whatever's left is probably your ligand."
        )
        self.remd_detect_ligand_button.clicked.connect(
            self.detect_ligand_resnames_simulation
        )
        ligand_resname_row_layout.addWidget(self.remd_ligand_resnames_edit)
        ligand_resname_row_layout.addWidget(self.remd_detect_ligand_button)
        self.remd_ligand_resname_label = QLabel("Ligand residue name(s):")
        remd_form.addRow(self.remd_ligand_resname_label, self.remd_ligand_resname_row)

        self.remd_friction_spin = QDoubleSpinBox()
        self.remd_friction_spin.setRange(0.1, 20.0)
        self.remd_friction_spin.setSingleStep(0.1)
        self.remd_friction_spin.setValue(1.0)
        self.remd_friction_spin.setSuffix(" /ps")
        remd_form.addRow("Friction coefficient:", self.remd_friction_spin)

        self.remd_checkpoint_interval_spin = QSpinBox()
        self.remd_checkpoint_interval_spin.setRange(1, 100000)
        self.remd_checkpoint_interval_spin.setValue(200)
        self.remd_checkpoint_interval_spin.setSuffix(" cycles")
        remd_form.addRow("Checkpoint interval:", self.remd_checkpoint_interval_spin)

        self.remd_platform_combo = QComboBox()
        self.remd_platform_combo.addItems(["CUDA", "OpenCL", "HIP", "CPU"])
        remd_form.addRow("Platform:", self.remd_platform_combo)

        remd_outdir_row, self.remd_output_dir_edit = folder_picker_row(
            "Default: ./rest2_output"
        )
        remd_form.addRow("Output directory:", remd_outdir_row)

        self.remd_restart_cb = QCheckBox("Resume from previous run (restart)")
        self.remd_restart_cb.setToolTip(
            "Continues the production run stored in the output directory from its "
            "last checkpoint (equilibration is skipped). The ladder, timestep and "
            "exchange interval always come from the stored run; raise Total "
            "production steps to extend it."
        )
        remd_form.addRow(self.remd_restart_cb)

        layout.addWidget(remd_group)

        layout.addStretch()

        self.remd_status_label = QLabel("")
        self.remd_status_label.setWordWrap(True)
        layout.addWidget(self.remd_status_label)

        self.remd_progress_bar = QProgressBar()
        self.remd_progress_bar.setRange(0, 100)
        self.remd_progress_bar.setValue(0)
        self.remd_progress_bar.setFormat("%p%")
        layout.addWidget(self.remd_progress_bar)

        self.remd_log_box = QPlainTextEdit()
        self.remd_log_box.setReadOnly(True)
        self.remd_log_box.setMaximumBlockCount(2000)
        self.remd_log_box.setFixedHeight(160)
        layout.addWidget(self.remd_log_box)

        self.remd_hint = hint_label("")
        layout.addWidget(self.remd_hint)

        self.remd_start_button = mark_primary(QPushButton("Start REST2-REMD"))
        self.remd_start_button.clicked.connect(self.run_simulation)
        layout.addWidget(self.remd_start_button)

        return page

    def _on_temp_distribution_changed(self, text):
        is_custom = text == "Custom list"
        self.remd_custom_temps_label.setVisible(is_custom)
        self.remd_custom_temps_edit.setVisible(is_custom)

    def run_simulation(self):
        """Runs EM/NPT/NVT equilibration + REST2-REMD production as a
        separate subprocess (simulation_run.main() via `python -c`), streaming
        its stdout into remd_log_box.

        This has to be a real subprocess rather than a QThread: openmmtools'
        ReplicaExchangeSampler registers a signal handler internally (for
        graceful checkpoint-on-interrupt), and Python's signal module only
        allows that from the main thread of a process. A Qt background
        thread doesn't qualify, which raises "signal only works in main
        thread of the main interpreter" - a real subprocess has its own main
        thread, so this sidesteps the problem entirely.
        """

        prmtop = getattr(self, "system_prmtop_path", None)
        inpcrd = getattr(self, "system_inpcrd_path", None)
        if not prmtop or not inpcrd:
            QMessageBox.warning(
                self,
                "Simulation Run",
                "No system found. Please run 'Generate System' on the "
                "System Generation page first.",
            )
            return

        custom_temperatures_str = ""
        if self.remd_temp_dist_combo.currentText() == "Custom list":
            raw = self.remd_custom_temps_edit.text().strip()
            if not raw:
                QMessageBox.warning(
                    self,
                    "Simulation Run",
                    "Please enter custom temperature values (comma-separated), "
                    "or choose Exponential/Linear spacing instead.",
                )
                return
            try:
                custom_temps = [float(t.strip()) for t in raw.split(",") if t.strip()]
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Simulation Run",
                    "Couldn't parse the custom temperatures - use comma-separated "
                    "numbers only, e.g. 300, 308, 317, 327, 340",
                )
                return
            if len(custom_temps) != self.remd_n_replicas_spin.value():
                QMessageBox.warning(
                    self,
                    "Simulation Run",
                    f"You entered {len(custom_temps)} value(s), but Number of "
                    f"replicas is set to {self.remd_n_replicas_spin.value()}. "
                    f"These must match exactly - either add/remove values or "
                    f"change the replica count.",
                )
                return
            if custom_temps != sorted(custom_temps):
                QMessageBox.warning(
                    self,
                    "Simulation Run",
                    "Custom temperatures must be in ascending order (state 0 is "
                    "always the lowest/reference temperature).",
                )
                return
            if abs(custom_temps[0] - self.remd_tmin_spin.value()) > 1e-6:
                QMessageBox.warning(
                    self,
                    "Simulation Run",
                    f"The first custom temperature ({custom_temps[0]} K) must "
                    f"equal T min ({self.remd_tmin_spin.value()} K) - state 0 is "
                    f"always the unscaled reference state at the real simulation "
                    f"temperature.",
                )
                return
            custom_temperatures_str = ",".join(str(t) for t in custom_temps)

        # Protein-only mode: no ligand residues. REST2 still heats the whole
        # protein (AA_RESNAMES is always part of the solute in simulation_run.py),
        # so an empty ligand list gives a valid protein-only solute-tempering run.
        if self.is_protein_only():
            ligand_resnames = ""
        else:
            ligand_resnames = ",".join(
                s.strip().upper()
                for s in self.remd_ligand_resnames_edit.text().split(",")
                if s.strip()
            )

        exchange_interval = self.remd_exchange_interval_spin.value()
        total_steps = self.remd_total_steps_spin.value()
        n_iterations = max(1, total_steps // exchange_interval)
        output_dir = self.remd_output_dir_edit.text().strip() or "rest2_output"

        # Launched by import, not as a script: the distributed builds ship
        # simulation_run only as a compiled extension (.pyd/.so), which can be
        # imported but not executed. The app directory is passed as the first
        # argument and put on sys.path so the import works from any cwd.
        app_dir = os.path.dirname(os.path.abspath(__file__))
        launcher = (
            "import sys; sys.path.insert(0, sys.argv.pop(1)); "
            "import simulation_run; simulation_run.main(sys.argv[1:])"
        )

        proc_args = [
            "-c",
            launcher,
            app_dir,
            "--prmtop",
            prmtop,
            "--inpcrd",
            inpcrd,
            "--ligand-resnames",
            ligand_resnames,
            "--n-replicas",
            str(self.remd_n_replicas_spin.value()),
            "--t-min",
            str(self.remd_tmin_spin.value()),
            "--t-max",
            str(self.remd_tmax_spin.value()),
            "--temp-distribution",
            self.remd_temp_dist_combo.currentText(),
            "--custom-temperatures",
            custom_temperatures_str,
            "--timestep",
            str(self.nvt_timestep_spin.value()),
            "--friction",
            str(self.remd_friction_spin.value()),
            "--hydrogen-mass",
            # 0 disables hydrogen mass repartitioning.
            str(self.hydrogen_mass_spin.value() if self.hmr_cb.isChecked() else 0),
            "--nonbonded-method",
            self.nonbonded_method_combo.currentText(),
            "--nonbonded-cutoff",
            str(self.nonbonded_cutoff_spin.value()),
            "--constraints",
            self.constraints_combo.currentText(),
            "--ewald-error-tolerance",
            str(self.ewald_tol_spin.value()),
            "--em-max-iter",
            str(self.em_max_iter_spin.value()),
            "--em-tolerance",
            str(self.em_tolerance_spin.value()),
            "--npt-steps",
            str(self.npt_steps_spin.value()),
            "--npt-pressure",
            str(self.npt_pressure_spin.value()),
            "--npt-barostat",
            self.npt_barostat_combo.currentText(),
            "--npt-barostat-interval",
            str(self.npt_barostat_interval_spin.value()),
            "--nvt-steps",
            str(self.nvt_steps_spin.value()),
            "--nvt-temperature",
            str(self.nvt_temp_spin.value()),
            "--nvt-thermostat",
            self.nvt_thermostat_combo.currentText(),
            "--nvt-restraint-force-constant",
            "4184.0",
            "--n-iterations",
            str(n_iterations),
            "--n-steps-per-iter",
            str(exchange_interval),
            "--checkpoint-interval",
            str(self.remd_checkpoint_interval_spin.value()),
            "--exchange-scheme",
            self.remd_exchange_scheme_combo.currentText(),
            "--output-dir",
            output_dir,
            "--platform",
            self.remd_platform_combo.currentText(),
        ]
        if self.is_protein_only():
            proc_args.append("--protein-only")
        if not self.rigid_water_cb.isChecked():
            proc_args.append("--no-rigid-water")
        if self.nvt_restraint_cb.isChecked():
            proc_args.append("--nvt-restrain-heavy-atoms")
        if self.remd_restart_cb.isChecked():
            proc_args.append("--restart")

        self._sync_analysis_from_simulation(output_dir, exchange_interval)

        self.remd_log_box.clear()
        self.remd_progress_bar.setValue(0)
        self.remd_status_label.setText("Starting REST2-REMD run (separate process)...")
        self.remd_start_button.setEnabled(False)

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(proc_args)
        process.setProcessChannelMode(
            QProcess.MergedChannels
        )  # combine stdout+stderr into one stream

        process.readyReadStandardOutput.connect(
            lambda: self._on_remd_process_output(process)
        )
        process.finished.connect(
            lambda code, status: self._on_remd_process_finished(
                code, status, output_dir
            )
        )
        process.errorOccurred.connect(self._on_remd_process_error_occurred)

        self._remd_process = process
        process.start()

    def _sync_analysis_from_simulation(self, output_dir, exchange_interval):
        """Copies the production settings onto the Analysis page so its REMD
        ladder / timing can't silently disagree with the run (analysis.py
        also reads run_metadata.json from the output directory)."""
        self.analysis_input_dir_edit.setText(output_dir)
        self.analysis_n_replicas_spin.setValue(self.remd_n_replicas_spin.value())
        self.analysis_tmin_spin.setValue(self.remd_tmin_spin.value())
        self.analysis_tmax_spin.setValue(self.remd_tmax_spin.value())
        self.analysis_timestep_spin.setValue(self.nvt_timestep_spin.value())
        self.analysis_steps_per_iter_spin.setValue(exchange_interval)
        self.analysis_checkpoint_interval_spin.setValue(
            self.remd_checkpoint_interval_spin.value()
        )

    def _on_remd_process_output(self, process):
        data = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            if not line.strip():
                continue
            self.remd_log_box.appendPlainText(line)

            # Pull progress out of lines like:
            #   "  Progress: 40/500 iterations (8.0%) - elapsed 1.6 min, ETA 18.4 min"
            match = re.search(
                r"Progress:\s*\d+/\d+ iterations \((\d+(?:\.\d+)?)%\)", line
            )
            if match:
                self.remd_progress_bar.setValue(round(float(match.group(1))))

            # NVT/NPT/EM don't emit percent-based progress lines, but do log
            # clear stage markers - use those to at least show rough movement.
            elif "[1/3] EM + NPT" in line:
                self.remd_progress_bar.setValue(1)
            elif "[2/3] NVT" in line:
                self.remd_progress_bar.setValue(3)
            elif "Building REST system" in line:
                self.remd_progress_bar.setValue(5)
            elif "Minimising each thermodynamic state" in line:
                self.remd_progress_bar.setValue(6)

    def _on_remd_process_finished(self, exit_code, exit_status, output_dir):
        self.remd_start_button.setEnabled(True)
        if exit_code == 0:
            storage_path = os.path.join(output_dir, "rest2_remd.nc")
            self.remd_progress_bar.setValue(100)
            self.remd_status_label.setText(
                f"REST2-REMD complete. Output: {storage_path}"
            )
        else:
            self.remd_status_label.setText("REST2-REMD failed - see log above.")
            QMessageBox.critical(
                self,
                "Simulation Run",
                f"simulation_run.py exited with code {exit_code}. See the log box for details.",
            )

    def _on_remd_process_error_occurred(self, error):
        # Covers cases like the interpreter/script not being found at all,
        # which 'finished' never fires for.
        self.remd_start_button.setEnabled(True)
        self.remd_status_label.setText("REST2-REMD failed to start - see log above.")
        QMessageBox.critical(
            self,
            "Simulation Run",
            f"Failed to launch simulation_run.py (error code {error}).",
        )

    # Page 6: Analysis
    def create_analysis_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(section_header("Input"))

        indir_row, self.analysis_input_dir_edit = folder_picker_row(
            "Default: rest2_output"
        )
        row_wrap = QFormLayout()
        row_wrap.addRow("Production directory:", indir_row)
        layout.addLayout(row_wrap)

        top_row, self.analysis_topology_path = file_picker_row(
            "Default: same prmtop used for the simulation", "Topology Files (*.prmtop)"
        )
        ref_row, self.analysis_ref_pdb_path = file_picker_row(
            "Default: complex.pdb from System Generation", "PDB Files (*.pdb)"
        )
        nc_row, self.analysis_nc_path = file_picker_row(
            "Default: auto-detect the .nc file in the production directory",
            "NetCDF Files (*.nc)",
        )
        nc_ckpt_row, self.analysis_nc_checkpoint_path = file_picker_row(
            "Default: auto-detect the *checkpoint*.nc file", "NetCDF Files (*.nc)"
        )

        form_files = QFormLayout()
        form_files.addRow("Topology (prmtop):", top_row)
        form_files.addRow("Reference structure (PDB):", ref_row)
        form_files.addRow("NetCDF file (optional):", nc_row)
        form_files.addRow("NetCDF checkpoint (optional):", nc_ckpt_row)
        layout.addLayout(form_files)

        layout.addWidget(
            section_header("REMD Ladder (must match your Simulation Run settings)")
        )

        form_ladder = QFormLayout()

        self.analysis_n_replicas_spin = QSpinBox()
        self.analysis_n_replicas_spin.setRange(1, 256)
        self.analysis_n_replicas_spin.setValue(16)
        form_ladder.addRow("Number of replicas:", self.analysis_n_replicas_spin)

        # Must span at least the Simulation run page's range, or a valid ladder
        # could not be entered here to read the run back.
        self.analysis_tmin_spin = QDoubleSpinBox()
        self.analysis_tmin_spin.setRange(100.0, 500.0)
        self.analysis_tmin_spin.setValue(300.0)
        self.analysis_tmin_spin.setSuffix(" K")
        form_ladder.addRow("T min (real):", self.analysis_tmin_spin)

        self.analysis_tmax_spin = QDoubleSpinBox()
        self.analysis_tmax_spin.setRange(250.0, 2000.0)
        self.analysis_tmax_spin.setValue(400.0)
        self.analysis_tmax_spin.setSuffix(" K")
        form_ladder.addRow("T max (effective):", self.analysis_tmax_spin)

        layout.addLayout(form_ladder)

        layout.addWidget(section_header("Selections & Timing"))

        form_sel = QFormLayout()

        self.analysis_ligand_sel_row = QWidget()
        analysis_ligand_sel_row_layout = QHBoxLayout(self.analysis_ligand_sel_row)
        analysis_ligand_sel_row_layout.setContentsMargins(0, 0, 0, 0)
        self.analysis_ligand_sel_edit = QLineEdit("resname MOL")
        self.analysis_detect_ligand_button = QPushButton("Detect...")
        self.analysis_detect_ligand_button.setToolTip(
            "Scan the topology for residues that aren't standard amino acids, "
            "water, or ions - whatever's left is probably your ligand."
        )
        self.analysis_detect_ligand_button.clicked.connect(
            self.detect_ligand_resnames_analysis
        )
        analysis_ligand_sel_row_layout.addWidget(self.analysis_ligand_sel_edit)
        analysis_ligand_sel_row_layout.addWidget(self.analysis_detect_ligand_button)
        self.analysis_ligand_sel_label = QLabel("Ligand selection (MDTraj syntax):")
        form_sel.addRow(
            self.analysis_ligand_sel_label, self.analysis_ligand_sel_row
        )

        self.analysis_receptor_sel_edit = QLineEdit("protein")
        form_sel.addRow(
            "Receptor selection (MDTraj syntax):", self.analysis_receptor_sel_edit
        )

        self.analysis_timestep_spin = QDoubleSpinBox()
        self.analysis_timestep_spin.setRange(0.5, 5.0)
        self.analysis_timestep_spin.setSingleStep(0.5)
        self.analysis_timestep_spin.setValue(2.0)
        self.analysis_timestep_spin.setSuffix(" fs")
        form_sel.addRow("Timestep:", self.analysis_timestep_spin)

        self.analysis_steps_per_iter_spin = QSpinBox()
        self.analysis_steps_per_iter_spin.setRange(1, 100000)
        self.analysis_steps_per_iter_spin.setValue(1000)
        form_sel.addRow(
            "Steps per exchange iteration:", self.analysis_steps_per_iter_spin
        )

        self.analysis_checkpoint_interval_spin = QSpinBox()
        self.analysis_checkpoint_interval_spin.setRange(1, 100000)
        self.analysis_checkpoint_interval_spin.setValue(200)
        self.analysis_checkpoint_interval_spin.setSuffix(" cycles")
        form_sel.addRow("Checkpoint interval:", self.analysis_checkpoint_interval_spin)

        self.analysis_stride_spin = QSpinBox()
        self.analysis_stride_spin.setRange(1, 10000)
        self.analysis_stride_spin.setValue(1)
        form_sel.addRow("Frame stride:", self.analysis_stride_spin)

        layout.addLayout(form_sel)

        layout.addWidget(section_header("Options"))

        self.analysis_skip_structural_cb = QCheckBox(
            "Skip structural analysis (REMD diagnostics only)"
        )
        layout.addWidget(self.analysis_skip_structural_cb)

        self.analysis_image_molecules_cb = QCheckBox(
            "Apply PBC image_molecules() before analysis"
        )
        layout.addWidget(self.analysis_image_molecules_cb)

        self.analysis_skip_energy_decomp_cb = QCheckBox(
            "Skip energy decomposition (E_solute / E_solute-water) - this is the slowest step"
        )
        layout.addWidget(self.analysis_skip_energy_decomp_cb)

        energy_decomp_form = QFormLayout()
        self.analysis_energy_decomp_stride_spin = QSpinBox()
        self.analysis_energy_decomp_stride_spin.setRange(1, 10000)
        self.analysis_energy_decomp_stride_spin.setValue(10)
        self.analysis_energy_decomp_stride_spin.setToolTip(
            "Extra frame thinning on top of the main Frame stride, just for the "
            "energy decomposition step (it recomputes energies via OpenMM per "
            "frame, so it's much slower than the other analyses)."
        )
        energy_decomp_form.addRow(
            "Energy decomposition stride:", self.analysis_energy_decomp_stride_spin
        )

        self.analysis_fes_outlier_z_spin = QDoubleSpinBox()
        self.analysis_fes_outlier_z_spin.setRange(0.0, 50.0)
        self.analysis_fes_outlier_z_spin.setSingleStep(0.5)
        self.analysis_fes_outlier_z_spin.setValue(5.0)
        self.analysis_fes_outlier_z_spin.setSpecialValueText("Off (0)")
        self.analysis_fes_outlier_z_spin.setToolTip(
            "Removes isolated frames (e.g. a stray RMSD = 0 point) from the "
            "RMSD-Rg free-energy surface: frames beyond this many robust "
            "z-scores are dropped, but only if they are at most 1% of all "
            "frames. CSV outputs keep every frame."
        )
        energy_decomp_form.addRow(
            "FES outlier cutoff (robust z):", self.analysis_fes_outlier_z_spin
        )
        layout.addLayout(energy_decomp_form)

        layout.addWidget(section_header("Output"))

        outdir_row, self.analysis_output_dir_edit = folder_picker_row(
            "Default: analysis_output"
        )
        form_out = QFormLayout()
        form_out.addRow("Output directory:", outdir_row)
        layout.addLayout(form_out)

        layout.addStretch()

        self.analysis_status_label = QLabel("")
        self.analysis_status_label.setWordWrap(True)
        layout.addWidget(self.analysis_status_label)

        self.analysis_log_box = QPlainTextEdit()
        self.analysis_log_box.setReadOnly(True)
        self.analysis_log_box.setMaximumBlockCount(2000)
        self.analysis_log_box.setFixedHeight(160)
        layout.addWidget(self.analysis_log_box)

        self.analysis_run_button = mark_primary(QPushButton("Run Analysis"))
        self.analysis_run_button.clicked.connect(self.run_analysis)
        layout.addWidget(self.analysis_run_button)

        return page

    def run_analysis(self):
        """Runs analysis.run_rest2_analysis(...) in a background thread,
        streaming log output into analysis_log_box."""

        topology = self.analysis_topology_path.text().strip() or getattr(
            self, "system_prmtop_path", None
        ) or "complex.prmtop"
        # Reference PDB default: the PDB tleap wrote alongside the prmtop
        # (a bare "complex.pdb" resolved against the GUI's working directory,
        # not the System Generation folder, so it was usually not found).
        ref_pdb = (
            self.analysis_ref_pdb_path.text().strip()
            or getattr(self, "system_pdb_path", None)
            or os.path.splitext(topology)[0] + ".pdb"
        )

        kwargs = dict(
            input_dir=self.analysis_input_dir_edit.text().strip() or "rest2_output",
            output_dir=self.analysis_output_dir_edit.text().strip()
            or "analysis_output",
            topology=topology,
            ref_pdb=ref_pdb,
            protein_only=self.is_protein_only(),
            nc_file=self.analysis_nc_path.text().strip() or None,
            nc_checkpoint=self.analysis_nc_checkpoint_path.text().strip() or None,
            n_replicas=self.analysis_n_replicas_spin.value(),
            t_min=self.analysis_tmin_spin.value(),
            t_max=self.analysis_tmax_spin.value(),
            # Protein-only: "none" is a valid MDTraj selection that matches no
            # atoms, so every ligand-specific plot/metric is cleanly skipped
            # while protein RMSD/Rg/RMSF still run.
            ligand_sel=(
                "none"
                if self.is_protein_only()
                else (self.analysis_ligand_sel_edit.text().strip() or "resname MOL")
            ),
            receptor_sel=self.analysis_receptor_sel_edit.text().strip() or "protein",
            timestep=self.analysis_timestep_spin.value(),
            steps_per_iter=self.analysis_steps_per_iter_spin.value(),
            checkpoint_interval=self.analysis_checkpoint_interval_spin.value(),
            stride=self.analysis_stride_spin.value(),
            skip_structural=self.analysis_skip_structural_cb.isChecked(),
            image_molecules=self.analysis_image_molecules_cb.isChecked(),
            skip_energy_decomposition=self.analysis_skip_energy_decomp_cb.isChecked(),
            energy_decomposition_stride=self.analysis_energy_decomp_stride_spin.value(),
            fes_outlier_z=self.analysis_fes_outlier_z_spin.value(),
        )

        self.analysis_log_box.clear()
        self.analysis_status_label.setText("Starting analysis...")
        self.analysis_run_button.setEnabled(False)

        self._analysis_thread = QThread()
        self._analysis_worker = AnalysisWorker(kwargs)
        self._analysis_worker.moveToThread(self._analysis_thread)

        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.log_line.connect(self.analysis_log_box.appendPlainText)
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.finished.connect(self._analysis_thread.quit)
        self._analysis_worker.error.connect(self._analysis_thread.quit)

        self._analysis_thread.start()

    def _on_analysis_finished(self, result):
        self.analysis_status_label.setText(
            f"Done. {len(result['plots'])} plots, {len(result['csvs'])} CSVs written to {result['output_dir']}/"
        )
        self.analysis_run_button.setEnabled(True)

    def _on_analysis_error(self, message):
        self.analysis_status_label.setText("Analysis failed - see log above.")
        self.analysis_run_button.setEnabled(True)
        QMessageBox.critical(self, "Analysis", message)


def apply_dark_theme(app):
    """Applies a dark Fusion palette to the whole application, not just the
    sidebar - QMainWindow/QWidget/QTabWidget/dialogs etc. all pick this up
    automatically since it's set on the QApplication itself."""
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(230, 230, 230))
    palette.setColor(QPalette.Base, QColor(24, 24, 24))
    palette.setColor(QPalette.AlternateBase, QColor(40, 40, 40))
    palette.setColor(QPalette.ToolTipBase, QColor(230, 230, 230))
    palette.setColor(QPalette.ToolTipText, QColor(30, 30, 30))
    palette.setColor(QPalette.Text, QColor(230, 230, 230))
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, QColor(230, 230, 230))
    palette.setColor(QPalette.BrightText, QColor(255, 80, 80))
    palette.setColor(QPalette.Link, QColor(100, 160, 235))
    palette.setColor(QPalette.Highlight, QColor(60, 110, 180))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    # Dimmer text for disabled widgets so they're still readable but visually distinct
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(120, 120, 120))

    app.setPalette(palette)

    # Fusion's palette gets the colours right but leaves every control with its
    # default boxy chrome. This stylesheet is the "polish" layer: rounded
    # inputs/buttons, a single accent colour (#2d6cdf) for focus + primary
    # actions, a card-like sidebar list, and readable section headers.
    app.setStyleSheet("""
        QWidget { font-size: 13px; }

        QToolTip {
            color: #e6e6e6;
            background-color: #2d2d2d;
            border: 1px solid #555555;
            padding: 4px 6px;
        }

        /* ---- Section headings (section_header()) ---- */
        QLabel#SectionHeader {
            color: #7fb8ff;
            font-size: 13px;
            font-weight: 600;
            padding: 12px 0 5px 0;
            border-bottom: 1px solid #3a3a3a;
            margin-bottom: 6px;
        }
        QLabel#Hint {
            color: #9a9a9a;
            font-size: 11px;
        }

        /* ---- Inputs ---- */
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
            background-color: #242424;
            color: #e6e6e6;
            border: 1px solid #454545;
            border-radius: 6px;
            padding: 5px 8px;
            selection-background-color: #2d6cdf;
            selection-color: #ffffff;
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus,
        QDoubleSpinBox:focus, QPlainTextEdit:focus {
            border: 1px solid #4a9eff;
        }
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,
        QDoubleSpinBox:disabled {
            background-color: #1e1e1e;
            color: #6a6a6a;
        }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox QAbstractItemView {
            background-color: #242424;
            color: #e6e6e6;
            border: 1px solid #454545;
            selection-background-color: #2d6cdf;
            selection-color: #ffffff;
            outline: 0;
        }

        /* ---- Buttons ---- */
        QPushButton {
            background-color: #2d2d2d;
            color: #e6e6e6;
            border: 1px solid #4a4a4a;
            border-radius: 6px;
            padding: 6px 14px;
        }
        QPushButton:hover  { background-color: #383838; border-color: #5a5a5a; }
        QPushButton:pressed { background-color: #262626; }
        QPushButton:disabled { color: #6a6a6a; border-color: #3a3a3a; }

        QPushButton[primary="true"] {
            background-color: #2d6cdf;
            color: #ffffff;
            border: 1px solid #2d6cdf;
            font-weight: 600;
            padding: 8px 18px;
        }
        QPushButton[primary="true"]:hover   { background-color: #3b7bef; }
        QPushButton[primary="true"]:pressed { background-color: #245ec2; }
        QPushButton[primary="true"]:disabled {
            background-color: #333333;
            border-color: #333333;
            color: #7a7a7a;
        }

        /* ---- Sidebar step list ---- */
        QListWidget#Sidebar {
            border: none;
            background: transparent;
            outline: 0;
        }
        QListWidget#Sidebar::item {
            padding: 10px 12px;
            margin: 2px 8px;
            border-radius: 6px;
            color: #cfcfcf;
        }
        QListWidget#Sidebar::item:selected {
            background-color: #2d6cdf;
            color: #ffffff;
        }
        QListWidget#Sidebar::item:hover:!selected {
            background-color: #333333;
        }
        QListWidget#Sidebar::item:disabled { color: #666666; }

        /* ---- Group boxes ---- */
        QGroupBox {
            font-weight: bold;
            border: 1px solid #444444;
            border-radius: 6px;
            margin-top: 12px;
            padding: 10px 8px 8px 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: #7fb8ff;
        }

        /* ---- Progress bar ---- */
        QProgressBar {
            border: 1px solid #444444;
            border-radius: 6px;
            text-align: center;
            background-color: #242424;
            color: #e6e6e6;
            height: 16px;
        }
        QProgressBar::chunk {
            background-color: #2d6cdf;
            border-radius: 5px;
        }

        /* ---- Scrollbars ---- */
        QScrollBar:vertical {
            background: transparent; width: 10px; margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #3f3f3f; border-radius: 5px; min-height: 24px;
        }
        QScrollBar::handle:vertical:hover { background: #505050; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar:horizontal {
            background: transparent; height: 10px; margin: 0;
        }
        QScrollBar::handle:horizontal {
            background: #3f3f3f; border-radius: 5px; min-width: 24px;
        }
        QScrollBar::handle:horizontal:hover { background: #505050; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
    """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_theme(app)

    window = MainWindow()
    # _size_window_to_screen() still runs in __init__ so the window has a
    # sensible size to restore to when the user un-maximises.
    window.showMaximized()

    sys.exit(app.exec())
