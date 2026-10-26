"""Desktop karaoke window: scrolling pitch view of the song's melody and the singer's voice."""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QKeySequence, QPainter, QPalette, QPen, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from karaoke.engine import DEFAULT_MIC_THRESHOLD_DB, KaraokeEngine
from karaoke.pitch import fold_octave, note_name

AUDIO_FILTER = "Audio (*.mp3 *.flac *.wav *.m4a *.ogg *.opus *.aac *.wma);;All files (*)"

BACKGROUND = QColor("#14151c")
GRID = QColor(255, 255, 255, 18)
GRID_C = QColor(255, 255, 255, 45)
LABEL = QColor(255, 255, 255, 110)
TARGET_PAST = QColor(90, 140, 210, 90)
TARGET_FUTURE = QColor(110, 180, 255, 210)
VOICE_HIT = QColor("#5be37d")
VOICE_MISS = QColor("#ff9f43")
PLAYHEAD = QColor(255, 255, 255, 90)
METER_IDLE = QColor("#4a5068")

# Mic sensitivity slider (0-100) <-> gate threshold in dBFS: 0 = -20 dB (ignores noise), 100 = -80 dB (picks up whispers).
def sensitivity_to_db(value: int) -> float:
    return -20.0 - 0.6 * value


def db_to_sensitivity(db: float) -> int:
    return round((-20.0 - db) / 0.6)


class PitchView(QWidget):
    """Piano-roll: time scrolls right-to-left, pitch is vertical, the playhead is fixed."""

    PAST, FUTURE = 3.0, 5.0  # seconds visible either side of the playhead
    GUTTER = 40  # px reserved for note names
    CURVE_STEP_PX = 3.0  # spacing of the points of the unsnapped melody curve

    def __init__(self):
        super().__init__()
        self.engine: KaraokeEngine | None = None
        self.snap = True  # draw held semitone notes instead of raw pitch curves
        self.setMinimumSize(640, 320)

    def set_engine(self, engine: KaraokeEngine | None) -> None:
        self.engine = engine
        if engine:
            lo, hi = engine.song.pitch.range()
            centre = (lo + hi) / 2
            span = max(hi - lo + 6, 18)
            self._lo, self._hi = centre - span / 2, centre + span / 2
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BACKGROUND)
        if not self.engine:
            p.setPen(LABEL)
            p.drawText(self.rect(), Qt.AlignCenter, "Open a song (or drop an audio file here) to start")
            return

        w, h = self.width(), self.height()
        now = self.engine.position
        play_x = w * self.PAST / (self.PAST + self.FUTURE)
        px_per_s = w / (self.PAST + self.FUTURE)
        semitone = h / (self._hi - self._lo)
        x_of = lambda t: play_x + (t - now) * px_per_s
        y_of = lambda m: (self._hi - m) * semitone

        self._draw_grid(p, w, semitone, y_of)

        # Reference melody: dim behind the playhead, bright ahead of it.
        song = self.engine.song
        ref = song.notes if self.snap else song.pitch
        t0, t1 = now - self.PAST, now + self.FUTURE
        times = np.arange(math.floor(t0 / ref.hop), math.ceil(t1 / ref.hop)) * ref.hop
        target = ref.at(times)
        thickness = max(semitone * 0.8, 6)
        if self.snap:
            runs = _note_runs(times, target, ref.hop)
            parts = (None, None)
        else:
            # A point every few pixels is enough for a smooth curve; stroking every 10 ms frame is the slow part.
            step = max(int(self.CURVE_STEP_PX / (px_per_s * ref.hop)), 1)
            times, target = times[::step], target[::step]
            split = int(np.searchsorted(times, now))
            parts = (slice(0, split + 1), slice(max(split - 1, 0), None))  # stroke each side once, overlapping at the playhead
        for colour, clip, part in ((TARGET_PAST, QRectF(0, 0, play_x, h), parts[0]),
                                   (TARGET_FUTURE, QRectF(play_x, 0, w - play_x, h), parts[1])):
            p.setClipRect(clip)
            if self.snap:
                p.setPen(Qt.NoPen)
                p.setBrush(colour)
                for a, b, note in runs:
                    _bar(p, x_of(a), x_of(b), y_of(note), thickness)
            else:
                p.setPen(QPen(colour, thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                for line in _polylines(x_of(times[part]), y_of(target[part])):
                    p.drawPolyline(line)
        p.setClipping(False)

        # Singer, folded to the octave of the target so a lower/higher voice still lines up.
        voice = [f for f in self.engine.voice_since(t0) if f.t <= now]
        targets = ref.at(np.array([f.t for f in voice])).tolist() if voice else []
        points, last = [], None  # (t, pitch as drawn, hit); None marks a gap
        for frame, anchor in zip(voice, targets):
            value = frame.note if self.snap else frame.midi
            if value != value or (last and frame.t - last[0] > 0.05):  # NaN: not singing
                points.append(None)
                last = None
                if value != value:
                    continue
            # Octave to draw in: the target; between melody notes, continue the current line;
            # at the start of a line, the nearest melody note (e.g. singing just before a phrase starts).
            if anchor != anchor:
                anchor = last[1] if last else ref.nearest_voiced(frame.t)
            if anchor != anchor:
                anchor = (self._lo + self._hi) / 2
            last = (frame.t, value - 12.0 * round((value - anchor) / 12.0), frame.hit)
            points.append(last)
        if self.snap:
            self._draw_voice_bars(p, points, x_of, y_of, max(semitone * 0.45, 4), ref.hop)
        else:
            self._draw_voice_curve(p, points, x_of, y_of)

        p.setPen(QPen(PLAYHEAD, 2))
        p.drawLine(QPointF(play_x, 0), QPointF(play_x, h))

        # Live marker and note readout.
        current = voice[-1] if voice and now - voice[-1].t < 0.25 and not np.isnan(voice[-1].note) else None
        if current and last:
            p.setPen(Qt.NoPen)
            p.setBrush(VOICE_HIT if current.hit else VOICE_MISS)
            p.drawEllipse(QPointF(play_x, y_of(last[1])), 8, 8)
        self._draw_labels(p, h, semitone, y_of)
        self._draw_readout(p, song.notes.at(now), current)

    def _draw_voice_bars(self, p: QPainter, points, x_of, y_of, thickness: float, hop: float) -> None:
        """One bar per stretch of frames on the same note with the same hit/miss state."""
        p.setPen(Qt.NoPen)
        run = []
        for point in points + [None]:
            if run and (point is None or point[1:] != run[0][1:]):
                p.setBrush(VOICE_HIT if run[0][2] else VOICE_MISS)
                _bar(p, x_of(run[0][0]), x_of(run[-1][0] + hop), y_of(run[0][1]), thickness)
                run = []
            if point:
                run.append(point)

    def _draw_voice_curve(self, p: QPainter, points, x_of, y_of) -> None:
        """One polyline per stretch of points with the same hit/miss state."""
        run = []
        for point in points + [None]:
            if run and (point is None or point[2] != run[-1][2]):
                if point is not None:
                    run.append(point)  # join up with the next stretch
                if len(run) > 1:
                    p.setPen(QPen(VOICE_HIT if run[0][2] else VOICE_MISS, 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    p.drawPolyline([QPointF(x_of(t), y_of(m)) for t, m, _ in run])
                run = []
            if point:
                run.append(point)

    def _notes(self) -> range:
        return range(math.ceil(self._lo), math.floor(self._hi) + 1)

    def _draw_grid(self, p: QPainter, w: int, semitone: float, y_of) -> None:
        p.setRenderHint(QPainter.Antialiasing, False)  # hairlines don't need it, and it's slow at full width
        for m in self._notes():
            p.setPen(QPen(GRID_C if m % 12 == 0 else GRID, 1))
            p.drawLine(QPointF(self.GUTTER, y_of(m)), QPointF(w, y_of(m)))
        p.setRenderHint(QPainter.Antialiasing)

    def _draw_labels(self, p: QPainter, h: int, semitone: float, y_of) -> None:
        """Note names in a gutter drawn over the curves so they stay readable."""
        p.fillRect(QRectF(0, 0, self.GUTTER, h), BACKGROUND)
        p.setFont(QFont(self.font().family(), 8))
        p.setPen(LABEL)
        for m in self._notes():
            if m % 12 == 0 or semitone > 14:
                p.drawText(QRectF(0, y_of(m) - 8, self.GUTTER - 6, 16), Qt.AlignRight | Qt.AlignVCenter, note_name(m))

    def _draw_readout(self, p: QPainter, target: float, current) -> None:
        parts = [f"Target {note_name(target)}" if not np.isnan(target) else "Target —"]
        if current:
            if np.isnan(target):
                parts.append(f"You {note_name(current.note)}")
            else:
                # Cents from the target note (octave-independent), from the unsnapped pitch.
                cents = (fold_octave(current.midi, target) - target) * 100
                parts.append(f"You {note_name(fold_octave(current.note, target))} ({cents:+.0f}¢)")
        p.setPen(QColor(255, 255, 255, 200))
        p.setFont(QFont(self.font().family(), 11))
        p.drawText(QRectF(44, 8, 400, 24), Qt.AlignLeft | Qt.AlignVCenter, "   ".join(parts))


def _note_runs(times, notes, hop: float) -> list[tuple[float, float, float]]:
    """(start, end, note) for each stretch of equal, non-NaN notes."""
    if len(notes) == 0:
        return []
    changes = np.flatnonzero(notes[1:] != notes[:-1]) + 1  # NaN != NaN: unvoiced frames become 1-frame runs
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(notes)]])
    voiced = ~np.isnan(notes[starts])
    return [(times[a], times[b - 1] + hop, note)
            for a, b, note in zip(starts[voiced].tolist(), ends[voiced].tolist(), notes[starts[voiced]].tolist())]


def _bar(p: QPainter, x0: float, x1: float, y: float, thickness: float) -> None:
    radius = thickness / 2
    p.drawRoundedRect(QRectF(x0, y - radius, max(x1 - x0, thickness), thickness), radius, radius)


def _polylines(xs: np.ndarray, ys: np.ndarray) -> list[list[QPointF]]:
    """Points of each unbroken (non-NaN) stretch of a curve."""
    voiced = ~np.isnan(ys)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], voiced.astype(np.int8), [0]])))
    return [[QPointF(x, y) for x, y in zip(xs[a:b].tolist(), ys[a:b].tolist())]
            for a, b in zip(edges[::2], edges[1::2])]


class LyricsView(QWidget):
    """Current lyric line with a karaoke wipe over the sung words, and the next line underneath.

    Japanese lines get furigana (kana readings) above the kanji.
    """

    LEAD_IN = 2.0  # seconds before a line starts when it moves into the main slot
    SUNG = QColor(110, 180, 255)  # matches the melody bars ahead of the playhead
    RUBY_SCALE = 0.5  # furigana size relative to the line

    def __init__(self):
        super().__init__()
        self.engine: KaraokeEngine | None = None
        self._ruby = False
        self.setFixedHeight(84)

    def set_engine(self, engine: KaraokeEngine | None) -> None:
        self.engine = engine
        lyrics = engine.song.lyrics if engine else None
        self._ruby = bool(lyrics and any(line.ruby for line in lyrics.lines))
        self.setFixedHeight(118 if self._ruby else 84)  # room for a furigana row above each line
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.fillRect(self.rect(), BACKGROUND)
        if not self.engine:
            return
        lyrics = self.engine.song.lyrics
        if not lyrics:
            p.setPen(LABEL)
            p.drawText(self.rect(), Qt.AlignCenter, "No synced lyrics found for this song")
            return
        main_h, next_h = (64, 44) if self._ruby else (44, 30)
        now = self.engine.position
        current = lyrics.line_at(now)
        upcoming = lyrics.next_line(now)
        started = current is not None
        if not started and upcoming is not None and lyrics.lines[upcoming].start + lyrics.user_offset - now <= self.LEAD_IN:
            current = upcoming  # about to start: show it, not yet highlighted
        if current is not None:
            upcoming = current + 1 if current + 1 < len(lyrics.lines) else None
            line = lyrics.lines[current]
            sung = self._sung_fraction(line, lyrics, now) if started else 0.0
            self._draw_line(p, line, QRectF(0, 4, self.width(), main_h), 20, QFont.DemiBold,
                            QColor(255, 255, 255, 235 if started else 150), sung)
        if upcoming is not None:
            self._draw_line(p, lyrics.lines[upcoming], QRectF(0, 4 + main_h, self.width(), next_h), 13, QFont.Normal,
                            QColor(255, 255, 255, 110), 0.0)

    @staticmethod
    def _sung_fraction(line, lyrics, now: float) -> tuple[int, float] | float:
        """(index of the word being sung, how far into it) or a plain 0..1 when there are no word timings."""
        if not line.words:
            return 1.0  # whole line lit while it is being sung
        t = now - lyrics.word_shift
        starts = [w.start for w in line.words]
        ends = starts[1:] + [max(line.words[-1].end, line.end)]
        for i, (a, b) in enumerate(zip(starts, ends)):
            if t < b:
                return i, min(max((t - a) / max(b - a, 1e-3), 0.0), 1.0)
        return len(starts), 0.0

    def _font(self, size: int, weight) -> QFont:
        font = QFont(self.font().family(), size, weight)
        font.setFamilies([self.font().family(), "Yu Gothic UI", "Meiryo UI"])  # Japanese UI fonts for kana/kanji
        return font

    def _draw_line(self, p: QPainter, line, rect: QRectF, size: int, weight, colour: QColor, sung) -> None:
        words = line.text.split()
        joined = " ".join(words)
        font = self._font(size, weight)
        while size > 10 and QFontMetricsF(font).horizontalAdvance(joined) > rect.width() - 40:
            size -= 1
            font.setPointSize(size)  # long line: shrink to fit
        metrics = QFontMetricsF(font)
        left = rect.x() + (rect.width() - metrics.horizontalAdvance(joined)) / 2
        ruby_font = self._font(max(round(size * self.RUBY_SCALE), 7), weight)
        ruby_h = QFontMetricsF(ruby_font).height() if self._ruby else 0.0
        text_rect = rect.adjusted(0, ruby_h, 0, 0)
        top = text_rect.center().y() - metrics.height() / 2
        # Furigana centred over their kanji, at the x positions those characters get in the line.
        ruby, offset = [], 0
        if "".join(text for text, _ in line.ruby) == joined:
            for text, reading in line.ruby:
                if reading:
                    centre = left + (metrics.horizontalAdvance(joined[:offset]) + metrics.horizontalAdvance(joined[:offset + len(text)])) / 2
                    ruby.append((QRectF(centre - 100, top - ruby_h, 200, ruby_h), reading))
                offset += len(text)

        def draw(pen: QColor) -> None:
            p.setPen(pen)
            p.setFont(font)
            p.drawText(text_rect, Qt.AlignCenter, joined)
            p.setFont(ruby_font)
            for box, reading in ruby:
                p.drawText(box, Qt.AlignHCenter | Qt.AlignBottom, reading)

        draw(colour)
        if not sung:
            return
        if isinstance(sung, tuple):
            index, into = sung
            done = " ".join(words[:index])
            x = metrics.horizontalAdvance(done + " " if done else "")
            if index < len(words):
                x += into * metrics.horizontalAdvance(words[index])
            else:
                x = metrics.horizontalAdvance(joined)
        else:
            x = sung * metrics.horizontalAdvance(joined)
        p.save()
        p.setClipRect(QRectF(left, rect.y(), x, rect.height()))  # covers the furigana row too
        draw(self.SUNG)
        p.restore()


class LevelMeter(QWidget):
    """Mic level bar with the sensitivity threshold marked; green while the voice is loud enough to track."""

    FLOOR = -80.0

    def __init__(self):
        super().__init__()
        self.setFixedHeight(10)
        self.setMinimumWidth(120)
        self.level = self.FLOOR
        self.threshold = DEFAULT_MIC_THRESHOLD_DB

    def set_level(self, db: float) -> None:
        # Instant attack, ~45 dB/s release so the bar is readable.
        self.level = max(min(db, 0.0), self.level - 0.75, self.FLOOR)
        self.update()

    def set_threshold(self, db: float) -> None:
        self.threshold = db
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        w, h = self.width(), self.height()
        x_of = lambda db: w * (db - self.FLOOR) / -self.FLOOR
        p.fillRect(0, 0, w, h, BACKGROUND)
        p.fillRect(QRectF(0, 0, x_of(self.level), h), VOICE_HIT if self.level >= self.threshold else METER_IDLE)
        p.fillRect(QRectF(x_of(self.threshold) - 1, 0, 2, h), QColor(255, 255, 255, 200))


class SongLoader(QObject):
    """Prepares a song (separation + melody) and opens the audio engine off the UI thread."""

    progress = Signal(str)
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def run(self) -> None:
        try:
            from karaoke.library import prepare_song

            song = prepare_song(self.path, progress=self.progress.emit)
            self.progress.emit("Loading audio...")
            self.loaded.emit(KaraokeEngine(song))
        except Exception as exc:  # surfaced in a dialog
            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Karaoke")
        self.resize(1100, 640)
        self.setAcceptDrops(True)
        self.engine: KaraokeEngine | None = None
        self._thread: QThread | None = None

        self.view = PitchView()
        self.title = QLabel("No song loaded")
        self.title.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.status = QLabel("")
        self.status.setStyleSheet("color: #9aa0b0;")
        self.score = QLabel("—")
        self.score.setStyleSheet("font-size: 28px; font-weight: 700; color: #5be37d;")
        self.score.setToolTip("Share of the melody you matched within half a semitone (octave-independent)")
        self.clock = QLabel("0:00 / 0:00")

        open_btn = QPushButton("Open song…")
        open_btn.clicked.connect(self.choose_song)
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        restart_btn = QPushButton("Restart")
        restart_btn.clicked.connect(self.restart)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_play)

        self.settings = QSettings("karaoke", "karaoke")
        self.mic = LevelMeter()
        self.mic.setToolTip("Microphone level. The white mark is the sensitivity threshold: "
                            "the bar turns green when your voice is loud enough to be tracked.")

        header = QHBoxLayout()
        header.addWidget(open_btn)
        header.addSpacing(12)
        header.addWidget(self.title)
        header.addStretch()
        header.addWidget(self.status)
        header.addSpacing(16)
        self.snap = QCheckBox("Snap to notes")
        self.snap.setToolTip("Show the melody and your voice as notes (C, C#, D, ...) instead of raw pitch curves")
        self.snap.setChecked(self.settings.value("snap_to_notes", True, type=bool))
        self.snap.toggled.connect(self._set_snap)
        self._set_snap(self.snap.isChecked())
        header.addWidget(self.snap)

        mixer = QGridLayout()
        self.sliders = {}
        for row, (channel, label, value) in enumerate(
            [("music", "Music", 100), ("guide", "Guide vocal", 0)]
        ):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(value)
            slider.valueChanged.connect(lambda v, c=channel: self._set_gain(c, v))
            mixer.addWidget(QLabel(label), row, 0)
            mixer.addWidget(slider, row, 1)
            self.sliders[channel] = slider
        self.latency = QSpinBox()
        self.latency.setRange(-500, 500)
        self.latency.setSingleStep(10)
        self.latency.setSuffix(" ms")
        self.latency.setToolTip("Shift your voice line if it looks early (+) or late (-) against the melody")
        self.latency.valueChanged.connect(self._set_latency)
        mixer.addWidget(QLabel("Voice offset"), 2, 0)
        mixer.addWidget(self.latency, 2, 1)
        self.sensitivity = QSlider(Qt.Horizontal)
        self.sensitivity.setRange(0, 100)
        self.sensitivity.setToolTip("Raise if your singing isn't picked up; lower if background noise draws a line")
        self.sensitivity_label = QLabel()
        self.sensitivity_label.setMinimumWidth(52)
        self.sensitivity.valueChanged.connect(self._set_sensitivity)
        mixer.addWidget(QLabel("Mic sensitivity"), 3, 0)
        mixer.addWidget(self.sensitivity, 3, 1)
        mixer.addWidget(self.sensitivity_label, 3, 2)
        self.lyrics_offset = QDoubleSpinBox()
        self.lyrics_offset.setRange(-60.0, 60.0)
        self.lyrics_offset.setSingleStep(0.1)
        self.lyrics_offset.setDecimals(1)
        self.lyrics_offset.setSuffix(" s")
        self.lyrics_offset.setToolTip("Shift the lyrics if they appear early (+) or late (-); saved per song")
        self.lyrics_offset.setEnabled(False)
        self.lyrics_offset.valueChanged.connect(self._set_lyrics_offset)
        mixer.addWidget(QLabel("Lyrics offset"), 4, 0)
        mixer.addWidget(self.lyrics_offset, 4, 1)

        transport = QVBoxLayout()
        buttons = QHBoxLayout()
        buttons.addWidget(self.play_btn)
        buttons.addWidget(restart_btn)
        transport.addLayout(buttons)
        transport.addWidget(self.clock)
        transport.addWidget(QLabel("Mic"))
        transport.addWidget(self.mic)
        transport.addStretch()

        scoring = QVBoxLayout()
        scoring.addWidget(QLabel("Score"), alignment=Qt.AlignRight)
        scoring.addWidget(self.score, alignment=Qt.AlignRight)
        scoring.addStretch()

        footer = QHBoxLayout()
        footer.addLayout(transport)
        footer.addSpacing(24)
        footer.addLayout(mixer, stretch=1)
        footer.addSpacing(24)
        footer.addLayout(scoring)

        root = QVBoxLayout()
        root.addLayout(header)
        root.addWidget(self.view, stretch=1)
        self.lyrics_view = LyricsView()
        root.addWidget(self.lyrics_view)
        root.addLayout(footer)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        saved = self.settings.value("mic_threshold_db", DEFAULT_MIC_THRESHOLD_DB, type=float)
        self.sensitivity.setValue(db_to_sensitivity(saved))
        self._set_sensitivity(self.sensitivity.value())

        self.timer = QTimer(self, interval=16, timeout=self._tick)
        self.timer.start()

    # --- song loading ----------------------------------------------------

    def choose_song(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open song", str(Path.home() / "Music"), AUDIO_FILTER)
        if path:
            self.load_song(Path(path))

    def load_song(self, path: Path) -> None:
        if self._thread and self._thread.isRunning():
            return
        self._close_engine()
        self.title.setText(path.stem)
        self.status.setText("Loading...")
        self._thread = QThread(self)
        self._loader = SongLoader(path)
        self._loader.moveToThread(self._thread)
        self._thread.started.connect(self._loader.run)
        self._loader.progress.connect(self.status.setText)
        self._loader.loaded.connect(self._on_loaded)
        self._loader.failed.connect(self._on_failed)
        self._loader.loaded.connect(self._thread.quit)
        self._loader.failed.connect(self._thread.quit)
        self._thread.start()

    def _on_loaded(self, engine: KaraokeEngine) -> None:
        try:
            engine.start()  # on the UI thread: WASAPI needs COM, which Qt initialises here
        except Exception as exc:
            engine.close()
            self._on_failed(f"Could not open audio devices: {exc}")
            return
        self.engine = engine
        for channel, slider in self.sliders.items():
            self._set_gain(channel, slider.value())
        self._set_latency(self.latency.value())
        self._set_sensitivity(self.sensitivity.value())
        self.view.set_engine(engine)
        self.lyrics_view.set_engine(engine)
        lyrics = engine.song.lyrics
        self.lyrics_offset.setEnabled(lyrics is not None)
        self.lyrics_offset.blockSignals(True)
        self.lyrics_offset.setValue(lyrics.user_offset if lyrics else 0.0)
        self.lyrics_offset.blockSignals(False)
        self.status.setText(f"Ready · {engine.sr} Hz · audio latency {engine.round_trip * 1000:.0f} ms · headphones recommended")

    def _on_failed(self, message: str) -> None:
        self.status.setText("Failed to load")
        QMessageBox.critical(self, "Could not load song", message)

    def _close_engine(self) -> None:
        if self.engine:
            self.engine.close()
        self.engine = None
        self.view.set_engine(None)
        self.lyrics_view.set_engine(None)
        self.lyrics_offset.setEnabled(False)
        self.play_btn.setText("Play")

    # --- controls --------------------------------------------------------

    def toggle_play(self) -> None:
        if not self.engine:
            return
        if self.engine.playing:
            self.engine.pause()
        else:
            self.engine.play()

    def restart(self) -> None:
        if self.engine:
            self.engine.seek(0.0)

    def _set_gain(self, channel: str, value: int) -> None:
        if self.engine:
            self.engine.gains[channel] = value / 100

    def _set_latency(self, ms: int) -> None:
        if self.engine:
            self.engine.latency_offset = ms / 1000

    def _set_lyrics_offset(self, seconds: float) -> None:
        if self.engine and self.engine.song.lyrics:
            self.engine.song.lyrics.user_offset = seconds
            self.engine.song.lyrics.save(self.engine.song.folder)

    def _set_snap(self, on: bool) -> None:
        self.view.snap = on
        self.settings.setValue("snap_to_notes", on)

    def _set_sensitivity(self, value: int) -> None:
        db = sensitivity_to_db(value)
        self.sensitivity_label.setText(f"{db:.0f} dB")
        self.mic.set_threshold(db)
        self.settings.setValue("mic_threshold_db", db)
        if self.engine:
            self.engine.mic_threshold_db = db

    def _tick(self) -> None:
        self.view.update()
        self.lyrics_view.update()
        if not self.engine:
            return
        pos, dur = self.engine.position, self.engine.duration
        self.clock.setText(f"{int(pos // 60)}:{int(pos % 60):02d} / {int(dur // 60)}:{int(dur % 60):02d}")
        self.play_btn.setText("Pause" if self.engine.playing else "Play")
        self.mic.set_level(self.engine.level_db)
        score = self.engine.scorer.score(pos)
        self.score.setText("—" if score is None else f"{score:.0f}%")

    # --- drag & drop -----------------------------------------------------

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls:
            self.load_song(Path(urls[0].toLocalFile()))

    def closeEvent(self, event) -> None:
        self._close_engine()
        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
        super().closeEvent(event)


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    for role, colour in [
        (QPalette.Window, "#1b1c24"), (QPalette.WindowText, "#e6e8ef"), (QPalette.Base, "#14151c"),
        (QPalette.AlternateBase, "#1f2029"), (QPalette.Text, "#e6e8ef"), (QPalette.Button, "#262834"),
        (QPalette.ButtonText, "#e6e8ef"), (QPalette.Highlight, "#5b8def"), (QPalette.HighlightedText, "#ffffff"),
        (QPalette.ToolTipBase, "#262834"), (QPalette.ToolTipText, "#e6e8ef"),
    ]:
        palette.setColor(role, QColor(colour))
    app.setPalette(palette)


def run(song: Path | None = None) -> int:
    # The audio callback and pitch analysis share the GIL with drawing; hand it over every 1 ms
    # (default 5 ms) so a repaint can't hold up a 10 ms audio block.
    sys.setswitchinterval(0.001)
    app = QApplication.instance() or QApplication(sys.argv)
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    if song:
        window.load_song(song)
    return app.exec()
