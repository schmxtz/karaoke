# Karaoke

A desktop karaoke app that turns any song file into a karaoke track. It removes the vocals, shows the melody as notes to sing along to, tracks your voice from the microphone in real time, scores how well you hit the notes, and shows time-synced lyrics with word-by-word highlighting (with furigana for Japanese songs).

Everything runs locally except the lyrics lookup.

## Features

- **Vocal removal**: [Mel-Band RoFormer](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model) via [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) splits a song into vocals and instrumental. A slider brings the original vocals back as a guide.
- **Melody**: the song's melody is extracted from the separated vocals with [CREPE](https://github.com/maxrmorrison/torchcrepe) and drawn as a scrolling piano roll, either as note bars (C, C#, D, ...) or as a raw pitch curve.
- **Live pitch tracking**: your voice is analysed in real time (YIN), compensated for audio latency, and drawn on top of the melody. Green means you're on the note, orange means off. Singing an octave higher or lower still counts.
- **Scoring**: the percentage of the melody you matched so far.
- **Synced lyrics**: fetched from [LRCLIB](https://lrclib.net) and automatically shifted to match your file (e.g. a YouTube rip with a longer intro). A per-song offset can be adjusted by hand.
- **Word-by-word highlighting**: each lyric line is aligned against the vocals with a [wav2vec2](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self) model, so the karaoke wipe follows the singer.
- **Furigana**: Japanese lyrics get kana readings above the kanji (MeCab + UniDic via [fugashi](https://github.com/polm/fugashi)).
- **Adjustable mic sensitivity**, voice offset and music/guide-vocal volume. Settings are remembered.

## Requirements

- **Windows 10/11.** This is the only platform tested so far. Linux should work, but it's untested. macOS isn't supported yet, because the project installs CUDA builds of PyTorch.
- **NVIDIA GPU** with a driver that supports CUDA 13 (driver 580 or newer). Without a GPU the models run on the CPU, which should work but is untested and much slower.
- **[uv](https://docs.astral.sh/uv/)** to install Python and the dependencies.
- **[FFmpeg](https://ffmpeg.org/)** (`ffmpeg` and `ffprobe` on your `PATH`).
- A microphone. Headphones are recommended, so the mic doesn't pick up the music.
- About 6 GB of disk space for the dependencies and models.

## Installation

1. Install the tools, if you don't have them yet:

   ```powershell
   winget install astral-sh.uv
   winget install Gyan.FFmpeg
   ```

   Open a new terminal afterwards, so both are on your `PATH`.

2. Clone the repository and install the dependencies:

   ```powershell
   git clone https://github.com/schmxtz/karaoke.git
   cd karaoke
   uv sync
   ```

   `uv sync` downloads Python 3.12 if needed, creates a `.venv` and installs everything, including the CUDA build of PyTorch (a few GB).

3. Start the app:

   ```powershell
   uv run python -m karaoke sing
   ```

On first use the app downloads its models: the vocal separation model, and the word-alignment model (about 1.2 GB). Both are cached under `%LOCALAPPDATA%\karaoke\models`.

## Usage

Open a song with **Open song…** or drag an audio file (MP3, FLAC, WAV, M4A, OGG, ...) onto the window. You can also pass it on the command line:

```powershell
uv run python -m karaoke sing "C:\path\to\song.mp3"
```

The first time a song is opened, it is prepared. This takes about a minute with a GPU: separating the vocals, extracting the melody, looking up the lyrics and timing the words. The results are cached, so after that the song opens in about a second.

| Control | What it does |
| --- | --- |
| Space / **Play** | Play or pause |
| **Restart** | Back to the start, resets the score |
| **Snap to notes** | Note bars vs. raw pitch curve |
| **Music** / **Guide vocal** | Volume of the instrumental and of the original vocals |
| **Voice offset** | Shift your voice line if it looks early or late against the melody |
| **Mic sensitivity** | Raise it if your singing isn't picked up; lower it if background noise draws a line. The white mark on the mic meter is the threshold. |
| **Lyrics offset** | Shift the lyrics if they're early or late. Saved per song. |

### Separating stems only

To only split songs into vocals and instrumental, without opening the app:

```powershell
uv run python -m karaoke separate song1.mp3 song2.mp3 -o output
```

This writes `vocals.flac` and `instrumental.flac` (24-bit) into one folder per song.

## Where things are stored

| Path | Contents |
| --- | --- |
| `%LOCALAPPDATA%\karaoke\library\<song>` | Stems, melody (`melody.npz`), lyrics with word timings and furigana (`lyrics.json`) |
| `%LOCALAPPDATA%\karaoke\models` | Downloaded models |
| Registry: `HKCU\Software\karaoke\karaoke` | Settings (mic sensitivity, snap to notes) |

To prepare a song again from scratch, delete its library folder.

## Troubleshooting

- **`karaoke.exe` is blocked by Windows ("Smart App Control" or "Application Control").** The launcher that `uv` generates isn't signed. Use `uv run python -m karaoke ...` (or `.venv\Scripts\python.exe -m karaoke ...`) instead of `uv run karaoke`.
- **The voice line is early or late.** Adjust **Voice offset**. The app already compensates for the latency the audio driver reports, but some devices report it inaccurately.
- **Lyrics are missing.** LRCLIB had no synced lyrics for the song, or the timings didn't match your recording. The app looks the song up by the file's tags, otherwise by an `Artist - Title` file name, so renaming the file can help. Songs without lyrics are looked up again after a week.
- **Your voice isn't picked up.** Check the mic meter: it should turn green while you sing. If it doesn't, raise **Mic sensitivity**, or check the default recording device in Windows.

## Known limitations

- Word timing uses an English model. For other languages the highlight is spread evenly across each line, so it's less precise.
- Live pitch tracking can miss very breathy or whispered singing.
- Songs where several voices sing different notes at once give a less clean melody.
- No seek bar yet.

## Project layout

```
src/karaoke/
  cli.py         command line (sing / separate)
  app.py         Qt window: pitch view, lyrics view, controls
  engine.py      audio playback + mic capture (one duplex stream), scoring
  separation.py  vocal/instrumental separation
  pitch.py       melody extraction (CREPE), live pitch (YIN), note snapping
  library.py     song preparation pipeline and cache
  lyrics.py      LRCLIB lookup, LRC parsing, offset detection
  align.py       word timing (wav2vec2 CTC forced alignment)
  furigana.py    Japanese readings
```

## Acknowledgements

- [Mel-Band RoFormer vocal model](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model) by Kimberley Jensen, run with [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
- [torchcrepe](https://github.com/maxrmorrison/torchcrepe)
- [LRCLIB](https://lrclib.net) for synced lyrics
- [wav2vec2](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self) by Meta AI, via Hugging Face Transformers
- [fugashi](https://github.com/polm/fugashi) and [unidic-lite](https://github.com/polm/unidic-lite)
