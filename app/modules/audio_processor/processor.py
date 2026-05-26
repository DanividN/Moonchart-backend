import os
import numpy as np
import librosa
import pretty_midi
import structlog
from typing import Dict, List, Tuple, Any

logger = structlog.get_logger()

# Clone Hero / Rock Band MIDI note standard mappings
NOTE_MAPPINGS = {
    "PART GUITAR": {
        "EXPERT": {"G": 96, "R": 97, "Y": 98, "B": 99, "O": 100},
        "HARD": {"G": 84, "R": 85, "Y": 86, "B": 87, "O": 88},
        "MEDIUM": {"G": 72, "R": 73, "Y": 74, "B": 75},
        "EASY": {"G": 60, "R": 61, "Y": 62}
    },
    "PART BASS": {
        "EXPERT": {"G": 96, "R": 97, "Y": 98, "B": 99, "O": 100},
        "HARD": {"G": 84, "R": 85, "Y": 86, "B": 87, "O": 88},
        "MEDIUM": {"G": 72, "R": 73, "Y": 74, "B": 75},
        "EASY": {"G": 60, "R": 61, "Y": 62}
    },
    "PART DRUMS": {
        "EXPERT": {"KICK": 96, "SNARE": 97, "HIHAT": 98, "TOM1": 99, "CYMBAL": 100}
    },
    "PART VOCALS": {
        "EXPERT": {"G": 96, "R": 97, "Y": 98, "B": 99, "O": 100},
        "HARD": {"G": 84, "R": 85, "Y": 86, "B": 87, "O": 88},
        "MEDIUM": {"G": 72, "R": 73, "Y": 74, "B": 75},
        "EASY": {"G": 60, "R": 61, "Y": 62}
    }
}

class AudioProcessor:
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Loads audio file safely using librosa."""
        logger.info("Loading audio file", path=file_path)
        y, sr = librosa.load(file_path, sr=self.sr)
        return y, sr

    def analyze_tempo(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Detects base BPM, dynamic tempo map and beat alignments.
        Uses librosa beat tracking.
        """
        logger.info("Analyzing tempo and beats")
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        
        # Estimate global tempo
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        beat_times = librosa.frames_to_time(beats, sr=self.sr)
        
        # Flat BPM conversion for safety
        bpm_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        
        logger.info("Tempo detection completed", bpm=bpm_val, beat_count=len(beat_times))
        return {
            "bpm": bpm_val,
            "beat_times": beat_times.tolist(),
            "beat_frames": beats.tolist()
        }

    def detect_notes_and_onsets(self, y: np.ndarray, instrument: str = "guitar") -> List[float]:
        """
        Detects note onsets (transients) using tailored DSP filters and peak picking for each instrument stem.
        """
        logger.info("Detecting onsets/transients with stem-specific DSP parameters", instrument=instrument)
        
        if instrument == "bass":
            # Bass: low-frequency onset detection
            # Compute STFT and isolate frequencies under 250 Hz
            stft = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=self.sr)
            low_freqs_mask = freqs <= 250
            if np.any(low_freqs_mask):
                onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(stft[low_freqs_mask, :], ref=np.max), sr=self.sr)
            else:
                onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
            # Bass notes typically have slower attack, increase post_max and wait to prevent double triggers
            peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr, wait=12, pre_max=6, post_max=6, pre_avg=12, post_avg=12, delta=0.04)
        
        elif instrument == "drums":
            # Drums: extremely sharp transients (Kick, Snare, Cymbals)
            onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
            # Use smaller wait and post_max to capture rapid successive drum fills (e.g. rolls)
            peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr, wait=5, pre_max=3, post_max=3, pre_avg=8, post_avg=8, delta=0.08)
            
        elif instrument == "vocals":
            # Vocals: syllable boundaries and vowel onset transitions
            # Smooth energy changes with RMS and onset strength
            onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
            # Vocal phrasing is sparse, use larger wait to avoid overcharting lyrics
            peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr, wait=15, pre_max=8, post_max=8, pre_avg=15, post_avg=15, delta=0.07)
            
        else: # guitar or fallback
            # Guitar: standard chord strums and hopo transitions
            onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
            peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr, wait=8, pre_max=4, post_max=4, pre_avg=10, post_avg=10, delta=0.06)

        onset_times = librosa.frames_to_time(peaks, sr=self.sr)
        return onset_times.tolist()

    def quantize_time(self, time_secs: float, bpm: float, snap_fraction: float = 0.0625, ppq: int = 480) -> float:
        """
        Cuantiza un timestamp en segundos al tick más cercano de la rejilla (snap_fraction).
        Por defecto snap_fraction = 0.0625 equivale a 1/16 (semicorcheas).
        """
        if time_secs < 0:
            return 0.0
        # Convertir tiempo físico (segundos) a beats
        beats = (time_secs * bpm) / 60.0
        # Convertir beats a ticks MIDI
        ticks = beats * ppq
        
        # Tamaño de la rejilla de snap en ticks
        snap_ticks = (snap_fraction * 4) * ppq
        if snap_ticks <= 0:
            return time_secs
            
        # Cuantizar aproximando al tick más cercano de la rejilla
        quantized_tick = round(ticks / snap_ticks) * snap_ticks
        
        # Convertir de vuelta a segundos
        quantized_beats = quantized_tick / ppq
        quantized_secs = (quantized_beats * 60.0) / bpm
        return float(quantized_secs)

    def generate_midi_chart(self, tempo_data: Dict[str, Any], track_onsets: Dict[str, List[float]], output_path: str, snap_fraction: float = 0.0625):
        """
        Genera un archivo MIDI multipista compatible con Clone Hero y Rock Band.
        Aplica cuantización a la cuadrícula y filtrado heurístico para evitar overcharting.
        """
        logger.info("Generating quantized MIDI chart", output_path=output_path, snap_fraction=snap_fraction)
        
        bpm = tempo_data["bpm"]
        # Crear objeto PrettyMIDI
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        
        for track_name, onsets in track_onsets.items():
            if track_name not in NOTE_MAPPINGS:
                continue
                
            # Crear instrumento MIDI para la pista
            instrument = pretty_midi.Instrument(program=0, name=track_name)
            mappings = NOTE_MAPPINGS[track_name]
            
            for difficulty, notes in mappings.items():
                logger.info("Mapping quantized difficulty for track", track=track_name, difficulty=difficulty)
                
                # Coeficiente de reducción para evitar overcharting y simplificar ritmo
                reduction_factors = {
                    "EXPERT": 1.0,
                    "HARD": 0.7,
                    "MEDIUM": 0.4,
                    "EASY": 0.2
                }
                
                factor = reduction_factors.get(difficulty, 1.0)
                
                # Filtrar onsets de acuerdo a la dificultad
                filtered_onsets = [t for i, t in enumerate(onsets) if (i % int(1/factor) == 0) if factor < 1.0] if factor < 1.0 else onsets
                
                available_colors = list(notes.keys())
                
                for t_start_raw in filtered_onsets:
                    # Cuantizar el inicio de la nota a la rejilla rítmica
                    t_start = self.quantize_time(t_start_raw, bpm, snap_fraction)
                    
                    # Asignar color cíclico para simular jugabilidad musical
                    color_idx = int(t_start * 10) % len(available_colors)
                    color = available_colors[color_idx]
                    pitch = notes[color]
                    
                    # Duración estándar corta para las notas de tap/strum (0.12 segundos)
                    t_end = t_start + 0.12
                    
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=t_start,
                        end=t_end
                    )
                    instrument.notes.append(note)
                    
            pm.instruments.append(instrument)
            
        # Escribir archivo MIDI final
        pm.write(output_path)
        logger.info("MIDI chart successfully created and quantized", output_path=output_path)


    def separate_stems_mock(self, file_path: str, output_dir: str) -> Dict[str, str]:
        """
        Extracts vocal, drums, bass, and other components.
        If meta-demucs is installed in the local path, it dynamically runs neural separation.
        Otherwise, it falls back to a high-fidelity FFT brick-wall frequency filter.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        stems = {
            "vocals": os.path.join(output_dir, "vocals.wav"),
            "drums": os.path.join(output_dir, "drums.wav"),
            "bass": os.path.join(output_dir, "bass.wav"),
            "guitar": os.path.join(output_dir, "guitar.wav")
        }
        
        import shutil
        import subprocess
        
        # Detect if Demucs CLI tool is installed locally
        demucs_path = shutil.which("demucs")
        if demucs_path:
            logger.info("Demucs CLI detected locally! Running real neural network Music Source Separation...", command=demucs_path)
            try:
                # Run Demucs CLI: model htdemucs (Hybrid Transformer)
                # Save into output_dir and enforce simple output filenames
                cmd = [
                    demucs_path,
                    "-n", "htdemucs",
                    "-o", output_dir,
                    "--filename", "{track}.{ext}",
                    file_path
                ]
                logger.info("Executing Demucs command", cmd=" ".join(cmd))
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Demucs nests output files in: {output_dir}/htdemucs/{input_filename_without_ext}/{stem}.wav
                filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
                demucs_nested_dir = os.path.join(output_dir, "htdemucs", filename_without_ext)
                
                if os.path.exists(demucs_nested_dir):
                    logger.info("Demucs completed successfully. Moving files to align with API expectation...")
                    import shutil as file_shutil
                    for stem_name in ["vocals", "drums", "bass", "other"]:
                        src = os.path.join(demucs_nested_dir, f"{stem_name}.wav")
                        # Map 'other' to 'guitar' for Clone Hero charts standard compatibility
                        dest_name = "guitar" if stem_name == "other" else stem_name
                        dest = os.path.join(output_dir, f"{dest_name}.wav")
                        if os.path.exists(src):
                            if os.path.exists(dest):
                                os.remove(dest)
                            file_shutil.move(src, dest)
                            
                    logger.info("Neural separation stems successfully loaded and aligned!")
                    return stems
            except Exception as e:
                logger.error("Failed to run local Demucs command, falling back to high-fidelity FFT", error=str(e))

        logger.info("Using high-fidelity FFT brick-wall filters fallback", file_path=file_path)
        y, sr = self.load_audio(file_path)
        n = len(y)
        
        import soundfile as sf
        
        # 1. BASS STEM: Low-pass brick-wall filter (Keep only < 180 Hz)
        logger.info("Generating high-fidelity Bass stem (< 180Hz)")
        fft_bass = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(n, d=1.0/sr)
        fft_bass[freqs > 180] = 0.0  # Cut off everything above 180Hz (vocals, guitar, cymbals)
        y_bass = np.fft.irfft(fft_bass, n=n)
        # Normalize
        max_bass = np.max(np.abs(y_bass))
        if max_bass > 0:
            y_bass = (y_bass / max_bass) * 0.75
        sf.write(stems["bass"], y_bass, sr)
        
        # 2. VOCALS STEM: Band-pass brick-wall filter (Keep 250 Hz - 3000 Hz)
        logger.info("Generating high-fidelity Vocals stem (250Hz - 3000Hz)")
        fft_vocals = np.fft.rfft(y)
        fft_vocals[freqs < 250] = 0.0   # Cut off bass & kick drum rumble
        fft_vocals[freqs > 3000] = 0.0  # Cut off extreme cymbals & high guitar frequencies
        y_vocals = np.fft.irfft(fft_vocals, n=n)
        # Normalize
        max_vocals = np.max(np.abs(y_vocals))
        if max_vocals > 0:
            y_vocals = (y_vocals / max_vocals) * 0.8
        sf.write(stems["vocals"], y_vocals, sr)
        
        # 3. DRUMS STEM: Band-suppression split (Keep < 120 Hz and > 4500 Hz, suppress vocals/guitars in mid range)
        logger.info("Generating high-fidelity Drums stem (Kick + Cymbals/Snare snap)")
        fft_drums = np.fft.rfft(y)
        # Zero out the vocal/guitar midrange (120 Hz to 4500 Hz)
        fft_drums[(freqs >= 120) & (freqs <= 4500)] = 0.0
        y_drums = np.fft.irfft(fft_drums, n=n)
        # Normalize
        max_drums = np.max(np.abs(y_drums))
        if max_drums > 0:
            y_drums = (y_drums / max_drums) * 0.85
        sf.write(stems["drums"], y_drums, sr)
        
        # 4. GUITAR STEM: Mid-high band-pass with a notch at vocal fundamental frequencies
        logger.info("Generating high-fidelity Guitar stem (180 Hz - 7000 Hz, vocal notch)")
        fft_guitar = np.fft.rfft(y)
        fft_guitar[freqs < 180] = 0.0   # Remove low bass rumble
        fft_guitar[freqs > 7000] = 0.0  # Remove high cymbal sizzle
        # Suppress typical vocal harmonics range (800 Hz - 1800 Hz) to keep the guitar distinct
        fft_guitar[(freqs >= 800) & (freqs <= 1800)] *= 0.15
        y_guitar = np.fft.irfft(fft_guitar, n=n)
        # Normalize
        max_guitar = np.max(np.abs(y_guitar))
        if max_guitar > 0:
            y_guitar = (y_guitar / max_guitar) * 0.78
        sf.write(stems["guitar"], y_guitar, sr)
        
        # Generate song.ogg for Clone Hero folder
        song_ogg_path = os.path.join(output_dir, "song.ogg")
        if not os.path.exists(song_ogg_path):
            logger.info("Converting master track to song.ogg using FFmpeg")
            import subprocess
            try:
                cmd = ["ffmpeg", "-y", "-i", file_path, "-codec:a", "libvorbis", "-qscale:a", "5", song_ogg_path]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                logger.error("Failed to convert to song.ogg via FFmpeg, copying source as fallback", error=str(e))
                import shutil as file_shutil
                file_shutil.copy(file_path, song_ogg_path)

        logger.info("High-fidelity FFT stem separation completed successfully")
        return stems
