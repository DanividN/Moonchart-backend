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

    def detect_notes_and_onsets(self, y: np.ndarray, instrument: str = "guitar", options: dict = None) -> Dict[str, List[Any]]:
        """
        Detects note onsets (transients) using tailored DSP filters.
        For drums, applies multi-band separation to isolate Kick, Snare, and Hi-Hats.
        """
        logger.info("Detecting onsets/transients with advanced DSP", instrument=instrument)
        from scipy.signal import butter, sosfilt
        
        options = options or {}
        sensitivity = float(options.get("sensitivity", 50.0))
        # sensitivity 0 -> higher thresholds (fewer notes), 100 -> lower thresholds (more notes)
        # map 0-100 to a scale factor for wait/pre/post/delta
        sens_factor = 1.0 - ((sensitivity - 50.0) / 100.0) # 50 = 1.0, 100 = 0.5, 0 = 1.5
        
        onsets_dict = {}
        
        if instrument == "drums":
            # Multi-Band Drum Transcription
            # 1. KICK (< 150 Hz)
            sos_kick = butter(4, 150, 'low', fs=self.sr, output='sos')
            y_kick = sosfilt(sos_kick, y)
            env_kick = librosa.onset.onset_strength(y=y_kick, sr=self.sr)
            peaks_kick = librosa.onset.onset_detect(onset_envelope=env_kick, sr=self.sr, wait=int(8*sens_factor), pre_max=int(4*sens_factor), post_max=int(4*sens_factor), pre_avg=int(8*sens_factor), post_avg=int(8*sens_factor), delta=0.06*sens_factor)
            onsets_dict["kick"] = [{"time": float(t), "pitch": 0.0} for t in librosa.frames_to_time(peaks_kick, sr=self.sr)]
            
            # 2. SNARE (200 - 1000 Hz)
            sos_snare = butter(4, [200, 1000], 'bandpass', fs=self.sr, output='sos')
            y_snare = sosfilt(sos_snare, y)
            env_snare = librosa.onset.onset_strength(y=y_snare, sr=self.sr)
            peaks_snare = librosa.onset.onset_detect(onset_envelope=env_snare, sr=self.sr, wait=int(10*sens_factor), pre_max=int(4*sens_factor), post_max=int(4*sens_factor), pre_avg=int(10*sens_factor), post_avg=int(10*sens_factor), delta=0.07*sens_factor)
            onsets_dict["snare"] = [{"time": float(t), "pitch": 0.0} for t in librosa.frames_to_time(peaks_snare, sr=self.sr)]
            
            # 3. HI-HAT / CYMBALS (> 5000 Hz)
            sos_hh = butter(4, 5000, 'high', fs=self.sr, output='sos')
            y_hh = sosfilt(sos_hh, y)
            env_hh = librosa.onset.onset_strength(y=y_hh, sr=self.sr)
            peaks_hh = librosa.onset.onset_detect(onset_envelope=env_hh, sr=self.sr, wait=int(4*sens_factor), pre_max=int(2*sens_factor), post_max=int(2*sens_factor), pre_avg=int(6*sens_factor), post_avg=int(6*sens_factor), delta=0.08*sens_factor)
            onsets_dict["hihat"] = [{"time": float(t), "pitch": 0.0} for t in librosa.frames_to_time(peaks_hh, sr=self.sr)]
            
        elif instrument == "bass":
            y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0, 5.0))
            onset_env = librosa.onset.onset_strength(y=y_percussive, sr=self.sr)
            peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr, wait=int(12*sens_factor), pre_max=int(6*sens_factor), post_max=int(6*sens_factor), pre_avg=int(12*sens_factor), post_avg=int(12*sens_factor), delta=0.04*sens_factor)
            
            pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=self.sr)
            onsets_with_pitch = []
            for peak in peaks:
                index = magnitudes[:, peak].argmax()
                pitch = pitches[index, peak]
                onsets_with_pitch.append({"time": float(librosa.frames_to_time(peak, sr=self.sr)), "pitch": float(pitch)})
            onsets_dict["default"] = onsets_with_pitch
            
        else: # guitar or vocals
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            onset_env = librosa.onset.onset_strength(y=y_percussive, sr=self.sr)
            peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr, wait=int(8*sens_factor), pre_max=int(4*sens_factor), post_max=int(4*sens_factor), pre_avg=int(10*sens_factor), post_avg=int(10*sens_factor), delta=0.05*sens_factor)
            
            pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=self.sr)
            onsets_with_pitch = []
            for peak in peaks:
                index = magnitudes[:, peak].argmax()
                pitch = pitches[index, peak]
                onsets_with_pitch.append({"time": float(librosa.frames_to_time(peak, sr=self.sr)), "pitch": float(pitch)})
            onsets_dict["default"] = onsets_with_pitch

        return onsets_dict

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

    def generate_midi_chart(self, tempo_data: Dict[str, Any], track_onsets: Dict[str, Dict[str, List[Any]]], output_path: str, snap_fraction: float = 0.0625, options: dict = None):
        """
        Genera un archivo MIDI multipista. Mapea categorías de batería a sus verdaderos MIDI pitches.
        """
        logger.info("Generating quantized MIDI chart with DSP mappings", output_path=output_path)
        options = options or {}
        bpm = float(options.get("bpm")) if options.get("bpm") else tempo_data["bpm"]
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        
        for track_name, onset_cats in track_onsets.items():
            if track_name not in NOTE_MAPPINGS: continue
                
            instrument = pretty_midi.Instrument(program=0, name=track_name)
            mappings = NOTE_MAPPINGS[track_name].get("EXPERT", {})
            
            # Map DSP categories to MIDI pitches
            category_to_pitch = {}
            if track_name == "PART DRUMS":
                category_to_pitch = {
                    "kick": mappings.get("KICK", 96),
                    "snare": mappings.get("SNARE", 97),
                    "hihat": mappings.get("HIHAT", 98),
                    "default": mappings.get("CYMBAL", 100)
                }
            else:
                category_to_pitch = {"default": mappings.get("G", 96)}
                
            for cat, onsets in onset_cats.items():
                pitch = category_to_pitch.get(cat, mappings.get("G", 96))
                for onset in onsets:
                    t_start_raw = onset["time"] if isinstance(onset, dict) else onset
                    t_start = self.quantize_time(t_start_raw, bpm, snap_fraction)
                    t_end = t_start + 0.12
                    
                    # For non-drums, randomly shift color just for visual variety in MIDI
                    if track_name != "PART DRUMS":
                        colors = list(mappings.values())
                        pitch = colors[int(t_start * 10) % len(colors)]
                        
                    note = pretty_midi.Note(velocity=100, pitch=pitch, start=t_start, end=t_end)
                    instrument.notes.append(note)
                    
            pm.instruments.append(instrument)
            
        # Escribir archivo MIDI final
        pm.write(output_path)
        logger.info("MIDI chart successfully created and quantized", output_path=output_path)

    def generate_frontend_notes(self, tempo_data: Dict[str, Any], track_onsets: Dict[str, Dict[str, List[Any]]], target_instrument: str, ticks_per_beat: int = 192, options: dict = None) -> List[Dict[str, Any]]:
        """
        Generates a JSON-serializable list of AI notes for the frontend using advanced rhythmic heuristics and Multi-Band DSP.
        Supports Chords, Sustains, HOPOs, Taps, Open Notes, and complete Drum kit mapping.
        """
        logger.info("Generating advanced AI JSON notes payload with Multi-Band DSP", instrument=target_instrument)
        options = options or {}
        # Prioritize frontend's static BPM to keep quantization perfectly aligned with the grid
        bpm = float(options.get("bpm")) if options.get("bpm") else tempo_data.get("bpm", 120.0)
        complexity = float(options.get("complexity", 50.0))
        
        midi_track_map = {
            "guitar": "PART GUITAR", "bass": "PART BASS", 
            "drums": "PART DRUMS", "vocals": "PART VOCALS"
        }
        track_name = midi_track_map.get(target_instrument, "PART GUITAR")
        onset_cats = track_onsets.get(track_name, {"default": []})
        
        # Flatten and sort all onsets to iterate chronologically
        flat_onsets = []
        for cat, onsets in onset_cats.items():
            for onset in onsets:
                flat_onsets.append((onset, cat))
        
        flat_onsets.sort(key=lambda x: x[0]["time"] if isinstance(x[0], dict) else x[0])
        
        frontend_notes = []
        last_tick = -9999
        
        for i, onset_item in enumerate(flat_onsets):
            cat = onset_item[1]
            onset_data = onset_item[0]
            if isinstance(onset_data, dict):
                t_secs = onset_data["time"]
                pitch_hz = onset_data.get("pitch", 0.0)
            else:
                t_secs = onset_data
                pitch_hz = 0.0
                
            # Calculate absolute tick
            beat = (t_secs * bpm) / 60.0
            raw_tick = int(round(beat * ticks_per_beat))
            
            # Snap to 1/16th notes (48 ticks at 192 ticks per beat)
            snap_ticks = 48
            tick = int(round(raw_tick / snap_ticks)) * snap_ticks
            
            # Calculate deltas to determine note speed/density
            delta_ticks_prev = tick - last_tick if last_tick != -9999 else 9999
            
            
            next_t_secs = flat_onsets[i+1][0] if i < len(flat_onsets) - 1 else flat_onsets[-1][0]
            if isinstance(next_t_secs, dict):
                next_t_secs = next_t_secs["time"]
                
            next_raw_tick = int(round(((next_t_secs * bpm) / 60.0) * ticks_per_beat))
            delta_ticks_next = (int(round(next_raw_tick / snap_ticks)) * snap_ticks) - tick if i < len(flat_onsets) - 1 else 9999
            
            beat_index = tick / ticks_per_beat
            measure_beat = beat_index % 4.0 # 0.0 to 3.99 in 4/4 time
            
            notes_to_add = []
            confidence = 0.85
            
            if target_instrument == "drums":
                # --- DRUMS TRUE DSP MAPPING ---
                if cat == "kick":
                    notes_to_add.append((0, "kick_pedal", "Bombo (Detección DSP Baja Frecuencia)", 0))
                elif cat == "snare":
                    notes_to_add.append((1, "strum", "Caja/Tarola (Detección DSP Frecuencia Media)", 0))
                elif cat == "hihat":
                    notes_to_add.append((2, "strum", "Hi-Hat/Platillo (Detección DSP Alta Frecuencia)", 0))
                else:
                    # Fallback for crashes/toms based on density
                    if delta_ticks_prev <= 48 and delta_ticks_next > 96:
                        notes_to_add.append((4, "strum", "Crash Cymbal (Heurística de Remate)", 0))
                    else:
                        notes_to_add.append((3, "strum", "Tom Medio (Fallback DSP)", 0))
            else:
                # Skip exact duplicates on same tick for guitar to avoid overlapping
                if delta_ticks_prev == 0:
                    continue
                    
                # --- GUITAR / BASS ADVANCED MAPPING ---
                if pitch_hz > 0:
                    if pitch_hz < 150: lane = 0
                    elif pitch_hz < 300: lane = 1
                    elif pitch_hz < 600: lane = 2
                    elif pitch_hz < 1000: lane = 3
                    else: lane = 4
                else:
                    lane = int((tick / snap_ticks * 3 + 1) % 5)
                    
                note_type = "strum"
                duration = 0
                reason = f"Nota de Rasgueo (Pitch: {int(pitch_hz)}Hz)" if pitch_hz > 0 else "Nota de Rasgueo Básica"
                
                # 1. HOPOs & Taps (Velocidad)
                if delta_ticks_prev <= 48 and complexity >= 30:
                    note_type = "hopo"
                    reason = "HOPO (Ligado Rápido)"
                    if delta_ticks_prev <= 24 and complexity >= 70:
                        note_type = "tap"
                        reason = "Tap/Slider (Trino o Shredding)"
                
                # 2. Sustains (Notas Largas)
                if delta_ticks_next >= 192 and (measure_beat < 0.25 or measure_beat % 1.0 < 0.25) and complexity >= 40:
                    # Larga duración hasta la siguiente nota
                    duration = min(delta_ticks_next - 48, 192 * 2) # Máximo 2 compases
                    if duration > 0:
                        reason = "Sustain (Vibrato o nota final)"
                
                notes_to_add.append((lane, note_type, reason, duration))
                
                # 3. Acordes (Chords)
                if (measure_beat < 0.25) and note_type == "strum" and complexity >= 50:
                    chord_lane = (lane + 1) % 5
                    notes_to_add.append((chord_lane, note_type, "Acorde Doble (Acento de compás)", duration))
                    
                    if measure_beat == 0 and beat_index > 16 and complexity >= 80:
                        chord_lane_3 = (lane + 3) % 5
                        notes_to_add.append((chord_lane_3, note_type, "Acorde Triple (Power Chord explosivo)", duration))
                
                # 4. Notas Abiertas (Open Notes - Metalcore)
                if target_instrument == "guitar" and delta_ticks_prev <= 96 and delta_ticks_next <= 96 and measure_beat % 0.5 < 0.1:
                    if len(notes_to_add) > 0 and note_type == "strum":
                        notes_to_add[0] = (7, "open", "Open Note (Palm Mute de Metalcore)", notes_to_add[0][3])
            
            for (l, n_type, r, d) in notes_to_add:
                # Deduplicate exact lane/tick
                if not any(n["tick"] == tick and n["lane"] == l for n in frontend_notes):
                    frontend_notes.append({
                        "id": f"ai-gen-{int(beat)}-{tick}-{l}-{n_type}",
                        "tick": tick,
                        "lane": l,
                        "confidence": confidence,
                        "reason": r,
                        "duration": d,
                        "type": n_type
                    })
            last_tick = tick
                
        logger.info("Generated advanced notes successfully", count=len(frontend_notes))
        return frontend_notes


    def separate_stems_mock(self, file_path: str, output_dir: str, target_instrument: str = None) -> Dict[str, str]:
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
            # Run Demucs CLI: model htdemucs (Hybrid Transformer)
            # Save into output_dir and enforce simple output filenames
            cmd = [
                demucs_path,
                "-n", "htdemucs",
                "-o", output_dir,
                file_path
            ]
            logger.info("Executing Demucs command", cmd=" ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Demucs process failed with exit code {result.returncode}:\n{result.stderr[-500:]}")
            
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
            else:
                raise Exception(f"Demucs completed but output directory {demucs_nested_dir} not found.")
        else:
            raise Exception("Demucs CLI not found. Neural separation is strictly required and FFT fallback has been disabled.")

        # Generate song.ogg for Clone Hero folder
        song_ogg_path = os.path.join(output_dir, "song.ogg")
        if not os.path.exists(song_ogg_path):
            logger.info("Converting master track to song.ogg using FFmpeg")
            cmd = ["ffmpeg", "-y", "-i", file_path, "-codec:a", "libvorbis", "-qscale:a", "5", song_ogg_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
        return stems
