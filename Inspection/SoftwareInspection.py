import datetime
import glob
import os


def performance_comparison_pydub_librosa(wav_path: str):
    import librosa, subprocess
    from pydub import AudioSegment
    import soundfile as sf

    files = glob.glob(os.path.join(wav_path, "**/*.wav"))
    tar_folder = os.path.join(wav_path, "TEMP_PerformanceTest")

    if os.path.exists(tar_folder):
        raise FileExistsError

    ffmpeg_path = os.path.join(tar_folder, "ffmpeg")
    pydub_path = os.path.join(tar_folder, "pydub")
    librosa_path = os.path.join(tar_folder, "librosa")
    cnt: int = 0

    os.mkdir(tar_folder)
    os.mkdir(ffmpeg_path)
    os.mkdir(pydub_path)
    os.mkdir(librosa_path)

    t0 = datetime.datetime.now()
    for f in files:
        dur = librosa.get_duration(path=f)
    t1 = datetime.datetime.now()
    print("Read duration Librosa: {}".format(t1 - t0))

    t0 = datetime.datetime.now()
    for f in files:
        dur = AudioSegment.from_wav(f).duration_seconds
    t1 = datetime.datetime.now()
    print("Read duration PyDub: {}".format(t1 - t0))

    def cut_librosa(cache=False):
        if cache:
            files_cached = dict()
            for f in files:
                files_cached[f], sr = librosa.load(f, sr=16000)

        cnt = 0
        t0 = datetime.datetime.now()
        for f in files:

            if cache:
                audio = files_cached[f]
            else:
                audio, sr = librosa.load(f, sr=16000)

            tar_filepath = os.path.join(librosa_path, "{}.wav".format(cnt))

            block = audio[int(16000 * 0.1): int(16000 * 2.1)]
            sf.write(tar_filepath, block, 16000)
            cnt += 1
        t1 = datetime.datetime.now()

        print("Cutting librosa/soundfile (Cache: {}) : {}".format(cache, t1 - t0))

    cut_librosa(False)
    cut_librosa(True)

    t0 = datetime.datetime.now()
    for f in files:
        tar_filepath = os.path.join(ffmpeg_path, "{}.wav".format(cnt))
        cut_cmd = ["ffmpeg", "-i", f, "-ss", str(0.1), "-to", str(2.1), tar_filepath]
        completed_process = subprocess.run(cut_cmd, stdout=subprocess.PIPE)
        cnt += 1
    t1 = datetime.datetime.now()
    print("Cutting ffmpeg: {}".format(t1 - t0))

    def cut_pydub(cache=False):
        if cache:
            files_cached = dict()
            for f in files:
                files_cached[f] = AudioSegment.from_wav(f)

        cnt = 0
        t0 = datetime.datetime.now()
        for f in files:
            tar_filepath = os.path.join(pydub_path, "{}.wav".format(cnt))

            if cache:
                new_clip = files_cached[f]
            else:
                new_clip = AudioSegment.from_wav(f)

            new_clip = new_clip[100:2100]
            new_clip.export(tar_filepath, format="wav")  # Exports to a wav file in the current path.
            cnt += 1
        t1 = datetime.datetime.now()
        print("Cutting PyDub (Cache: {}) : {}".format(cache, t1 - t0))

    cut_pydub(False)
    cut_pydub(True)
