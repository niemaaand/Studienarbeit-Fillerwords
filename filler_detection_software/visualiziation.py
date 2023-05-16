import threading
from tkinter import *

import numpy
import pyaudio
import torchaudio
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from utils.filler_occurence import FillerOccurrence


class Visualizer:
    def __init__(self, fillers, wav_path):
        self._fillers: list = fillers
        self._wav_path: str = wav_path
        self._sample_rate = 16000
        self._current_filler: int = 0
        self._s_displayed: float = 10
        self._duration: float = 0
        self._btn_show_next_filler: Button
        self._btn_play_current_filler: Button
        self._slider_position: Slider
        self._wav_sig: numpy.ndarray
        self._currently_playing_audio_lock = threading.Lock()
        self._stop_playing_audio: bool = True
        self._pause_playing_audio: bool = True
        self._show_fillers_offset = 0

    def get_plot(self) -> (Figure, Axes):
        wav_torch, self._sample_rate = torchaudio.load(self._wav_path)
        self._wav_sig = wav_torch[0].cpu().detach().numpy()
        wav_time = np.linspace(0, len(self._wav_sig) / self._sample_rate, num=len(self._wav_sig))
        self._duration = len(wav_torch[0]) / self._sample_rate

        fig: Figure = plt.figure(figsize=(10, 5), dpi=100)
        plot1: Axes = fig.add_subplot(111)
        fig.subplots_adjust(bottom=0.13)
        actual_plot = plot1.plot(wav_time, self._wav_sig)

        return fig, plot1

    def show_figure(self, window: Tk):
        fig = self.build_figure()
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack()
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()
        canvas.get_tk_widget().pack()

    def build_window(self) -> Tk:
        tk_window = Tk()
        tk_window.title('Visualization of recognized filler words')
        tk_window.geometry("1300x700")
        return tk_window

    def mark_fillers(self, plot):
        for filler_occurrence in self._fillers:
            plot.axvspan(filler_occurrence.start_time, filler_occurrence.end_time, color="yellow", alpha=0.5 * filler_occurrence.probability)
        return plot

    def add_slider(self, fig, plot):
        slider_color = 'Green'
        axis_position = plt.axes([0.2, 0.02, 0.65, 0.03])
        edge_slider = 0.5
        max_slider = self._duration + edge_slider - self._s_displayed
        if max_slider < 0:
            max_slider = edge_slider

        self._slider_position = Slider(axis_position, 'Pos', valmin=-edge_slider, valmax=max_slider, valinit=0.0,
                                       track_color=slider_color)

        def update(val):
            pos = self._slider_position.val
            plot.axis([pos, pos + min(self._duration, self._s_displayed), -1, 1])
            fig.canvas.draw_idle()

            if self._fillers[self._current_filler].start_time - self._show_fillers_offset != pos:
                i: int = self._current_filler
                while True:
                    occurrency: FillerOccurrence = self._fillers[i % len(self._fillers)]
                    previous_occurency: FillerOccurrence = self._fillers[(i - 1) % len(self._fillers)]

                    if (occurrency.start_time >= pos >= previous_occurency.start_time) or \
                        (previous_occurency.start_time > occurrency.start_time and pos <= occurrency.start_time):
                       break
                    elif pos >= self._slider_position.valmax or pos >= self._fillers[-1].start_time:
                        i = 0
                        break
                    #elif (previous_occurency.start_time >= pos and pos <= occurrency.start_time):
                     #   (previous_occurency.start_time > occurrency.start_time)

                    i += 1

                self._current_filler = i % len(self._fillers)

        # update function called using on_changed() function
        self._slider_position.on_changed(update)
        update(0)
        return fig, plot

    def build_figure(self) -> Figure:
        fig1, plot1 = self.get_plot()
        plot1 = self.mark_fillers(plot1)
        fig1, plot1 = self.add_slider(fig1, plot1)
        return fig1

    def __show_btn_next_filler__(self, master):
        def show_next_filler():

            if not (self._slider_position.val + self._s_displayed < self._fillers[self._current_filler].end_time):
                self._current_filler += 1

            if self._current_filler >= len(self._fillers):
                self._current_filler = 0

            new_slider_val = self._fillers[self._current_filler].start_time - self._show_fillers_offset
            if new_slider_val > self._slider_position.valmax:
                new_slider_val = self._slider_position.valmax

            self._slider_position.set_val(new_slider_val)

        self._btn_show_next_filler = Button(master=master, text="Next filler", command=show_next_filler)
        self._btn_show_next_filler.pack()

    def __show_btn_play_current_filler__(self, master):
        def play_current_filler():

            if self._current_filler < 0:
                self._current_filler = 0

            clip = self._wav_sig[int(self._fillers[self._current_filler].start_time * self._sample_rate):
                                 int(self._fillers[self._current_filler].end_time * self._sample_rate)]

            self._currently_playing_audio_lock.acquire()
            self._stop_playing_audio = False
            self._pause_playing_audio = False
            self._currently_playing_audio_lock.release()
            #self.__play_audio__(clip, )
            t = threading.Thread(target=self.__play_audio__, args=[clip, False, self._fillers[self._current_filler].start_time])
            t.start()

        self._btn_play_current_filler = Button(master=master, text="Play current filler", command=play_current_filler)
        self._btn_play_current_filler.pack()

    def __play_audio__(self, audio: numpy.ndarray, slide=False, start_time = 0):
        p = pyaudio.PyAudio()

        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=self._sample_rate,
                        output=True,
                        stream_callback=None)

        cnt: int = 1
        CHUNK = 1024
        d = audio[0: CHUNK]

        while not self._stop_playing_audio and len(d) == CHUNK:
            while not self._pause_playing_audio and len(d) == CHUNK:
                d = audio[cnt * CHUNK: (cnt+1) * CHUNK]
                stream.write(d, num_frames=len(d))
                if slide:
                    self._slider_position.set_val(start_time + (cnt * CHUNK / self._sample_rate))
                cnt += 1

        stream.stop_stream()
        stream.close()

        p.terminate()
        self._currently_playing_audio_lock.acquire()
        self._stop_playing_audio = True
        self._pause_playing_audio = True
        self._currently_playing_audio_lock.release()

    def __show_btn_play_all__(self, master):
        def play_all():

            new_thread = False
            if self._stop_playing_audio and self._pause_playing_audio:
                new_thread = True

            self._currently_playing_audio_lock.acquire()
            self._stop_playing_audio = False
            self._pause_playing_audio = False
            self._currently_playing_audio_lock.release()

            if new_thread:
                t = threading.Thread(target=self.__play_audio__,
                                     args=[self._wav_sig[int(self._slider_position.val * self._sample_rate):
                                                         len(self._wav_sig) - 1], True, self._slider_position.val])
                t.start()

        self._btn_play_all = Button(master=master, text="Play all", command=play_all)
        self._btn_play_all.pack()

        def pause_play():
            self._currently_playing_audio_lock.acquire()
            self._pause_playing_audio = True
            self._currently_playing_audio_lock.release()

        self._btn_pause_play = Button(master=master, text="Pause playing", command=pause_play)
        self._btn_pause_play.pack()

        def stop_play():
            self._currently_playing_audio_lock.acquire()
            self._stop_playing_audio = True
            self._pause_playing_audio = True
            self._currently_playing_audio_lock.release()

        self._btn_stop_play = Button(master=master, text="Stop playing", command=stop_play)
        self._btn_stop_play.pack()

    def visualize_filler_occurrences(self):

        main_window = self.build_window()

        self.show_figure(main_window)
        self.__show_btn_next_filler__(main_window)
        self.__show_btn_play_current_filler__(main_window)
        self.__show_btn_play_all__(main_window)

        main_window.mainloop()


# path = "00302.wav"
path = "a16z_a16z Podcast Teams, Trust, and Object Lessons.wav"



