''' plot_spec_slider.py (c) Yukei Murakami @ JHU, June 2023

Contents:
    - list of emission/absorption lines
    - gui parameters
    - pypeit output parsers
    - matplotlib front end (plotting)
    - QT back end
'''


import sys
import time
import numpy as np
import pandas as pd
import os

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

# from scipy.ndimage import gaussian_filter
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from PyQt5.QtCore import Qt, QObject, QEvent
from PyQt5 import QtWidgets, QtCore, QtGui #pyqt stuff

# ------------
# list of emission/absorption lines
# -------------   
# add '_abs' in the name of absorption lines
em_lines = dict(
    OVI_1034 = 1033.82,
    Lyα_1215 = 1215.24,
    MgII_2799 = 2799.117,
    OII_3727 = 3727,
    Hδ_4101 = 4101,
    G_4306_abs = 4306,
    Hγ_4340 = 4340,
    Hβ_4861 = 4861,
    OIII_4959 = 4959,
    OIII_5007 = 5007,
    Mg_2117_abs = 5177,
    OI_6300 = 6300,
    Hα_6563 = 6562.8,
    NII_6584 = 6584,
    SII_6716 = 6716,
    SII_6731 = 6731,
    CaII_8489 = 8489
)

# ------------
# gui parameters
# -------------   
EMLINE_COLOR = 'orange'
ABSLINE_COLOR = 'violet'

# matplotlib figure/axes parameters
DPI_INITIAL = 101
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 600

POS_MAIN_AXIS = [0.1,0.12,0.85,0.8]
POS_REDSHIFT_SLIDER = [0.48, 0, 0.45, 0.05]
POS_BIN_SLIDER = [0.05, 0, 0.3, 0.05]
FIG_WIDTH = 10
FIG_HEIGHT = 6


# ------------
# pypeit output parsers
# ------------- 
class Pypeit_parser():
    ''' a dummy class to contain pypeit parsers'''
    def __init__(self,):
        return None
    
    def load_fits(spec1d_fits_path):
        '''just an alias to astropy.io.fits.open'''
        return fits.open(spec1d_fits_path)

    def load_txt(spec1d_txt_path):
        '''loads spec1d_*.txt and parses it into pandas DataFrame'''
        df_spec1d = pd.read_csv(spec1d_txt_path,sep='|',encoding="utf-8", skipinitialspace=True)
        df_spec1d = df_spec1d.rename(columns=dict(zip(df_spec1d.columns.values,df_spec1d.columns.str.strip())))
        df_spec1d = df_spec1d.apply(lambda series: series.astype(str).str.strip())
        df_spec1d['s2n'] = df_spec1d['s2n'].astype(float)
        return df_spec1d

    def load_spec(objname,df_spec1d,hdul,cut_edges,extraction_idx=0):
        # identify object by name
        spec1d_idx = df_spec1d[df_spec1d['objname'].eq(objname)].index
        if len(spec1d_idx)>1:
            print("Warning: more than one spectrum with the same object name. Using the first data")
            spec1d_idx = [spec1d_idx[extraction_idx]]   

        # load data
        df = Table(hdul[spec1d_idx[0]+1].data).to_pandas()
        wav = df['BOX_WAVE'].values
        wav_idx = np.arange(wav.shape[0])
        flux = df['BOX_COUNTS'].values
        err = df['BOX_COUNTS_SIG'].values
        sky = df['BOX_COUNTS_SKY'].values
        s = (wav_idx>cut_edges) & (wav_idx < wav_idx.max()-cut_edges) & (wav>0) & (wav<9500) #& (flux>0)
        s = s & (flux <= 10000) & (flux >= -10000)

        return wav[s], flux[s], err[s]#, , sky[s]

# ------------
# matplotlib front end
# -------------   
def draw_emlines(z_guess,ax):
    ylim = ax.get_ylim()
    yscale = lambda y_ax_coord: y_ax_coord*(ylim[1]-ylim[0]) + ylim[0]
    wav_prev = 0
    x_offset = 0
    emline_objects = []
    emline_init = True
    absline_init = True
    for key,_wav in em_lines.items():
        # prep
        em_wav = _wav * (1+z_guess)

        # labels & colors
        if 'abs' in key:
            color = ABSLINE_COLOR
            label, absline_init = ('absorption lines',False) if absline_init else (None,False)
        else:
            color = EMLINE_COLOR
            label, emline_init = ('emission lines',False) if emline_init else (None,False)
                    
        if (em_wav - wav_prev) < 80:
            x_offset += 80
        else:
            x_offset = 0
        wav_prev = em_wav

        # plot
        line = ax.axvline(em_wav,c=color,ls=':',label=label)
        text = ax.text(em_wav-10+x_offset,yscale(0.03),key,
                    ha='right',va='bottom',
                    rotation=90,fontsize=6)
        line.wav_rest = _wav
        emline_objects.append((line,text))
    return emline_objects
    
def bin_spectrum(wav,flux,binning_width,mode='average'):
    if mode == 'sum':
        N_bins = int(np.floor(len(wav)/binning_width))
        binned_wav = []
        binned_flux = []
        for i in range(N_bins):
            __wav = np.nanmean(wav[i*binning_width:i*binning_width+binning_width])
            __flux = flux[i*binning_width:i*binning_width+binning_width].sum()
            binned_wav.append(__wav)
            binned_flux.append(__flux)
        binned_flux = np.asarray(binned_flux)/binning_width
    
    if mode == 'average':
        r = int((binning_width-1)/2)
        binned_wav = wav[r:-r]
        binned_flux = [np.nanmean(flux[i-r:i+r]) for i in np.arange(r,len(wav)-r)]
        
    return binned_wav, binned_flux

def plot_spectrum(wav,flux,fig,ax,title='',
                  # smoothing_sigma=5,
                  binning_width = 9,
                  extraction_idx=None):

    # ------------
    # static part
    # -------------    
    # prepare binned data
    binned_wav, binned_flux = bin_spectrum(wav,flux,binning_width)

    # plot raw data & binned data
    ax.plot(wav,flux,c='lightgray',zorder=0,label='data')
    binned_plot, = ax.step(binned_wav,binned_flux,c='k',where='mid',zorder=1,lw=1,
                             label='Binning applied')

    # prettify
    ylim = ax.get_ylim()
    ax.tick_params(direction='in',labelsize=8)
    ax.set_ylim(ylim[0],ylim[1]*1.)
    ax.set_xlim(wav.min()-100,wav.max()+100)
    ax.set_ylabel('counts',fontsize=12)
    ax.set_xlabel(r'wavlength ($\AA$)',fontsize=11,labelpad=0)
    ax.set_title(title,fontsize=13,x=0.5,y=1.02,va='bottom')
    
    # emission lines
    z_guess_init = 0.1
    emline_objects = draw_emlines(z_guess_init,ax)
    ax.legend(frameon=False,fontsize=8,bbox_to_anchor=(0.5,0.99),
              ncols = 4,
              loc='lower center')
    
    
    # ------------
    # interactive part
    # -------------    
    # functions to update elements
    def update_emline_location(redshift):
        '''update the axvline and text location for emission lines'''
        for line,text in emline_objects:
            wav = line.wav_rest * (1+redshift)
            line.set_xdata([wav])
            text.set_position((wav,text.get_position()[1]))
        fig.canvas.draw()
        
    def update_binning(binning_width):
        '''update the binning of data'''
        binned_wav, binned_flux = bin_spectrum(wav,flux,int(binning_width))
        binned_plot.set_xdata(binned_wav)
        binned_plot.set_ydata(binned_flux)
        fig.canvas.draw()

    # interactive objects
    z_slider_ax = fig.add_axes(POS_REDSHIFT_SLIDER)
    z_slider = Slider(z_slider_ax, 'redshift', 0, 1, valinit=z_guess_init)
    z_slider.on_changed(update_emline_location)
    
    bin_init = 9
    bin_slider_ax = fig.add_axes(POS_BIN_SLIDER)
    bin_slider = Slider(bin_slider_ax, 'bin', 3, 31, valstep=2,valinit=bin_init)
    bin_slider.on_changed(update_binning)
 
    # interactive objects need to be stored in memory to stay active
    # (otherwise Python will garbage-collect them)
    # Return the objects and store them somewhere
    interactive_objects = [z_slider,bin_slider]
    
    return interactive_objects


# ------------
# QT back end
# -------------                         
class SpecPlotWindow(QtWidgets.QMainWindow):
    def __init__(self,wav,flux,title=''):
        # QT app initialization
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.resize(WINDOW_WIDTH,WINDOW_HEIGHT)
        layout = QtWidgets.QVBoxLayout(self._main)
        
        # add matplotlib tool widget and matplotlib canvas to app
        fig = Figure(figsize=(FIG_WIDTH,FIG_HEIGHT),dpi=DPI_INITIAL)
        fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet('background-color: white;')
        layout.addWidget(NavigationToolbar(canvas, self))
        layout.addWidget(canvas)

        # add axis, plot
        self.ax = canvas.figure.add_axes(POS_MAIN_AXIS)
        self.sliders = plot_spectrum(wav,flux,fig,self.ax,title=title,binning_width=10)

        # Connect events to the custom handler
        self.figure = fig
        self.canvas = canvas
        self.current_dpi = DPI_INITIAL
        self.dpi_factor = 1
        self.current_width = self.canvas.size().width()
        self.current_height = self.canvas.size().height()
        
        self.installEventFilter(self)
            
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize and obj is self:
            self.update_figure()
        elif event.type() == QEvent.KeyPress and obj is self:
            if event.key() == Qt.Key_Q:
                self.close()
        return super().eventFilter(obj, event)

    def update_figure(self):
        # get an actual "current" dpi (which could be different from stored value due to automatic scaling)
        current_dpi = self.figure.get_dpi()
        
        # get canvas size
        new_width = self.canvas.size().width()
        new_height = self.canvas.size().height()   
        
        # check if matplotlib automatically rescaled dpi to account for screen scaling
        # this is probably an issue unique to Mac OSX
        if current_dpi != self.current_dpi:
            self.dpi_factor = int(current_dpi / (new_width/FIG_WIDTH))
            if self.dpi_factor <= 0:
                self.dpi_factor = 1
            
        # Calculate the DPI based on the window size
        if new_width/new_height >= FIG_WIDTH/FIG_HEIGHT:
            dpi = (new_width/FIG_WIDTH) * self.dpi_factor
            self.figure.set_dpi(dpi)
            self.figure.set_size_inches(FIG_WIDTH,new_height/dpi * self.dpi_factor)
        else:
            dpi = (new_height/FIG_HEIGHT) * self.dpi_factor
            self.figure.set_dpi(dpi)
            self.figure.set_size_inches(new_width/dpi * self.dpi_factor,FIG_HEIGHT)   
            
        # save data
        self.current_width = new_width
        self.current_height = new_height
        self.current_dpi = dpi
        
        self.canvas.draw()
        self.update()
        
def launch_window(wav,flux,title='',*QApp_args):
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(list(QApp_args))
    app = SpecPlotWindow(wav,flux,title=title)
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()    