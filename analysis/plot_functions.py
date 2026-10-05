import yt
import numpy as npy
from scipy.interpolate import interp1d
#from yt.mods import *
import pylab
import unyt as u
import h5py
from matplotlib import colormaps as clmp
import matplotlib.colors as colors
from matplotlib.colors import LogNorm,Normalize
import matplotlib.pyplot as plt
from params import mu_mol,muJy,mJy,arcmin,arcsec,freq_array,eta_B,L_turb,DATASET_NAME_DENSITY,DATASET_NAME_X,DATASET_NAME_Y,DATASET_NAME_Z,DATASET_NAME_VCONV,DATASET_NAME_TEMP,tracer_filename_base1,tracer_filename_base2,Secondary




def plot_map(name,map,extent,xlabel,ylabel,vmin,vmax,cmap,label):
    map = npy.where(map  < 0.01*vmin, vmin*0.01, map)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.imshow(map,extent=extent, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, origin='lower')
    plt.colorbar(label=label)  # カラーバーを追加
    plt.savefig(name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.clf()



def plot_map_smooth(name,map,extent,xlabel,ylabel,vmin,vmax,cmap,label):
    map = npy.where(map  < 0.01*vmin, vmin*0.01, map)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(r"{:s}".format(xlabel))
    ax.set_ylabel(r"{:s}".format(ylabel))
    plt.imshow(map,extent=extent, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, origin='lower',interpolation="bicubic")
    plt.colorbar(label=r"{:s}".format(label))  # カラーバーを追加
    plt.savefig(name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.clf()


def plot_map_lin(name,map,extent,xlabel,ylabel,vmin,vmax,cmap,label):
    map = npy.where(map  < vmin-10.0, vmin-1.0, map)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(r"{:s}".format(xlabel))
    ax.set_ylabel(r"{:s}".format(ylabel))
    plt.imshow(map, cmap=cmap, extent=extent, origin='lower',norm = Normalize(vmin = vmin, vmax = vmax))
    plt.colorbar(label=label)  # カラーバーを追加
    plt.savefig(name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.clf()




def plot_hist_linx(name,data,xmin,xmax,bin,col,xlabel,logy = False,ylabel = None):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(xlabel)
    if logy==True:
        ax.set_yscale("log")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlim(xmin, xmax)
    ax.hist(data, bins=npy.linspace(xmin,xmax,bin), color=col, alpha=1.0)
    fig.savefig(name, bbox_inches="tight", dpi = 200)
    plt.close(fig)
    plt.clf()


def plot_hist_logx(name,data,xmin,xmax,bin,col,xlabel):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(r"{:s}".format(xlabel))
    ax.set_xscale("log")
    ax.set_xlim(npy.power(10.0,xmin), npy.power(10.0,xmax))
    ax.hist(data, bins=npy.logspace(xmin,xmax,bin), color=col, alpha=1.0)
    ax.axvline(npy.mean(data), lw = 1.5, ls = "--")
    fig.savefig(name, bbox_inches="tight", dpi = 200)
    plt.close(fig)
    plt.clf()

def plot_2hist_logx(name,data1,data2,xmin,xmax,bin,col1,col2,xlabel):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(r"{:s}".format(xlabel))
    ax.set_xscale("log")
    ax.set_xlim(npy.power(10.0,xmin), npy.power(10.0,xmax))
    ax.hist(data1, bins=npy.logspace(xmin,xmax,bin), color=col1, alpha=0.5)
    ax.hist(data2, bins=npy.logspace(xmin,xmax,bin), color=col2, alpha=0.5)
    fig.savefig(name, bbox_inches="tight", dpi = 200)
    plt.close(fig)
    plt.clf()

def plot_2hist_logx_logy(name,data1,data2,xmin,xmax,bin,col1,col2,xlabel):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(r"{:s}".format(xlabel))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(npy.power(10.0,xmin), npy.power(10.0,xmax))
    ax.hist(data1, bins=npy.logspace(xmin,xmax,bin), color=col1, alpha=0.5)
    ax.hist(data2, bins=npy.logspace(xmin,xmax,bin), color=col2, alpha=0.5)
    fig.savefig(name, bbox_inches="tight", dpi = 200)
    plt.close(fig)
    plt.clf()
 


def plot_tracer_scatter_loglog(name,x_data,y_data,xmin,xmax,ymin,ymax,xlabel,ylabel,col,alpha):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(r"{:s}".format(xlabel))
    ax.set_ylabel(r"{:s}".format(ylabel))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(x_data,y_data, s = 0.1, c= col, alpha = alpha)
    fig.savefig(name, bbox_inches="tight", dpi = 200)
    plt.close(fig)
    plt.clf()


def plot_tracer_scatter(name,x_data,y_data,xmin,xmax,ymin,ymax,xlabel,ylabel,col):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(r"{:s}".format(xlabel))
    ax.set_ylabel(r"{:s}".format(ylabel))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.scatter(x_data,y_data, s = 0.1, c= col)
    fig.savefig(name, bbox_inches="tight", dpi = 200)
    plt.close(fig)
    plt.clf()


def plot_point_point_fit_log(name,datax,datay,xmin,xmax,ymin,ymax,xlabel,ylabel,logx_min,logy_min,col):
    from scipy.stats import linregress
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter(datax, datay, c = col, s = 10, alpha = 0.5)
    log_x = npy.log10(datax)
    log_y = npy.log10(datay)
    valid_indices = ~npy.isnan(log_x) & ~npy.isnan(log_y) & ~npy.isinf(log_x) & ~npy.isinf(log_y) 
    valid_indices =  (log_y > logy_min) & (log_x > logx_min)
    if npy.any(valid_indices)>0:
        log_x_cut = log_x[valid_indices]
        log_y_cut = log_y[valid_indices]
        ax.scatter(npy.power(10.0, log_x_cut), npy.power(10.0,log_y_cut), c = 'black', s = 10)
        # 線形フィット (logy = a*logx + b)
        slope, intercept, r_value, p_value, std_err = linregress(log_x_cut, log_y_cut)
        fit_line = slope * log_x_cut + intercept
        ax.plot(npy.power(10.0, log_x_cut), npy.power(10.0, fit_line), color="red", label=f"Fit: (a = {slope:.2f}, b = {intercept:.2f})")
    else:
        slope, intercept, r_value = 0,0,0
    ax.grid(linestyle='-', lw = 0.5)
    ax.legend(fontsize = 14.0)
    fig.savefig(name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.clf()
    
    return slope, intercept, r_value