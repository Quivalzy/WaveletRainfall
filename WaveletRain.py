

# Section 1 : Import Library and Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import waveletFunctions as wf
from matplotlib.ticker import ScalarFormatter
from matplotlib.gridspec import GridSpec
# End of Section 1

# Section 2: Data Reading and Processing
# Read Data
data = xr.open_dataarray('Modul_3/chirps-v2.0.1981.2015.days_p10.nc')

# Locations for Case Study : Kediri City and Yogyakarta City
kediri = [112.02,-7.82]
yogya = [110.36, -7.81]

# Slice data for desired cities
rainMeanKd = data.sel(time=slice('1981-01-01', '2015-12-31')).sel(lon=kediri[0],lat=kediri[1],method='nearest')
rainMeanYg = data.sel(time=slice('1981-01-01', '2015-12-31')).sel(lon=yogya[0],lat=yogya[1],method='nearest')

# Convert to Numpy Array
rainMeanKd= rainMeanKd.to_numpy()
rainMeanYg = rainMeanYg.to_numpy()

# Create daily frequencies
d_ref = pd.date_range(start='1981-01-01', end='2015-12-31', freq='1D')

# Convert to Pandas Series
rainMeanKd = pd.Series(rainMeanKd, index=d_ref)
rainMeanYg = pd.Series(rainMeanYg, index=d_ref)

# Resample to 3 Months (Seasonal)
rainSeasonKd = rainMeanKd.resample('3M', label='right').mean()
rainSeasonYg = rainMeanYg.resample('3M', label='right').mean()

# Drop one last point to match seasonal resample
rainSeasonKd = rainSeasonKd.drop('2016-01-31')
rainSeasonYg = rainSeasonYg.drop('2016-01-31')

# Normalize to Yearly Rain Rate
VarKd = np.std(rainSeasonKd)
rainYearKd = (rainSeasonKd - np.mean(rainSeasonKd)) / VarKd
VarYg = np.std(rainSeasonYg)
rainYearYg = (rainSeasonYg - np.mean(rainSeasonYg)) / VarYg

# Define times
n1 = len(rainYearKd)
dt1 = 0.25
time1 = np.arange(n1) * dt1 + 1980
n2 = len(rainYearYg)
dt2 = 0.25
time2 = np.arange(n2) * dt2 + 1980
xlim = (1981, 2016)
# End of Section 2

# Section 3: Wavelet Analysis and Visualization
# Define Mother Experiments
mothers = ['DOG', 'PAUL']

for exp in mothers:
    # Kediri
    indata = rainYearKd
    dj = 0.25         
    pad = 0           
    s0 = 2*dt1         
    j1 = 7/dj         
    mother = exp
    lag1 = 0.74      
    # Wavelet Transform
    wave, period, scale, coi = wf.wavelet(indata, dt=dt1, pad=pad, dj=dj, s0=s0, J1=j1, mother=mother)  
    power = (np.abs(wave))**2                                                                       
    global_ws = (np.sum(power,axis=1))/n1                                                            
    signif = wf.wave_signif(VarKd,dt=dt1,scale=scale,sigtest=0,lag1=lag1,mother=mother)  
    sig95 = signif[:,np.newaxis].dot(np.ones(n1)[np.newaxis,:])                            
    sig95 = power/sig95                                                                   
    # Wavelet Spectrum and Significancy
    dof = n1-scale                                                                                      
    global_signif = wf.wave_signif(VarKd,dt=dt1,scale=scale,sigtest=1,lag1=lag1,dof=dof,mother=mother)  
    # Scale Average on 2-8 Year
    avg = np.logical_and(scale>=2,scale<8)   
    if (exp == 'DOG'):                             
        Cdelta = 3.541
    elif (exp == 'PAUL'):
        Cdelta = 1.132                                                        
    scale_avg = scale[:,np.newaxis].dot(np.ones(n1)[np.newaxis,:])         
    scale_avg = power/scale_avg                                          
    scale_avg = dj*dt1/Cdelta*sum(scale_avg[avg,:])                        
    scale_signif = wf.wave_signif(VarKd,dt=dt1,scale=scale,sigtest=2,lag1=lag1,dof=[2,7],mother=mother)
    
    # Data Visualization
    fig = plt.figure(figsize=(9, 10))
    gs = GridSpec(3, 4, hspace=0.4, wspace=0.75)
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=0, hspace=0)
    plt.subplot(gs[0, 0:3])
    plt.plot(time1, rainYearKd, 'red')
    plt.xlim(xlim[:])
    plt.xlabel('Time (year)')
    plt.ylabel('CH (mm)')
    plt.title('a) Seasonal Average Rain - Kediri City')
    #--- Contour plot wavelet power spectrum
    plt3 = plt.subplot(gs[1, 0:3])
    levels = [0, 0.5, 1, 2, 4, 999]
    CS = plt.contourf(time1, period, power, len(levels))  #*** or use 'contour'
    im = plt.contourf(CS, levels=levels, colors=['white','bisque','orange','orangered','darkred'])
    plt.xlabel('Time (year)')
    plt.ylabel('Period (years)')
    plt.title('b) Wavelet Power Spectrum (contours at 0.5,1,2,4\u0000C$^2$)')
    plt.xlim(xlim[:])
    # 95# significance contour, levels at -99 (fake) and 1 (95# signif)
    plt.contour(time1, period, sig95, [-99, 1], colors='k')
    # cone-of-influence, anything "below" is dubious
    ts = time1;
    coi_area = np.concatenate([[np.max(scale)], coi, [np.max(scale)],[np.max(scale)]])
    ts_area = np.concatenate([[ts[0]], ts, [ts[-1]] ,[ts[0]]]);
    L = plt.plot(ts_area,(coi_area),'blue',linewidth=1)
    F = plt.fill(ts_area,(coi_area),'blue',alpha=0.3,hatch="x")
    # format y-scale
    plt3.set_yscale('log', base=2, subs=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(ScalarFormatter())
    plt3.ticklabel_format(axis='y', style='plain')
    plt3.invert_yaxis()
    #--- Plot global wavelet spectrum
    plt4 = plt.subplot(gs[1, -1])
    plt.plot(global_ws, period)
    plt.plot(global_signif, period, '--')
    plt.xlabel('Power (\u0000C$^2$)')
    plt.title('c) Global Wavelet Spectrum')
    plt.xlim([0, 1.25 * np.max(global_ws)])
    # format y-scale
    plt.ylim([np.min(period), np.max(period)])
    plt4.set_yscale('log', base=2, subs=None)
    ax = plt.gca().yaxis
    ax.set_major_formatter(ScalarFormatter())
    plt4.ticklabel_format(axis='y', style='plain')
    plt4.invert_yaxis()
    # --- Plot 2--8 yr scale-average time series
    plt.subplot(gs[2, 0:3])
    plt.plot(time1, scale_avg, 'k')
    plt.plot(xlim,scale_signif+[0,0],'--')
    plt.xlim(xlim[:])
    plt.xlabel('Time (year)')
    plt.ylabel('Avg variance (\u0000C$^2$)')
    plt.title('d) 2-8 yr Scale-average Time Series')
    plt.savefig('Modul_3/'+exp+'_Kediri.png')

    # Yogyakarta
    indata = rainYearYg #data
    # b. transformasi wavelet
    wave, period, scale, coi = wf.wavelet(indata, dt=dt2, pad=pad, dj=dj, s0=s0, J1=j1, mother=mother)  # fungsi wavelet
    power = (np.abs(wave))**2                                                                       # hitung wavelet power spectrum
    global_ws = (np.sum(power,axis=1))/n2                                                            # time-average over all scale (global_ws)
    # c. pengaturan selang kepercayaan
    signif = wf.wave_signif(VarYg,dt=dt2,scale=scale,sigtest=0,lag1=lag1,mother=mother)  # fungsi signifikansi
    sig95 = signif[:,np.newaxis].dot(np.ones(n2)[np.newaxis,:])                            # buat signif --> (J+1)x(N) array
    sig95 = power/sig95                                                                   # ketika ratio power dan signif > 1, power signifikan
    # d. spektrum wavelet dan selang kepercayaan
    dof = n2-scale                                                                                      # the scale corrects for padding at edges
    global_signif = wf.wave_signif(VarYg,dt=dt2,scale=scale,sigtest=1,lag1=lag1,dof=dof,mother=mother)  # fungsi signifikansi untuk global_ws
    # e. scale-average pada rentang periode 2 dan 8
    avg = np.logical_and(scale>=2,scale<8)                                # Logika untuk mengambil rentang skala 2 - 8
    Cdelta = 0.776                                                        # Cdelta untuk MORLET wavelet
    scale_avg = scale[:,np.newaxis].dot(np.ones(n2)[np.newaxis,:])         # buat scale --> (J+1)x(N) array
    scale_avg = power/scale_avg                                           # [Eqn(24)] di referensi
    scale_avg = dj*dt2/Cdelta*sum(scale_avg[avg,:])                        # [Eqn(24)] di referensi
    scale_signif = wf.wave_signif(VarYg,dt=dt2,scale=scale,sigtest=2,lag1=lag1,dof=[2,7],mother=mother)# fungsi signifikansi untuk scale average
    # PLOT
    #--- Plot time series
    fig = plt.figure(figsize=(9, 10))
    gs = GridSpec(3, 4, hspace=0.4, wspace=0.75)
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=0, hspace=0)
    plt.subplot(gs[0, 0:3])
    plt.plot(time2, rainYearYg, 'red')
    plt.xlim(xlim[:])
    plt.xlabel('Time (year)')
    plt.ylabel('CH (mm)')
    plt.title('a) Seasonal Average Rain - Yogyakarta City')
    #--- Contour plot wavelet power spectrum
    plt3 = plt.subplot(gs[1, 0:3])
    levels = [0, 0.5, 1, 2, 4, 999]
    CS = plt.contourf(time2, period, power, len(levels))  #*** or use 'contour'
    im = plt.contourf(CS, levels=levels, colors=['white','bisque','orange','orangered','darkred'])
    plt.xlabel('Time (year)')
    plt.ylabel('Period (years)')
    plt.title('b) Wavelet Power Spectrum (contours at 0.5,1,2,4\u0000C$^2$)')
    plt.xlim(xlim[:])
    # 95# significance contour, levels at -99 (fake) and 1 (95# signif)
    plt.contour(time1, period, sig95, [-99, 1], colors='k')
    # cone-of-influence, anything "below" is dubious
    ts = time2;
    coi_area = np.concatenate([[np.max(scale)], coi, [np.max(scale)],[np.max(scale)]])
    ts_area = np.concatenate([[ts[0]], ts, [ts[-1]] ,[ts[0]]]);
    L = plt.plot(ts_area,(coi_area),'blue',linewidth=1)
    F = plt.fill(ts_area,(coi_area),'blue',alpha=0.3,hatch="x")
    # format y-scale
    plt3.set_yscale('log', base=2, subs=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(ScalarFormatter())
    plt3.ticklabel_format(axis='y', style='plain')
    plt3.invert_yaxis()
    #--- Plot global wavelet spectrum
    plt4 = plt.subplot(gs[1, -1])
    plt.plot(global_ws, period)
    plt.plot(global_signif, period, '--')
    plt.xlabel('Power (\u0000C$^2$)')
    plt.title('c) Global Wavelet Spectrum')
    plt.xlim([0, 1.25 * np.max(global_ws)])
    # format y-scale
    plt.ylim([np.min(period), np.max(period)])
    plt4.set_yscale('log', base=2, subs=None)
    ax = plt.gca().yaxis
    ax.set_major_formatter(ScalarFormatter())
    plt4.ticklabel_format(axis='y', style='plain')
    plt4.invert_yaxis()
    # --- Plot 2--8 yr scale-average time series
    plt.subplot(gs[2, 0:3])
    plt.plot(time2, scale_avg, 'k')
    plt.plot(xlim,scale_signif+[0,0],'--')
    plt.xlim(xlim[:])
    plt.xlabel('Time (year)')
    plt.ylabel('Avg variance (\u0000C$^2$)')
    plt.title('d) 2-8 yr Scale-average Time Series')
    plt.savefig('Modul_3/'+exp+'_Yogyakarta.png')

# End of Section 