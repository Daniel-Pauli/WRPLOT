#!/usr/bin/env python
# coding: utf-8

# In[1]:


## initialize and load data
from wrplot import wrplotter
import matplotlib.pyplot as plt
wrplot = wrplotter()

clight = 2.9979E5
Msun = 1.98847e33
Rsun = 6.96e10
Lsun = 3.839e33
G = 6.67428e-8
radiation = 7.56578e-15
m = 1.6605388e-24
kb = 1.380650e-16
c = 2.997e10
yr = 3.15576e7


# In[2]:


name1 = "./gal-ob-vd3_18-38_line.txt"

# load model 1
model1 = wrplot.command_include(name1)

# cut of wavelenghts longer than 7000 Ang
model1 = wrplot.command_cut(model1,(3000,7000),axis="x")

# can convolve it with a gaussian
model1 = wrplot.command_convol(model1,"gauss",0.5)
# or with rotation
model1 = wrplot.command_convol(model1,"rot",300)


# In[3]:


wrplot.figure(figsize=(8.27, 11.69),gridsize=(8,5))

wrplot.get_subplot((0,5),(0,2))

xrange = (3000,4000)
xticks = (200,20)
yrange = (0,1.25)
yticks = (0.5,0.1)
yrange = (0,1.5)
yticks = (0.5,1)
ylabel = r"$\mathrm{Norm.\,\,flux}$"

wrplot.plot(model1[0],model1[1],SYMBOL=("-",""),COLOR=2,LABELS=(r"$\lambda\,\,/\,\,\AA$",ylabel),NOLAB=(False,False),
            XRANGE=xrange,XTICKS=xticks,YRANGE=yrange,YTICKS=yticks,DARUNTER=(0,0),PEN=1.5) 

wrplot.get_subplot((0,5),(2,4))

xrange = (4000,5000)

wrplot.plot(model1[0],model1[1],SYMBOL=("-",""),COLOR=2,LABELS=(r"$\lambda\,\,/\,\,\AA$",ylabel),NOLAB=(False,False),
            XRANGE=xrange,XTICKS=xticks,YRANGE=yrange,YTICKS=yticks,DARUNTER=(0,0),PEN=1.5) 

wrplot.get_subplot((0,5),(4,6))

xrange = (5000,6000)

wrplot.plot(model1[0],model1[1],SYMBOL=("-",""),COLOR=2,LABELS=(r"$\lambda\,\,/\,\,\AA$",ylabel),NOLAB=(False,False),
            XRANGE=xrange,XTICKS=xticks,YRANGE=yrange,YTICKS=yticks,DARUNTER=(0,0),PEN=1.5) 

wrplot.get_subplot((0,5),(6,8))

xrange = (6000,7000)

wrplot.plot(model1[0],model1[1],SYMBOL=("-",""),COLOR=2,LABELS=(r"$\lambda\,\,/\,\,\AA$",ylabel),NOLAB=(False,False),
            XRANGE=xrange,XTICKS=xticks,YRANGE=yrange,YTICKS=yticks,DARUNTER=(0,0),PEN=1.5) 

plt.subplots_adjust(hspace=1.5,wspace=0.05)


# In[ ]:




