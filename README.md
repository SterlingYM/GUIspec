# GUIspec
Matplotlib-based, customizable GUI for data visualization/inspection of spectroscopic data.
Under development (as of Jun 2023)

Features you can expect to be added:
- packaging
- toggle on/off emission lines
- custom plotting kwargs

Usage:
~~~python
import GUIspec
import numpy as np
wav = np.arange(3000,8000)
flux = np.random.normal(1e-8,1e-9,wav.shape[0])
objname = 'EXAMPLE'
GUIspec.launch_window(wav,flux,title=objname)
~~~
