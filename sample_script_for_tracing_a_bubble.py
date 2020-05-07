#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
from optical_raytracer import *

        
def main():
    # define the beam of 2mm width and 20 parallel rays starting at -10mm and end at 20mm.
    # water_x defines, when the index of refraction is turned to 1.333
    # x-axis is optical axis
    beam = Beam(width=2e-3,raycount = 21, water_x=-10e-3, startx=-11e-3, endx=20e-3)

    # add a bubble to the scene
    bubble = Bubble(R=1e-3)
    bubble.plot_lens()
    beam.add_lens(bubble)

    # ------------------

    beam.refract_rays()
    beam.end()
    beam.plot_rays()
    
    plt.axis('equal')
    plt.grid(False)
    plt.show()
  
if __name__ == '__main__':
    main()
