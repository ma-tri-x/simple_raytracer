#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
from optical_raytracer import *

        
def main():
    # define the beam of 25.4mm width and 9 parallel rays starting at -10mm and end at 120mm.
    # water_x defines, when the index of refraction is turned to 1.333
    # x-axis is optical axis
    point_source = PointSourceArray(width=20e-3, raycount = 4, water_x=100.0, startx=-10e-3, endx=100e-3, beam_offset_y=0.0, spray_angle=40.,number_of_sources=7)
    #point_source = Beam(raycount = 7, water_x=100.0, startx=-10e-3, endx=600e-3, beam_offset_y=0.0)

   
    # add a cuvette filled with water to the scene
    # left side: x= -0.005, right side = left_side + 119e-3
    # refraction does not happen at right side of cuvette, unless
    # cuvette.offset_x set and cuvette.offset_x + d < beam.water_x
    #cuvette = WaterLens(d=119e-3)
    #cuvette.h = h
    #cuvette.offset_x = -0.005 # if not set, then cuvette.offset_x = beam.water_x
    #cuvette.plot_lens()
    #beam.add_lens(cuvette)

    ## TEMPLATES OF OTHER LENSES/MIRRORS:
    ##---------------------------
    #best_form_lens_f50_1inch = SphLLens(d=60e-3, Rl=0.172, Rr=60.1e-3)
    #best_form_lens_f50_1inch.h = 25.4e-3
    #best_form_lens_f50_1inch.offset_x = 0.0
    #best_form_lens_f50_1inch.plot_lens()
    #point_source.add_lens(best_form_lens_f50_1inch)
    
    #mirror = MirrorF50(offset_x=10e-3)
    ##mirror = ParabMirror(offset_x=10e-3,d=2e-3,h=h)
    ##mirror.plot_mirror()
    ##beam.add_lens(mirror)
    
    ##water = NegativeCuvette(d=40e-3,offset_x=-10e-3)
    ##water.h = h
    ##water.plot_lens()
    ##beam.add_lens(water)
    
    #par = ParabLens(d=1e-3,a=20, b=0)
    #par.h = h
    #par.plot_lens()
    #beam.add_lens(par)
    
    #asph = AsphLens()
    #asph.offset_x=0.0
    #asph.h=50e-3
    #asph.plot_lens()
    #point_source.add_lens(asph)
    
    sph = SphLLens(Rr=50e-3,Rl="inf",d=3e-3)
    sph.offset_x=10.0e-3
    sph.h=100e-3
    sph.plot_lens()
    point_source.add_lens(sph)

    #bubble = Bubble(R=1e-3)
    #bubble.plot_lens()
    #beam.add_lens(bubble)
    
    #wat = WaterLens()
    #wat.h=h
    #wat.offset_x=10e-3
    #wat.d = 10e-3
    #wat.plot_lens()
    #beam.add_lens(wat)

    # ---------------------
    point_source.refract_rays()
    point_source.end()
    point_source.write_output()
    point_source.plot_rays()
    
    plt.axis('equal')
    plt.grid(False)
    plt.show()
  
if __name__ == '__main__':
    main()
