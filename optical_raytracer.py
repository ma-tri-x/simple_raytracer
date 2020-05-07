#!/bin/python

import numpy as np
import matplotlib.pyplot as plt

DEBUG=False
DEBUGADV=False

class Ray(object):
    def __init__(self, startx, starty):
        self.x = [startx]
        self.y = [starty]
        self.nx= 1.0
        self.ny= 0.0
        self.use = True
        
    def pos_x(self,idx):
        return self.x[idx]
    
    def pos_y(self,idx):
        return self.y[idx]
    
    def add_pos(self,x,y):
        self.x.append(x)
        self.y.append(y)
        
    def delete_last_pos(self):
        self.x.pop()
        self.y.pop()
    
    def propagate(self,c):
        self.x[-1] += self.nx*c
        self.y[-1] += self.ny*c

class Lens(object):
    def __init__(self, offset_x=0., n=1.75, h=1e-2):
        self.offset_x=offset_x
        self.n = n
        self.h = h
        self.intervals=100
        
    def left_x(self):
        return
    
    def right_x(self):
        return
        
    def _f(self, ray_inst, sidefunc, y):
        y0 = ray_inst.pos_y(-1)
        if ray_inst.ny == 0.:
            res = (y-y0)**2
        else:
            m_inv = ray_inst.nx/ray_inst.ny
            x0 = ray_inst.pos_x(-1)
            res = sidefunc(y) - (m_inv*(y-y0)+x0)
        return res
    
    def Newtonstep(self,ray_inst,sidefunc,yn):
        ynn= yn - self._f(ray_inst, sidefunc,yn)/((self._f(ray_inst, sidefunc,yn+1e-7)-self._f(ray_inst, sidefunc,yn-1e-7))/2e-7)
        if DEBUG: print "ynn = {}".format(ynn)
        return ynn
    
    def Newton_find_yzero(self, ray_inst, sidefunc, yn = 0.):
        if np.abs(ray_inst.ny) < 1e-10:
            return ray_inst.pos_y(-1)
        epsilon = 1e-4
        #
        ny = yn
        ynn= self.Newtonstep(ray_inst,sidefunc,yn)
        while np.abs((ynn-yn)/(yn+1e-10)) > epsilon:
            ny = yn
            yn = ynn
            ynn= self.Newtonstep(ray_inst,sidefunc,yn)
            if np.abs(ny - ynn) < 1e-7: return 0.5*(ynn+yn)
        return ynn
            
    def _rot_counterclock(self, phi, vec):
        b_vec = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi),np.cos(phi)]]).dot(vec)
        return b_vec
            

    def refract_ray_curved(self,ray_inst, water_x):
        for sidefunc in [self.left_x,self.right_x]:
            if DEBUG: print "-- {}".format(sidefunc)
            willhit=True
            iteration = 0
            yn = 0.
            while ray_inst.use and willhit and iteration < 2:
                yhit = self.Newton_find_yzero(ray_inst, sidefunc, yn=yn)
                xhit = sidefunc(yhit)
                if np.abs(yhit) > self.h/2. or np.abs(xhit-ray_inst.pos_x(-1))<1e-7: willhit= False
                if willhit:
                    # tangentiale und orthonormale berechnen und plotten
                    lens_deriv_x = sidefunc(yhit+1e-6) - sidefunc(yhit-1e-6)
                    lens_tangential = np.array([lens_deriv_x,2e-6])
                    lens_tangential /= np.linalg.norm(lens_tangential)
                    lens_orthonormal = np.array([lens_tangential[1],-lens_tangential[0]])
                    if lens_orthonormal[0] > 1e-8: 
                        lens_orthonormal *=-1
                    if DEBUG: plt.arrow(xhit,yhit,lens_orthonormal[0]*1e-3,lens_orthonormal[1]*1e-3,head_width=0.,width=1e-4)
                    
                    nray = np.array([ray_inst.nx,ray_inst.ny])
                    angle_left = np.arccos(nray.dot(lens_orthonormal))
                    if angle_left > np.pi/2.: 
                        angle_left = np.pi - angle_left
                    env_n = 1.0
                    if xhit > water_x: env_n = 1.333
                    ratio = env_n/self.n
                    if np.mod(iteration,2) == 1: ratio = 1./ratio
                    if sidefunc == self.right_x: ratio = 1./ratio
                    if DEBUG: print "- iter: {}, ratio: {}".format(iteration, ratio)
                    if DEBUG: print "- np.abs(ratio * np.sin(angle_left)): {}".format(np.abs(ratio * np.sin(angle_left)))
                    if np.abs(ratio * np.sin(angle_left)) < 1.0:
                        angle_2 = np.arcsin(ratio * np.sin(angle_left))
                        angle_tilt = angle_left-angle_2
                        if DEBUG: print "- angle_left {}, angle_2 {}, angle_tilt: {}".format(angle_left/np.pi*180., angle_2/np.pi*180.,angle_tilt/np.pi*180.)
                        nrot = np.array([0.,0.])
                        signA = (-1)**(iteration+1)
                        #if sidefunc == self.right_x: signA = -signA
                        signB = -signA
                        if (nray+lens_orthonormal)[1] < 0.:
                            nrot = self._rot_counterclock(signA*angle_tilt,nray)
                        if (nray+lens_orthonormal)[1] >= 0.:
                            nrot = self._rot_counterclock(signB*angle_tilt,nray)
                        ray_inst.nx = nrot[0]
                        ray_inst.ny = nrot[1]
                        ray_inst.add_pos(xhit,yhit)
                        if DEBUG: print "- using xhit: {}, yhit {}".format(xhit,yhit)
                        if DEBUG: print "- new nx: {}, ny {}".format(ray_inst.nx,ray_inst.ny)
                        if DEBUGADV: plt.arrow(xhit,yhit,ray_inst.nx*1e-4,ray_inst.ny*1e-4,head_width=0.,width=0.3e-4,color="green")
                        willhit = False
                        if np.abs(ratio * np.sin(angle_left)) >= 0.90:
                            willhit = True
                            fst_deriv = (sidefunc(yhit+1e-8) - sidefunc(yhit-1e-8))/2e-8
                            sec_deriv = (sidefunc(yhit+1e-8) - 2.*sidefunc(yhit) + sidefunc(yhit-1e-8))/1e-8/1e-8
                            curv = sec_deriv/(1.+fst_deriv*fst_deriv)**(1.5)
                            rad = np.abs(1./curv)
                            yn_new = yhit + ray_inst.ny * 2.*rad*np.cos(angle_2)
                            if DEBUG: print "- close to TIR. R: {}, Proposing new yn: {}".format(rad,yn_new)
                            yn = yn_new
                            
                        iteration += 1
                    else:
                        if DEBUG : print "- inside TIR"
                        angle_tilt = 2.*(np.pi/2. - angle_left)
                        nrot = np.array([0.,0.])
                        if (nray+lens_orthonormal)[1] < 0.:
                            nrot = self._rot_counterclock( angle_tilt,nray)
                        if (nray+lens_orthonormal)[1] >= 0.:
                            nrot = self._rot_counterclock(-angle_tilt,nray)
                        ray_inst.nx = nrot[0]
                        ray_inst.ny = nrot[1]
                        ray_inst.add_pos(xhit,yhit)
                        if DEBUG: print "- using xhit: {}, yhit {}".format(xhit,yhit)
                        if DEBUG: print "- new nx: {}, ny {}".format(ray_inst.nx,ray_inst.ny)
                        ray_inst.add_pos(xhit+nrot[0]*1e-3,yhit+nrot[1]*1e-3)
                        if DEBUGADV: plt.arrow(xhit,yhit,ray_inst.nx*1e-4,ray_inst.ny*1e-4,head_width=0.,width=0.3e-4,color="red")
                        
                        iteration += 1
                        ray_inst.use = False
            
    def plot_lens(self):
        div = 30
        lyl = range(div+1)
        ly = [-self.h/2. + k*self.h/div for k in lyl]
        lx = [self.left_x(y) for y in ly]
        rx = [self.right_x(y) for y in ly]
        plt.plot(lx,ly)
        plt.plot(rx,ly)
        
class Mirror(Lens):
    def __init__(self, offset_x=0., h=1e-2):
        self.offset_x=offset_x
        self.h = h
        
    def left_x(self):
        return
        
    def _f(self, ray_inst, sidefunc, y):
        return Lens()._f(ray_inst, sidefunc, y)
    
    def Newtonstep(self,ray_inst,sidefunc,yn):
        return Lens().Newtonstep(ray_inst,sidefunc,yn)
    
    def Newton_find_yzero(self, ray_inst, sidefunc, yn = 0.):
        return Lens().Newton_find_yzero(ray_inst, sidefunc, yn = 0.)
            
    def _rot_counterclock(self, phi, vec):
        return Lens()._rot_counterclock(phi, vec)
            

    def refract_ray_curved(self,ray_inst, water_x):
        for sidefunc in [self.left_x]:
            if DEBUG: print "-- Mirror"
            willhit = True
            yhit = self.Newton_find_yzero(ray_inst, sidefunc)
            xhit = sidefunc(yhit)
            if np.abs(yhit) > self.h/2.: willhit= False
            if willhit:
                if DEBUG : print "- hit and reflecting"
                # tangentiale und orthonormale berechnen und plotten
                lens_deriv_x = sidefunc(yhit+1e-6) - sidefunc(yhit-1e-6)
                lens_tangential = np.array([lens_deriv_x,2e-6])
                lens_tangential /= np.linalg.norm(lens_tangential)
                lens_orthonormal = np.array([lens_tangential[1],-lens_tangential[0]])
                if lens_orthonormal[0] > 1e-8: 
                    lens_orthonormal *=-1
                if DEBUG: plt.arrow(xhit,yhit,lens_orthonormal[0]*1e-3,lens_orthonormal[1]*1e-3,head_width=0.,width=1e-4)
                nray = np.array([ray_inst.nx,ray_inst.ny])
                angle_left = np.arccos(nray.dot(lens_orthonormal))
                if angle_left > np.pi/2.: 
                    angle_left = np.pi - angle_left
                nray = np.array([-ray_inst.nx,-ray_inst.ny])
                angle_tilt = 2.*(angle_left)
                nrot = np.array([0.,0.])
                if (nray+lens_orthonormal)[1] < 0.:
                    nrot = self._rot_counterclock(-angle_tilt,nray)
                if (nray+lens_orthonormal)[1] >= 0.:
                    nrot = self._rot_counterclock( angle_tilt,nray)
                ray_inst.nx = nrot[0]
                ray_inst.ny = nrot[1]
                ray_inst.add_pos(xhit,yhit)
                if DEBUG: print "- xhit: {}, yhit {}".format(xhit,yhit)
                if DEBUG: print "- new nx: {}, ny {}".format(ray_inst.nx,ray_inst.ny)
                #ray_inst.add_pos(xhit+nrot[0]*1e-3,yhit+nrot[1]*1e-3)
                if DEBUGADV: plt.arrow(xhit,yhit,ray_inst.nx*1e-4,ray_inst.ny*1e-4,head_width=0.,width=0.3e-4,color="red")
                
            
    def plot_mirror(self):
        div = 30
        lyl = range(div+1)
        ly = [-self.h/2. + k*self.h/div for k in lyl]
        lx = [self.left_x(y) for y in ly]
        plt.plot(lx,ly)
        
    
        
class AsphLens(Lens):
    def __init__(self):
        self.info="thorlabs ACL7560U"
        self.R=31.384e-3
        self.k=-1.911
        self.A=5.0e-6
        self.d=30e-3
        self.n=1.43
        self.offset_x = Lens().offset_x
        self.h = Lens().h
        
    def right_x(self,y):
        x = -(y*y/(self.R*(1.+np.sqrt(1.-(1.+self.k)*y*y/self.R/self.R))) + self.A*y*y*y*y) + self.offset_x + self.d   #/17.8e-3*27.7e-3
        return x
    
    def left_x(self,y):
        return self.offset_x
    
class SphLLens(Lens):
    def __init__(self, Rl, Rr, d):
        self.info="some spherical lens"
        self.R=Rl
        self.Rr=Rr
        self.d = d
        self.h = Lens().h
        self.n=1.43
        self.offset_x=Lens().offset_x
        
    def left_x(self,y):
        posx = np.sqrt(self.R*self.R - y*y) - self.offset_x - np.sqrt(self.R*self.R - self.h*self.h/4.)
        return -posx
    
    def right_x(self,y):
        if self.Rr=="inf":
            return self.d + self.offset_x
        else:
            return np.sqrt(self.Rr*self.Rr - y*y) + self.d + self.offset_x - np.sqrt(self.Rr*self.Rr - self.h*self.h/4.)
    
class ParabLens(Lens):
    def __init__(self, a, b, d):
        self.info="some spherical lens"
        self.a=a
        self.b=b
        self.d = d
        self.h = Lens().h
        self.n=1.43
        self.offset_x=Lens().offset_x
        
    def left_x(self,y):
        return self.a*y*y  + self.offset_x
    
    def right_x(self,y):
        return self.d + self.a*(self.h/2.)**2 + self.offset_x + self.b*(self.h/2.)**2 - self.b*y*y

class Beam(object):
    def __init__(self, width=5e-3, raycount=10, startx=-10e-3, water_x=0., endx=60e-3, debug=False):
        self.width = width
        self.startx = startx
        self.endx = endx
        self.water_x = water_x
        self.water_n = 1.333
        self.raycount = raycount
        self.rays = []
        self.lenses = []
        self.debug = debug
        if self.raycount == 1:
            self.rays.append(Ray(startx,self.width))
        else:
            for i in range(self.raycount):
                new_y = -self.width/2. + i*self.width/(self.raycount-1.)
                #print -self.width/2. + i*self.width/(self.raycount-1.)
                self.rays.append(Ray(startx,new_y))
        
    def end(self):
        num=0
        for i in self.rays:
            if i.use:
                endx = self.endx
                last_y = 0.
                if np.abs(i.ny) > np.abs(2.*i.nx): endx = i.pos_x(-1) + 5e-3 
                last_y = i.ny/i.nx * endx + (i.pos_y(-1) - i.ny/i.nx*i.pos_x(-1))
                i.add_pos(endx,last_y)
                if DEBUG: print "---END: num {}, last_x {}, last_y {}".format(num,endx,last_y)
            if DEBUG: plt.text(i.pos_x(-1)+1e-4,i.pos_y(-1),str(num))
            num+=1
            
    #def write_output(self):
        #with open("out.dat", 'w') as out:
            #for ray in self.rays:
                #for j,x in enumerate(ray.x):
                    #out.write("{}  {}\n".format(x,ray.y[j]))
                #out.write("\n")
    
    def add_lens(self,lens_inst):
        self.lenses.append(lens_inst)
        
    def refract_rays(self):
        for lens in self.lenses:
            for num,ray in enumerate(self.rays):
                print "---{}---".format(num)
                lens.refract_ray_curved(ray,self.water_x)
                
    def plot_rays(self):
        for num,ray in enumerate(self.rays):
            if DEBUG: plt.plot(ray.x,ray.y,'o-')
            else: plt.plot(ray.x,ray.y,color=(np.mod(num-1,2),np.mod(num,2),0,1))
        
    
class WaterLens(Beam,Lens):
    def __init__(self,d=100e-3):
        self.h = Lens().h
        self.d = d
        self.n = 1.333
        self.offset_x = Beam().water_x
            
    def left_x(self,y):
        return self.offset_x
    
    def right_x(self,y):
        return self.offset_x + self.d
    
class NegativeCuvette(Beam,Lens):
    def __init__(self,d=120e-3,offset_x=0.0):
        self.h = Lens().h
        self.d = d
        self.n = 1.333
        self.offset_x = offset_x
            
    def left_x(self,y):
        return self.offset_x
    
    def right_x(self,y):
        return self.offset_x - self.d
    
    
class Bubble(Lens):
    def __init__(self, R=1e-3, offset_x=0.):
        self.R = R
        self.h = 2*R
        self.n = 1.0
        self.offset_x = offset_x
        self.intervals = Lens().intervals
            
    def left_x(self,y):
        if np.abs(y) > self.R: return 0
        return self.offset_x - np.sqrt(self.R**2 - y*y)
    
    def right_x(self,y):
        if np.abs(y) > self.R: return 0
        return self.offset_x + np.sqrt(self.R**2 - y*y)
            
class MirrorF50(Mirror):
    def __init__(self, offset_x=0., h=0.0254):
        self.offset_x=offset_x
        self.h = h
        self.R = 0.1
    
    def left_x(self,y):
        posx = np.sqrt(self.R*self.R - y*y) + self.offset_x - np.sqrt(self.R*self.R - self.h*self.h/4.)
        return posx
    
class ParabMirror(Mirror):
    def __init__(self, offset_x=0., h=0.0254, d=3e-3):
        self.offset_x=offset_x
        self.h = h
        self.R = 0.1
        self.d = d
        self.coeff = 2.*self.d/self.h**2
    
    def left_x(self,y):
        posx = -self.coeff*y*y + self.offset_x
        return posx