import numpy as np
import tqdm
import pickle
from scipy.spatial import Delaunay, ConvexHull

import matplotlib.pyplot as plt

def volume(tetrahedron):
    """Calculate the volume of a tetrahedron"""
    diff = tetrahedron-tetrahedron[-1]
    a = diff[0]
    b = diff[1]
    c = diff[2]
    return np.absolute(np.dot(a,
                              np.cross(b,c))/6)

def same_side(tetrahedron,points):
    """Check whether the points lie on the same side of the plane defiend by the first 3 points of the tetrahedron as the 4th point"""
    normal = np.cross(tetrahedron[1]-tetrahedron[0],
                      tetrahedron[2]-tetrahedron[0])
    dot_opposite = np.dot(normal,tetrahedron[-1]-tetrahedron[0])
    dot_point = np.matmul((points-tetrahedron[0]),normal)
    test = np.sign(dot_opposite) == np.sign(dot_point)
    return test

def inside(tetrahedron,r, skin=2):
    """Check whether the points are inside the tetarhedron"""
    #find a meaningful bounding box
    t = tetrahedron

    bbox = ((r[:,0]>t[:,0].min()-skin)*(r[:,0]<t[:,0].max()+skin)
        *(r[:,1]>t[:,1].min()-skin)*(r[:,1]<t[:,1].max()+skin)
        *(r[:,2]>t[:,2].min()-skin)*(r[:,2]<t[:,2].max()+skin))
    points = r[bbox]
    # print(points.shape)

    test = np.ones(points.shape[0]).astype(bool)
    for k in range(4):
        test *= same_side(np.roll(t,k, axis=0),points)
    return test

def voxel_vol(tetrahedron):
    # create an image out of the tetrahedron

    t = tetrahedron+tetrahedron.min(axis=0)
    t = t.astype(int)
    img = np.zeros((t[:,0].max()+1,t[:,1].max()+1,t[:,2].max()+1))
    img[t[:,0],t[:,1],t[:,2]] = 1

    hull = ConvexHull(t)
    deln = Delaunay(t[hull.vertices]) 
    idx = np.stack(np.indices(img.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(img.shape)
    out_img[out_idx] = 1
    return np.sum(out_img)

class AlphaAnalyser:
    def __init__(self,jaw,skin=0):
        self.name = jaw['name']
        img = jaw['image']
        self.alpha = jaw['alpha_shape']
        vertices = self.alpha['points']/jaw['voxel_size'] 
        self.vertices = vertices.astype(int)
        r = np.array(img.nonzero()).T.astype(int)
        allr = np.array((img>=0).nonzero()).T
        self.r = r
        self.skin = skin
        bbox = ((allr[:,0]>r[:,0].min()-skin)*(allr[:,0]<r[:,0].max()+skin)
        *(allr[:,1]>r[:,1].min()-skin)*(allr[:,1]<r[:,1].max()+skin)
        *(allr[:,2]>r[:,2].min()-skin)*(allr[:,2]<r[:,2].max()+skin))
        self.validr = allr[bbox]
        # print(self.validr.shape, self.r.shape )

    def compute_inner_voxels(self, tqdmon=True):
        """Counting the number of voxels per simplex"""
        self.num_voxels = []
        self.voxel_volume = []
        self.volume = []
        print (":: Counting the number of voxels per simplex ")
        if tqdmon:
            f = tqdm.tqdm
        else:
            f = lambda x: x
        for s in f(self.alpha['simplices']):
            v = volume(self.vertices[s])
            if v<1:
                continue
            else:
                self.num_voxels.append(np.sum(inside(self.vertices[s],self.r)))
                self.volume.append(v)
                # self.voxel_volume.append(np.sum(inside(self.vertices[s], self.validr, skin=0)))



    def dump(self, folder="output"):
        res = {}
        res['num_voxels'] = np.array(self.num_voxels)
        res['voxel_vols'] = np.array(self.voxel_volume)
        res['vols'] = np.array(self.volume)
        res['ratios'] =res['num_voxels']/res['vols'] 
        # for i,j in zip(self.num_voxels,res['vols']):
            # print(i,j)
        # print(res['num_voxels'] )
        # plt.hist(res['ratios'], bins=32)
        # plt.show()
        pickle.dump(res,open(folder+f'/alpha_analyse_{self.name}.pkl', 'bw'))
