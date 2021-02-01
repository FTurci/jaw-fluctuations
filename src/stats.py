import numpy as np 
import pickle 
import time
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit

def model(x,a,b,c):
    return a*np.exp(-(x/b)**c)

def sample_balls(r_centres,tree_data,sizes, maxcount=13000,stride=5):
  """Find number of neighbouring points in r_data within balls of specified sizes at centres r_centres. Requires a cKDTree input for the r_data points."""
  
  tree_centres = cKDTree(r_centres)
  counts = []
  empty = []
  bins = np.arange(0,maxcount,stride)
  for rcut in sizes:
      ngh = tree_centres.query_ball_tree(tree_data, rcut)
      num = [len(nlist) for nlist in ngh]
      counts.append(sum(num))
      H,e = np.histogram(num, bins=bins)
      empty.append(H[0])
  
  results = {}
  results['counts'] = counts
  results['empty'] = empty
  results['histo'] = (H,bins)
  
  return results



class BallSampler:
  def __init__(self, data,sigma=10,mindist=5):
    #     blur the data to connect eventually disconnected regions
    self.data = data
    self.gaussian = ndimage.gaussian_filter(data, sigma)
    self.mindist = mindist

  def sample(self,N=1000,k=3,sizes = np.arange(3,16), maxcount=13000, stride=5, ):
        #   identify as inner points only those that are sufficiently distant from the background
    inner = ndimage.distance_transform_cdt(self.gaussian)>self.mindist
    self.inner = inner
    self.r_in = np.array(inner.nonzero()).T
    np.random.shuffle(self.r_in)
    self.r = np.array(self.data.nonzero()).T
    self.tree_data = cKDTree(self.r)
    
    
    """Run k independent sampling procedures from N centres."""
    self.runs = {}
    self.run_params = {}
    self.run_params['N'] = N
    self.run_params['sizes'] = sizes
    self.run_params['maxcount'] = maxcount
    self.run_params['stride'] = stride
    self.run_params['k'] = k
    
    for j in range(1, k+1):
      centres = self.r_in[N*j:N*(j+1)]

      self.runs[j] = sample_balls(centres,self.tree_data,sizes, maxcount,stride)
  
  def get_stat(self):
    # get the statistics
    results = {}
    for key,value in self.runs[1].items():
      if key!='histo':
        table = [ self.runs[j][key] for j in range(1,self.run_params['k']+1)]

        results['avg_'+key] = np.mean(table, axis=0)
        results['std_'+key] = np.std(table, axis=0)


    popt,pcov = curve_fit(model, self.run_params['sizes'],results['avg_empty'], p0=[500,1,1])
    
    fit = {}
    fit['model'] = model
    fit['popt'] = popt
    fit['pcov'] = pcov
    
    results['fit']= fit
    return results
  
  def get_remoteness(self,threshold=0):
    negative = ~(self.data>threshold)
    negative[self.gaussian==0]=0
    negative[self.data>0]=0

    self.distance = ndimage.distance_transform_cdt(negative)
    self.remoteness = self.distance[self.distance>0]
    self.avg_remoteness = self.remoteness.mean()