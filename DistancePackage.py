from scipy.optimize import minimize
import numpy as np
class DistanceLearn:
  """
  Distance Learn Algorithm:
  
  Given a matrix X where the rows are samples and the columns are features and pairwise similarity between samples, 
  create a new space that preserves pairwise distances between samples in feature space. In this transformed space,
  the Euclidean distance between points is proportional to the error between points.
  """
  
  def CalcAngle(self, v, u):
    return np.arccos( np.dot(v,u)/(np.linalg.norm(v)*np.linalg.norm(u)))
  
  def RotateVec( self, u, angle ):
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    if np.shape( u ) != [2,1]:
      np.reshape(u, (2,1))
    return np.dot(R, u)
  
  
  def InitializeTransformMatrix(self):
    """
    InitializeTransformMatrix:
    
    Initialize transformation matrix so that angle between vectors (stimuli) function of similarity. This 
    space is meaningless at this stage - will need to rotate space s.t. transform is invariant to ones vector 
    """
    S = np.eye(self.num_samples)
    a = (np.pi/2 - self.confusion_matrix[0,1])/2
    S[:2,0] = self.RotateVec(S[:2,0], a)
    S[:2,1] = self.RotateVec(S[:2,1], -a)
    for i in range(2,self.num_samples):
      d = []
      for j in range(i):
        d.append( np.cos(self.confusion_matrix[i,j]) )
      np.array(d)
      S[:,i] = np.dot( d, np.linalg.pinv(S[:,range(i)]) )
      l = np.square(np.linalg.norm(S[:,i]))
      if (l <= 1):
        S[i,i] = np.sqrt(1 - np.square(np.linalg.norm(S[:,i])))
      else:
        S[i,i] = -np.sqrt(np.square(np.linalg.norm(S[:,i]))-1)
    return S

  def InitializeWeights(self):
    return np.random.rand(self.num_samples*self.num_samples)
  
  def ObjFunc(self, T):
    """
    Define objective function for rotating. Specifically want to rotate such that transform is invariant to 
    ones vector.
    """
    Ts = []
    As = []
    for i in range(self.num_features):
      Ts.append(T[i*self.num_samples:(i+1)*self.num_samples])
      As.append(np.sum(self.A[i,:]))
    
    As = np.array(As)
    o = 0
    for t in Ts:
      o = o + (0.5*(1-np.dot(t,As))**2)
    return o
  
  def ObjFuncDeriv(self, T):
    """
    Derivative of objective function.
    """
    dfdT = []
    Ts = []
    As = []
    for i in range(self.num_samples):
      Ts.append(T[i*self.num_samples:(i+1)*self.num_samples])
      As.append(np.sum(self.A[i,:]))
    
    As = np.array(As)
    for i in range(self.num_samples):
      for j in range(i*self.num_samples, (i+1)*self.num_samples):
        dfdT.append( -(1-np.dot(Ts[i],As)) * (T[j]*As[j%self.num_samples]) )
    return np.array(dfdT)

  def GenLambdaCons1(self, Tp, i, epsilon):
    """
    First constraint for objective function: 
    
    rotate s.t. norm of axes of compressed space is fixed.
    """
    return lambda t: -0.5*np.linalg.norm(self.A[:,i] - np.dot(Tp, self.A[:,i]))**2 + epsilon
  
  def GenLambdaCons2(self, Tp, i, j, epsilon):
    """
    Second constraint for objective function:
    
    rotate s.t. angle between vectors is fixed (reduces to making sure dot product is identical because
    of constraint 1)
    """
    TA1 = np.dot(Tp, self.A[:,i])
    TA2 = np.dot(Tp, self.A[:,j])
    return lambda t: -0.5*(np.dot(self.A[:,i], self.A[:,j]) - np.dot(TA1, TA2))**2 + epsilon
  
  def GenConstraints(self, T, epsilon):
    """
    Generate all them constraints
    """
    Tp = np.reshape(T, (self.num_samples, self.num_samples) )
    C = []
    for i in range(self.num_samples):
      C.append(self.GenLambdaCons1(Tp, i,epsilon))
    
    for i in range(self.num_samples):
      for j in range(i, self.num_samples):
        C.append(self.GenLambdaCons2(Tp, i, j, epsilon))
            
    cons=[]
    for c in C:
      cons.append({'type': 'ineq',
                   'fun' : c})
    return cons
    
  def GetTransformMatrix(self):
    return self.transform_matrix
  
  def RLearn(self, X, confusion_matrix, epsilon=0):
    """
    RLearn:
    
    Transforms a matrix of samples and features such that samples that are similar to each other are closer 
    together, while samples that are different are further from each other. This is a transformation of the 
    feature space in which the stimuli exist.
    
    Input:
    X: (num_samples X num_features), a matrix where the columns are the features and the rows are samples
    confusion_matrix: (num_samples X num_samples), a similarity matrix where the (i,j)th entry is the similarity between samples i and j
    epsilon: [default 0], a tradeoff parameter. Larger epsilon means preserving angle is more important, smaller epsilon means invariance to ones vector most important

    """
    self.confusion_matrix = confusion_matrix
    self.epsilon = epsilon
    self.F = X.T
    self.X = X
    self.num_features = np.shape(self.F)[0]
    self.num_samples = np.shape(self.X)[0]
    self.A = self.InitializeTransformMatrix()
    self.T = self.InitializeWeights()
    self.constraints = self.GenConstraints( self.T, self.epsilon )
    self.res = minimize(self.ObjFunc, self.T, jac=self.ObjFuncDeriv, 
                        constraints=self.constraints, method='SLSQP', options={'disp':True} )
    self.T = np.reshape(self.res.x, (self.num_samples, self.num_samples))
    self.transform_matrix = np.dot(self.T,self.A)
    
    self.transformed_X = np.dot(self.transform_matrix, self.X)
    self.transformed_F = self.transformed_X.T
    
    print(np.dot(self.transform_matrix, np.ones(self.num_samples)))
    print(self.transformed_F)

    return self.transformed_X
          
    
