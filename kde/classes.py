import numpy as np
from scipy.interpolate import RectBivariateSpline as spl2D

import pykde
from stat_tools import weighted_cov


class KDE:
    """Initialize KDE object

    Parameters
    ----------
    data
    use_cuda
    weights
    alpha
    method

    """
    def __init__(self, data, use_cuda, weights=[], alpha=0.3, method='silverman'):
        self.use_cuda = use_cuda
        if self.use_cuda:
            import pycuda.driver as cuda
            import pycuda.autoinit
            self.cuda = cuda

        if len(np.array(data).shape) ==1:
            self.data = np.array([data])
        else:
            self.data = np.array(data)

        self.d, self.n = self.data.shape

        self.alpha = alpha

        if len(weights) == self.n:
            self.w = weights*1.0/sum(weights)
            self.setCovariance(weights=True)
        elif len(weights) == 0:
            self.w = np.ones(self.n)*1.0/self.n
            self.setCovariance(weights=False)
        else:
            raise AssertionError("Length of data (%d) and length of weights (%d) incompatible." % (self.n, len(weights)))

        self.hMethod = method

    def setCovariance(self, weights=False):
        """Set covariance from data and weights

        Parameters
        ----------
        weights : bool

        """
        if weights:
            self.c = weighted_cov(self.data, weights=self.w, bias=False)
        else:
            self.c = np.cov(self.data)

        if self.d != 1:
            self.c_inv = np.linalg.inv(self.c)
            self.detC = np.linalg.det(self.c_inv)
        else:
            self.c_inv = 1.0/self.c
            self.detC = self.c_inv

    def calcLambdas(self, weights=False, weightedCov=False):
        """Calculate bandwidth lambda for data points in KDE function

        Parameters
        ----------
        weights : bool
        weightedCov : bool

        """
        self.configure("lambdas",  weights=weights, weightedCov=weightedCov)

        if self.use_cuda:
            self.cuda_calc_lambdas()
        else:
            if use_grid and self.d == 2:
                # grid #
                N_grid = 100
                ext = np.zeros([2, self.d])
                for i in range(self.d):
                    diff = ( np.max(self.data[i]) - np.min(self.data[i]) ) * 0.1
                    ext[i][0], ext[i][1] = np.min(self.data[i]) - diff, np.max(self.data[i]) + diff
                spaces1D = [np.linspace(t[0],t[1],N_grid) for t in ext]
                grid = np.array(np.meshgrid(*spaces1D))
                grid = grid.reshape([self.d, len(grid.flatten()) / self.d])

                # kde #
                w = self.w if weights else np.ones(len(self.w))*1.0 / len(self.w)
                ss_kde = pykde.gaussian_kde(self.data, weights=w,
                                            adaptive=False, weight_adaptive_bw=False,
                                            alpha=self.alpha, bw_method=self.hMethod)

                vals =  ss_kde(grid)    # evaluate in log instead of linear to avoid negative values
                spline1 = spl2D(*spaces1D, z=(np.array(vals)).reshape([N_grid, N_grid]).T, kx=3, ky=3)   # TRANSPOSE HERE!!!!
                kde_vals2 = np.array([spline1( x,y)[0][0] for x,y in self.data.T]) # go back to linear world

                #print "Ratio direct:"
                #print ( np.array(kde_vals2) - np.array(kde_vals) ) / np.array(kde_vals)

                # lambdas #
                glob_sum = np.exp(np.sum(np.log(kde_vals2)*1.0/len(kde_vals2)))
                self.lambdas = np.power(kde_vals2 / glob_sum, (-1.0)* self.alpha) #self.alpha*(-1.0)) # hack !!!
            else:
                self.lambdas = kde_nDim.getLambda_ND(
                    int(self.d),
                    list(self.c_inv.flatten()),
                    list(self.data.flatten()),
                    list(self.w_norm_lambdas),
                    float(self.detC),
                    self.h,
                    self.alpha
                )

    def kde(self, points, weights=True, weightedCov=True):
        """Evaluate kde function

        Parameters
        ----------
        points
        weights : bool
        weightedCov : bool

        """
        if len(np.array(points).shape) ==1:
            self.points = np.array([points])
        else:
            self.points = np.array(points)
        self.d_pt, self.m = self.points.shape

        if self.d > 1 and not self.d == self.d_pt:
            assert self.d == self.m
            points = zip(*points[::1])
            self.d_pt, self.m = np.array(points).shape
            import warnings.warn
            warnings.warn("Dimensions of given points did not fit initialized"
                          " kde function. Rotate given sample and proceed"
                          " with fingers crossed.")

        self.configure("kde", weights=weights, weightedCov=weightedCov)

        if self.use_cuda:
            self.cuda_kde(points)
        else:
            self.values = kde_nDim.kde_ND(
                int(self.d),
                list(self.c_inv.flatten()),
                list(self.data.flatten()),
                list(self.points.flatten()),
                list(self.w_norm),
                list(self.preFac),
                float(self.detC),
                self.h,
            )

    def configure(self, mode, weights=False, weightedCov=False):
        """Get h, tempNorm, w_norm, w_lambdas, preFac

        Parameters
        ----------
        mode
        weights : bool
        weightedCov : bool

        """
        if isinstance(self.hMethod, str):
            if self.hMethod == 'silverman':
                # (n * (d + 2) / 4.)**(-1. / (d + 4)).
                self.h = np.power(1.0/(self.n*(self.d+2.0)/4.0), 1.0/(self.d+4.0))
            elif self.hMethod == 'scott':
                self.h = np.power(1.0/(self.n), 1.0/(self.d+4.0))
            else:
                raise ValueError("%s unknown string as normalization"
                                 " constant. Implemented are 'scott',"
                                 " 'silverman'" %(self.hMethod,))
        elif isinstance(self.hMethod, (int, float)):
                self.h = self.hMethod
        else:
            raise ValueError("Normalization constant must be of type int,"
                             " float or str!")

        self.setCovariance(weights=weightedCov)

        if weights: self.weights = self.w
        else: self.weights = np.ones(self.n)*1.0/self.n

        if mode == "lambdas":
            self.w_norm_lambdas = self.weights * np.sqrt(self.detC / np.power(2.0*np.pi*self.h*self.h, self.d))
            self.preFac = -0.5/np.power(self.h, 2)
        elif mode == "kde":
            self.w_norm = self.weights * np.sqrt(self.detC / np.power(2.0*np.pi*self.h*self.h*np.array(self.lambdas)*np.array(self.lambdas), self.d))
            self.preFac = -0.5/np.power(self.h*np.array(self.lambdas), 2)
        else:
            raise ValueError("Could not configure kde object. Unknown mode: %s" %(mode,))

    def cuda_calc_lambdas(self):
        """Calculate lambdas using cuda implementation"""
        from pycuda.compiler import SourceModule

        # conversion of python variables
        n = np.int32(self.n)
        logSum = np.zeros(n)
        kde_val_la = np.zeros(n)

        h_kde_val_la = np.array(kde_val_la).astype(np.float64)
        h_logSum = logSum.astype(np.float64)
        h_w_norm_lambdas = np.array(self.w_norm_lambdas).astype(np.float32)

        # reservation of memory on gpu
        d_kde_val_la = self.cuda.mem_alloc(h_kde_val_la.nbytes)
        d_logSum = self.cuda.mem_alloc(h_logSum.nbytes)
        d_w_norm_lambdas = self.cuda.mem_alloc(h_w_norm_lambdas.nbytes)

        # memory copy to gpu
        self.cuda.memcpy_htod(d_kde_val_la, h_kde_val_la)
        self.cuda.memcpy_htod(d_logSum, h_logSum)
        self.cuda.memcpy_htod(d_w_norm_lambdas, h_w_norm_lambdas)

        # dimension-dependent memory allocation
        if self.d == 2:
            h_x1 = np.array(self.data[0]).astype(np.float32)
            h_x2 = np.array(self.data[1]).astype(np.float32)
            d_x1 = self.cuda.mem_alloc(h_x1.nbytes)
            d_x2 = self.cuda.mem_alloc(h_x2.nbytes)
            self.cuda.memcpy_htod(d_x1, h_x1)
            self.cuda.memcpy_htod(d_x2, h_x2)
            addParam = "const float *x2, const double c11, const double c12, const double c21, const double c22"
            addDeclare = "double ent2;"
            calculation = """
                ent1 = x1[j]-x1[idx];
                ent2 = x2[j]-x2[idx];
                thisKde += w_norm_lambda[j] * exp(preFac * (ent1*(c11*ent1+c12*ent2) + ent2*(c21*ent1+c22*ent2)));
            """
        elif self.d == 1:
            h_x1 = np.array(self.data[0]).astype(np.float32)
            d_x1 = self.cuda.mem_alloc(h_x1.nbytes)
            self.cuda.memcpy_htod(d_x1, h_x1)
            addParam = "const double c"
            addDeclare = ""
            calculation = """
                ent1 = x1[j]-x1[idx];
                thisKde += w_norm_lambda[j] * exp(preFac * (ent1*c*ent1));
            """

        # define function on gpu to be executed
        mod = SourceModule("""
            __global__ void CalcLambda(const float *x1, """+addParam+""", const int n, const double preFac, const float *w_norm_lambda, double *logSum, double *kde){
                int idx = threadIdx.x + blockIdx.x*blockDim.x;
                if (idx < n){
                    double thisKde, ent1;
                    """+addDeclare+"""
                    int j;
                    thisKde = 0.0;
                    for (j=0; j < n; j++) {
                        """+calculation+"""
                    } // for
                   logSum[idx] = 1.0/n * log(thisKde);
                   kde[idx] = thisKde;
                } // if
                __syncthreads();
            } // CalcLambda_2d
        """)

        if n >= 512:
            bx = np.int32(512)
        else:
            bx = np.int32(n)
        gx = np.int32(n/bx)
        if n % bx != 0: gx += 1

        func = mod.get_function("CalcLambda") # code compiling
        if self.d == 2:
            # call of gpu function
            func(d_x1, d_x2, self.c_inv[0][0], self.c_inv[1][0], self.c_inv[0][1], self.c_inv[1][1], n, self.preFac, d_w_norm_lambdas,  d_logSum, d_kde_val_la, block=(int(bx),1,1), grid=(int(gx),1,1))
        elif self.d == 1:
            # call of gpu function
            func(d_x1, self.c_inv, n, self.preFac, d_w_norm_lambdas,  d_logSum, d_kde_val_la, block=(int(bx),1,1), grid=(int(gx),1,1))

        # backward copy from gpu to cpu memory
        self.cuda.memcpy_dtoh(h_logSum, d_logSum)
        self.cuda.memcpy_dtoh(h_kde_val_la, d_kde_val_la)

        self.logSum = sum(h_logSum)
        self.invGlob = 1.0/np.exp(self.logSum)
        self.lambdas = np.array(1.0/np.power(self.invGlob*np.array(h_kde_val_la), self.alpha))

    def cuda_kde(self, points, weights=True):
        """Calculate kde values using CUDA implementation

        Parameters
        ----------
        points
        weights : bool

        """
        from pycuda.compiler import SourceModule

        if len(np.array(points).shape) == 1:
            self.points = np.array([points])
        else:
            self.points = np.array(points)
        self.d_pt, self.m = self.points.shape

        # conversion of python variables
        n = np.int32(self.n)
        m = np.int32(self.m)
        kde_val = np.zeros(self.m)

        h_preFac = np.array(self.preFac).astype(np.float64)
        h_w_norm = np.array(self.w_norm).astype(np.float64)
        h_kde_val = np.array(kde_val).astype(np.float64)

        # reservation of memory on gpu
        d_preFac = self.cuda.mem_alloc(h_preFac.nbytes)
        d_w_norm = self.cuda.mem_alloc(h_w_norm.nbytes)
        d_kde_val = self.cuda.mem_alloc(h_kde_val.nbytes)

        # memory copy to gpu
        self.cuda.memcpy_htod(d_preFac, h_preFac)
        self.cuda.memcpy_htod(d_w_norm, h_w_norm)
        self.cuda.memcpy_htod(d_kde_val, h_kde_val)

        # dimension-dependent memory allocation
        if self.d == 2:
            h_x1 = np.array(self.data[0]).astype(np.float32)
            h_x2 = np.array(self.data[1]).astype(np.float32)
            h_y1 = np.array(self.points[0]).astype(np.float32)
            h_y2 = np.array(self.points[1]).astype(np.float32)
            d_x1 = self.cuda.mem_alloc(h_x1.nbytes)
            d_x2 = self.cuda.mem_alloc(h_x2.nbytes)
            d_y1 = self.cuda.mem_alloc(h_y1.nbytes)
            d_y2 = self.cuda.mem_alloc(h_y2.nbytes)
            self.cuda.memcpy_htod(d_x1, h_x1)
            self.cuda.memcpy_htod(d_x2, h_x2)
            self.cuda.memcpy_htod(d_y1, h_y1)
            self.cuda.memcpy_htod(d_y2, h_y2)
            addDeclare = "double ent2;"
            addParam = "const float *x2, const float *y2, const double c11, const double c12, const double c21, const double c22"
            calculation = """
                ent1 = x1[j]-y1[idx];
                ent2 = x2[j]-y2[idx];
                thisKde += w_norm[j] * exp(preFac[j] * (ent1*(c11*ent1+c12*ent2) + ent2*(c21*ent1+c22*ent2)));
            """
        elif self.d == 1:
            h_x1 = np.array(self.data[0]).astype(np.float32)
            h_y1 = np.array(self.points[0]).astype(np.float32)
            d_x1 = self.cuda.mem_alloc(h_x1.nbytes)
            d_y1 = self.cuda.mem_alloc(h_y1.nbytes)
            self.cuda.memcpy_htod(d_x1, h_x1)
            self.cuda.memcpy_htod(d_y1, h_y1)
            addParam = "const double c"
            addDeclare = ""
            calculation = """
                ent1 = x1[j]-y1[idx];
                thisKde +=  w_norm[j] * exp(preFac[j] * c * pow(ent1, 2));
            """

        # define executed function
        mod = SourceModule("""
            __global__ void CalcKde(const float *x1, const float *y1, """+addParam+""", const int n, const int m, const double *preFac, const double *w_norm, double *kde){
                int idx = threadIdx.x + blockIdx.x*blockDim.x;
                if (idx < m){
                    double thisKde, ent1;
                    """+addDeclare+"""
                    int j;
                    thisKde = 0.0;
                    for (j=0; j < n; j++) {
                        """+calculation+"""
                    } // for
                    kde[idx] = thisKde;
                } // if
                __syncthreads();
            } // CalcKde_2d
        """)

        if n >= 512:
            bx = np.int32(512)
        else:
            bx = np.int32(n)
        gx = np.int32(self.m/bx)
        if n/bx != 0.0: gx += 1

        # code compiling
        func = mod.get_function("CalcKde")
        if self.d == 2:
            # call of gpu function
            func(d_x1, d_y1, d_x2, d_y2, self.c_inv[0][0], self.c_inv[1][0], self.c_inv[0][1], self.c_inv[1][1], n, m, d_preFac, d_w_norm, d_kde_val, block=(int(bx),1,1), grid=(int(gx),1,1))
        elif self.d == 1:
            # call of gpu function
            func(d_x1, d_y1, self.c_inv, n, m, d_preFac, d_w_norm, d_kde_val, block=(int(bx),1,1), grid=(int(gx),1,1))

        # backward copy from gpu to cpu memory
        self.cuda.memcpy_dtoh(h_kde_val, d_kde_val)

        self.values = h_kde_val
