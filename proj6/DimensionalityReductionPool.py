from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
import numpy, time


class DimensionalityReductionPool:

    # common param
    random_state = None
    n_jobs = None
    runtime = None

    # PCA param
    pca_obj = None

    # LDA param
    lda_obj = None

    # Kernel PCA param
    gamma_kernel_pca = None
    kernel_pca_obj = None

    def __init__(self, random_state, n_jobs, gamme_kernel_pca):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.runtime = numpy.zeros(3)
        # PCA object
        self.pca_obj = PCA(n_components='mle', svd_solver='full', random_state=self.random_state)
        # LDA object
        self.lda_obj = LinearDiscriminantAnalysis(n_components=None)
        # Kernel PCA object
        self.gamma_kernel_pca = gamme_kernel_pca
        self.kernel_pca_obj = KernelPCA(n_components=None, kernel='rbf', gamma=self.gamma_kernel_pca,
                                        random_state=self.random_state, n_jobs=self.n_jobs)

    def PCA_fit(self, X):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting PCA on X")
        self.pca_obj.fit(X=X)
        self.runtime[0] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[0], 2)) + " seconds")

    def LDA_fit(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting LDA on X and Y")
        self.lda_obj.fit(X=X, y=Y)
        self.runtime[1] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[1], 2)) + " seconds")

    def KernelPCA_fit(self, X):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting Kernel PCA on X")
        self.kernel_pca_obj.fit(X=X)
        self.runtime[2] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[2], 2)) + " seconds")

    def Fit_all(self, X, Y):
        self.PCA_fit(X)
        self.LDA_fit(X, Y)
        self.KernelPCA_fit(X)

    def Transform_all(self, X):

        transformed_X_dict = {
            'PCA': self.pca_obj.transform(X),
            'LDA': self.lda_obj.transform(X),
            'KernelPCA': self.kernel_pca_obj.transform(X)
        }

        return transformed_X_dict

    def GetDimRedObjects(self):

        DimRedObj = {
            'PCA': self.pca_obj,
            'LDA': self.lda_obj,
            'KernelPCA': self.kernel_pca_obj
        }

        return DimRedObj



