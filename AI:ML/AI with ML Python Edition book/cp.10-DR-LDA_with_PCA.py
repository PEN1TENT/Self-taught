# LDA with PCA on ORL (SSS problem)
import numpy as np
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("PCA", "cp.10-DR-PCA.py")
PCA_module = importlib.util.module_from_spec(spec)
sys.modules["PCA_module"] = PCA_module
spec.loader.exec_module(PCA_module)
PCA = PCA_module.PCA

spec2 = importlib.util.spec_from_file_location("kNN_module", "cp.4-kNN-kNN.py")
kNN_module = importlib.util.module_from_spec(spec2)
sys.modules["kNN_module"] = kNN_module
spec.loader.exec_module(kNN_module)
kNN = kNN_module.kNN

spec3 = importlib.util.spec_from_file_location(
    "ORL_dataset_module", "cp.10-DR-ORL_dataset.py"
)
ORL_dataset_module = importlib.util.module_from_spec(spec3)
sys.modules["ORL_dataset_module"] = ORL_dataset_module
spec3.loader.exec_module(ORL_dataset_module)

spec4 = importlib.util.spec_from_file_location("LDA", "cp.10-DR-LDA.py")
LDA_module = importlib.util.module_from_spec(spec4)
sys.modules["LDA_module"] = LDA_module
spec.loader.exec_module(LDA_module)
LDA = LDA_module.LDA

Xtrain, Ltrain, Xtest, Ltest = ORL_dataset_module.load(split_train_test = 0.5)
n_PC = np.min(Xtrain.shape)
A_pca, mu_pca, Ytrain_pca = PCA(Xtrain, n_PC) # get all PCs
Ytest_pca = (Xtest - mu_pca) @ A_pca # project test set 
rate = np.zeros((n_PC, n_PC))
for i in range(n_PC):
    A_lda, mu_lda, Ytrain_lda = LDA(Ytrain_pca[:, :i + 1], Ltrain, i + 1)
    for j in range(i + 1):
        Ytest_lda = (Ytest_pca[L, :i + 1] - mu_lda) @ A_lda[:, :j + 1]
        Ztest = kNN(Ytrain_lda[:, :j + 1], Ltrain, Ytest_lda)
        rate[i, j] = np.sum(Ztest == Ltest) / len(Ztest) * 100
        print(i, j, rate[i, j])
i, j = np.unrabel_index(rate.argmax(), rate.shape)
print('Best accuracy:', rate[i, j], 'PCA:', i, 'LDA:', j)

i = np.arange(1, n_PC + 1)
j = np.arange(1, n_PC + 1)
I, J = np.meshgrid(i, j)
rate[rate == 0] = np.nan

from mayavi import mlab as plt 
plt.mesh(I, J, rate)
plt.axes(xlabel = 'Dimension of PCA', ylabel = 'Dimension of LDA', zlabel = 'Accuracy (%)')
plt.show()
    