import Source.AI_Backend as AIB
import numpy as np
import matplotlib.pyplot as plt
print("Hello world")
def PlotKernel(K):
    n = 1
    m = K.shape[3]
    print(K.shape)
    for i in range(n):
        for j in range(m):
            img = K[:, :, :, i * m + j]
            img = img[..., ::-1]

            plt.subplot(n, m, i * m + j + 1)
            plt.axis("off")
            plt.imshow(img)
            plt.title(i * m + j + 1)

x = AIB.AI()
x.LoadData(r"C:\Users\Thomacdebabo\Desktop\VarroaData")
x.addModel((100,100,3))
#x.loadModel("VarroaI_2.hdf5")
#x.trainDataGen(10)
Kernels = x.getKernels()
print(x.evaluateModel())
print(x.make_prediction_ValData())


plt.show()

