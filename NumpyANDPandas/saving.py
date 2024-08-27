import numpy as np

def main():

    np1 = np.array([1,2,3,4,5,6])
    np1 += 1
    np.save('mi-manruquito', np1)

    np2 = np.load('mi-manruquito.npy')
    print(np2)

    n1=np.array([[1,2,3,4],[6,7,8,9]])
    print(n1.shape)
    

main()