import numpy as np 


def main():
    #Para stackear arreglos en numpy, tienen que tener la misma dimension
    np1 = np.array([1,2,3,4])
    np2 = np.array([10,11,12,13])
    #El VStacked es para apilar los arreglos de manera vertical
    npVStacked = np.vstack([np1, np2])
    #El HStacked es para apilar los arreglos de manera horizontal
    npHStacked = np.hstack([np1, np2])
    #El ColumnStacked es para apilar los arreglos de manera vertical
    npColumnStacked = np.column_stack([np1, np2])

    print("VStack\n")
    print(npVStacked)
    print("\nHStack\n")
    print(npHStacked)
    print("\nColumnStack\n")
    print(npColumnStacked)



main()