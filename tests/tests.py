import pre
import numpy as np
import sys
sys.path.append('../duq/')


# def plotting_tests_MC_Dropout():
#    import test_MC_Dropout
#    return test_MC_Dropout

def check_env():
    import check_env
    return check_env


def normalise_test():
    data1 = np.array([[1, 2, 3],
                      [2, 3, 4],
                      [7, 4, 1]])
    mean1 = np.array([1, 1, 5])
    std1 = np.array([0.5, 0.1, 3])
    res1 = pre.normalise(data1, mean1, std1)
    true1 = np.array([[0, 10, -0.666667],
                      [2, 20, -0.333333],
                      [12, 30, -1.33333]])

    # data2 = pd.DataFrame({})

    assert np.allclose(res1, true1)

    mean2 = 1
    std2 = 0.5
    res2 = pre.normalise(data1, mean2, std2)
    true2 = np.array([[0, 2, 4],
                      [2, 4, 6],
                      [12, 6, 0]])
    assert np.allclose(res2, true2)


def test_API():
    return NotImplemented
