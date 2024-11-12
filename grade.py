import unittest
import numpy as np
import importlib.util
import os
import pandas as pd
import q3_sol as q3org

class TestQ3(unittest.TestCase):
    marks = 0
    failed_cases = []

    @classmethod
    def setUpClass(cls):
        cls.marks = 0
        cls.failed_cases = []

    @classmethod
    def tearDownClass(cls):
        # Display total marks and failed cases for each test at the end
        print(f"Total marks: {cls.marks}")
        if cls.failed_cases:
            print(f"Failed test cases: {cls.failed_cases}")
        else:
            print("All test cases passed!")

    def run_test(self, func, *args, expected, test_case_number):
        try:
            if type(expected) == tuple:
                res = func(*args)
                if all(np.array_equal(a, b) for a, b in zip(res, expected)):
                    self.__class__.marks += 2
                elif np.array_equal(res[0], expected[0]): 
                    self.__class__.marks += 1
                    self.__class__.failed_cases.append(str(test_case_number)+ ' partial')
                    
                else:
                    self.__class__.failed_cases.append(test_case_number)
            else:
                if np.allclose(func(*args), expected):
                    self.__class__.marks += 1
                else:
                    self.__class__.failed_cases.append(test_case_number)
        except Exception as e:
            self.__class__.failed_cases.append(test_case_number)

    def test_d_lin_reg_th(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        th = np.array([1, 2])
        self.run_test(self.q3.d_lin_reg_th, X, y, th, expected=X, test_case_number=1)

    def test_d_square_loss_th(self):
        X = np.array([[2., 3.], [4., 5.]])
        y = np.array([[2.5, 3.5]])
        th = np.array([[0.5], [0.25]])
        th0 = np.array([[1.]])
    
        # Expected output based on these inputs
        expected = -2*(y - q3org.lin_reg(X, th, th0))*q3org.d_lin_reg_th(X, th, th0)
        self.run_test(self.q3.d_square_loss_th, X, y, th, th0, expected=expected, test_case_number=2)

    def test_d_mean_square_loss_th(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([[3, 7]])
        th = np.array([[0.5], [1.0]])
        th0 = np.array([[1]])
        n = X.shape[1]  # number of samples
        # Assuming d_square_loss_th(x, y, th, th0) calculates the gradient for each example
        n = X.shape[1]  # number of samples
        expected = (1 / n) * np.sum(q3org.d_square_loss_th(X, y, th, th0), axis=1, keepdims=True)
        self.run_test(self.q3.d_mean_square_loss_th, X, y, th, th0, expected=expected, test_case_number=3)

    def test_d_lin_reg_th0(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        th = np.array([[1, 2, 3], [4, 5, 6]])
        th0 = np.array([[1, 2, 3]])
        expected = np.ones(X.shape[1])
        self.run_test(self.q3.d_lin_reg_th0, X, th, th0, expected=expected, test_case_number=4)

    def test_d_square_loss_th0(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([[1, 2]])
        th = np.array([[1], [2], [3]])
        th0 = np.array([[2]])
        expected = -2*(y - q3org.lin_reg(X, th, th0))
        self.run_test(self.q3.d_square_loss_th0, X, y, th, th0, expected=expected, test_case_number=5)

    def test_d_mean_square_loss_th0(self):
        X = np.array([[1, 2,3], [3, 4,6]])
        y = np.array([[1, 5, 3]])
        th = np.array([[1],[4]])
        th0 = np.array([[ 2]])
        n = X.shape[1]  # number of samples
        expected =  (1 / n) * np.sum(q3org.d_square_loss_th0(X, y, th, th0))
        self.run_test(self.q3.d_mean_square_loss_th0, X, y, th, th0, expected=expected, test_case_number=6)

    def test_d_ridge_obj_th(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([[5, 2]])
        th = np.array([[1],[3],[5],[2]])
        th0 = np.array([[ 7]])
        lam = 0.3
        expected = q3org.d_mean_square_loss_th(X, y, th, th0) + 2*lam*th
        self.run_test(self.q3.d_ridge_obj_th, X, y, th,th0,lam, expected=expected, test_case_number=7)

    def test_d_ridge_obj_th0(self):
        X = np.array([[1, 2, 6], [3, 5, 6], [5, 6, 7]])
        y = np.array([[7, 2, 3]])
        th = np.array([[1], [4], [7]])
        th0 = np.array([[1]])
        lam = 0.9
        expected = q3org.d_mean_square_loss_th0(X, y, th, th0)
        self.run_test(self.q3.d_ridge_obj_th0, X, y, th, th0, lam, expected=expected, test_case_number=8)

    def test_sgd(self):
        def downwards_line():
            X = np.array([[0.0, 0.1, 0.2, 0.3, 0.42, 0.52, 0.72, 0.78, 0.84, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0]])
            y = np.array([[0.4, 0.6, 1.2, 0.1, 0.22, -0.6, -1.5, -0.5, -0.5, 0.0]])
            return X, y
    
        X, y = downwards_line()
        
        def J(Xi, yi, w):
            # translate from (1-augmented X, y, theta) to (separated X, y, th, th0) format
            return float(q3org.ridge_obj(Xi[:-1,:], yi, w[:-1,:], w[-1:,:], 0))
        
        def dJ(Xi, yi, w):
            def f(w): return J(Xi, yi, w)
            return q3org.num_grad(f)(w)
        
        w0 = np.zeros((X.shape[0], 1))
        step_size_fn = lambda i: 0.01 / (1 + 0.001 * i)
        max_iter = 5


        w = w0  # Initialize weight vector
        fs = []  # List to store cost function values at each step
        ws = [w.copy()]  # List to store weight vectors at each step

        # seed random number generator for reproducibility
        np.random.seed(0)

        for i in range(max_iter):
            # Choose a random data point index
            idx = np.random.randint(0, X.shape[1])
            Xi = X[:, idx:idx+1]  # Single column vector from X
            yi = y[:, idx:idx+1]  # Corresponding single label

            # Compute cost and gradient for this data point
            cost = J(Xi, yi, w)
            gradient = dJ(Xi, yi, w)

            # Update the weight vector using the computed gradient
            step_size = step_size_fn(i)
            w = w - step_size * gradient

            # Store the cost and weight for tracking
            fs.append(cost)
            ws.append(w.copy())

        expected = (w, fs, ws)
        # set the seed for q3.sgd numpy seed to 0
        np.random.seed(0)
        self.run_test(self.q3.sgd, X, y, J, dJ, w0, step_size_fn, max_iter, expected=expected, test_case_number=9)

def load_and_run_tests(base_folder):
    results = []

    for filename in os.listdir(base_folder):
        if filename == "q3_sample.py":
            a=1
        # Full path of the Python file
        file_path = os.path.join(base_folder, filename)
        
        # Check if the file is a Python file
        if filename.endswith(".py"):
            # Check if file exists
            if not os.path.exists(file_path):
                # Assign zero marks and None failed cases if the file is missing
                results.append((filename, 0, None))
                continue

            # Dynamically load the Python file
            spec = importlib.util.spec_from_file_location("q3", file_path)
            q3 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(q3)

            # Assign loaded module to TestQ3 for testing
            TestQ3.q3 = q3

            # Run all tests in TestQ3
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestQ3)
            runner = unittest.TextTestRunner()
            result = runner.run(suite)

            # Collect results
            total_marks = TestQ3.marks
            failed_cases = TestQ3.failed_cases.copy()

            # Reset for the next file
            TestQ3.marks = 0
            TestQ3.failed_cases = []

            # Append result with file name instead of folder name
            name = filename.split("_")[0]
            results.append((name, total_marks, failed_cases))

    # Store results in a DataFrame
    df = pd.DataFrame(results, columns=['Name', 'Marks', 'Failed Cases'])
    # sort values by Name
    df = df.sort_values(by='Name')
    # save to a csv file
    df.to_csv('./A1/results.csv', index=False)

if __name__ == '__main__':
    base_folder = "./all/"  # Folder where 'name 1', 'name 2', etc., are located
    load_and_run_tests(base_folder)