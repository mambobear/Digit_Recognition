package recognition;

public class Matrix {

    public static double[][] product(double[][] W, double[][] A) {
        int w_rows = W.length;
        int w_cols = W[0].length;
        int a_rows = A.length;
        int a_cols = A[0].length;

        assert w_cols == a_rows;
        double[][] product = new double[w_rows][a_cols];
        for (int r = 0; r < w_rows; r++) {
            for (int c = 0; c < a_cols; c++) {
                double val = 0;
                for (int i = 0; i < w_cols; i++) {
                    val += W[r][i] * A[i][c];
                }
                product[r][c] = val;
            }
        }
        return product;
    }

    public static double[] product(double[][] W, double[] x) {
        int w_rows = W.length;
        int w_cols = W[0].length;

        assert w_cols == x.length;
        double[] product = new double[w_rows];
        for (int r = 0; r < w_rows; r++) {
            double val = 0;
            for (int i = 0; i < w_cols; i++) {
                val += W[r][i] * x[i];
            }
            product[r] = val;

        }
        return product;
    }

    public static double[][] computeTranspose(double[][] A) {
        int rows = A.length;
        int cols = A[0].length;

        double[][] transpose = new double[cols][rows];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                transpose[r][c] = A[c][r];
            }
        }
        return transpose;
    }

    public static double[][] sigmoid(double[][] M) {
        int rows = M.length;
        int cols = M[0].length;
        double[][] S = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                S[r][c] = 1 / (1 + Math.exp(-M[r][c]));
            }
        }

        return S;
    }

    public static double[][] add(double[][] A, double[][] B) {
        int a_rows = A.length;
        int a_cols = A[0].length;
        int b_rows = B.length;
        int b_cols = B[0].length;

        assert a_rows == b_rows && a_cols == b_cols;
        double[][] sum = new double[a_rows][a_cols];
        for (int r = 0; r < a_rows; r++) {
            for (int c = 0; c < a_cols; c++) {
                sum[r][c] = A[r][c] + B[r][c];
            }
        }
        return sum;
    }

    public static double[][] productElementwise(double[][] A, double[][] B) {
        int a_rows = A.length;
        int a_cols = A[0].length;
        int b_rows = B.length;
        int b_cols = B[0].length;

        assert a_rows == b_rows && a_cols == b_cols;
        double[][] sum = new double[a_rows][a_cols];
        for (int r = 0; r < a_rows; r++) {
            for (int c = 0; c < a_cols; c++) {
                sum[r][c] = A[r][c] * B[r][c];
            }
        }
        return sum;
    }

    public static double[][] subtract(double[][] A, double[][] B) {
        int a_rows = A.length;
        int a_cols = A[0].length;
        int b_rows = B.length;
        int b_cols = B[0].length;

        assert a_rows == b_rows && a_cols == b_cols;
        double[][] sum = new double[a_rows][a_cols];
        for (int r = 0; r < a_rows; r++) {
            for (int c = 0; c < a_cols; c++) {
                sum[r][c] = A[r][c] - B[r][c];
            }
        }
        return sum;
    }

    public static double[][] multiplyByScalar(double[][] M, double d) {
        int rows = M.length;
        int cols = M[0].length;
        double[][] product = new double[rows][cols];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                product[r][c] = M[r][c] * d;
            }
        }
        return product;
    }

    public static double normSquared(double[][] M) {
        int rows = M.length;
        int cols = M[0].length;

        double sum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                sum += Math.pow(M[r][c], 2);
            }
        }
        return sum;
    }

    public static double norm(double[][] M) {
        return Math.sqrt(normSquared(M));
    }

    public static void printMatrix(double[][] W) {
        int rows = W.length;
        int cols = W[0].length;

        System.out.printf("dim: %d x %d\n", rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                System.out.printf("%7.2f ", W[r][c]);
            }
            System.out.println();
        }
        System.out.println();
    }


}
