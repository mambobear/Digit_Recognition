package recognition;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

public class NeuralNetwork implements Serializable {

    private int nSamples = 0;
    private double[][][] weightMatrices; // ok
    private double[][][] layers; // ok

    private int nLayers;  // ok
    private int[] layerSizes; // ok

    private double[][] idealOutput; //ok

    boolean isTrained = false;

    double alpha = 0.0001;
    //double alpha = 0.0005;

    public NeuralNetwork(String trainingDataFile, int[] layerSizes) {

        HashMap<Integer, ArrayList<double[]>> samples = new HashMap<>();

        File sampleDir = new File(trainingDataFile);
        File[] contents = sampleDir.listFiles();
        assert contents != null;
        for (File f : contents) {
            if (!f.isFile()) continue;
            readInTestSample(f, samples);
        }

        int minSampleSize = Integer.MAX_VALUE;
        for (Integer key : samples.keySet()) {
            ArrayList<double[]> digitSamples = samples.get(key);
            int size = digitSamples.size();
            if (size < minSampleSize) minSampleSize = size;

//            System.out.println(key + ": " + size);
//            Random rand = new Random();
//            int idx = rand.nextInt(size);
//            printSampleData(digitSamples.get(idx));
//            System.out.println();
        }
//        System.out.println("minSampleSize = " + minSampleSize);
//        System.out.println();

        int nSamples = minSampleSize;
        //nSamples = (int) Math.floor(minSampleSize * 0.5);
        //nSamples = 1000;
        nSamples = 1500;
        double[][] output = createIdealOutput(nSamples);
        // Matrix.printMatrix(output);

        double[][] trainingData = createTrainingData(samples, nSamples);

        initNetwork(trainingData, output, layerSizes);
    }

    private void initNetwork(double[][] trainingData, double[][] output, int[] layerSizes) {

        //this.trainingData = trainingData;

        // make sure testData and idealOutput dimensions are consistent with layer sizes
        assert trainingData.length == layerSizes[0] + 1;
        assert output.length == layerSizes[layerSizes.length - 1];

        assert trainingData[0].length == output[0].length;
        this.nSamples = trainingData[0].length;

        this.idealOutput = output;
        this.layerSizes = layerSizes;
        this.nLayers = layerSizes.length;

        // init layers, with first layer equal training data
        this.layers = new double[this.nLayers][][];
        this.layers[0] = trainingData;

        this.weightMatrices = new double[this.nLayers - 1][][];
        Random rand = new Random(System.currentTimeMillis());

        // init weight matrices
        // init W_0, ..., W_{nLayers - 2} => add last row [0, 0, ..., 0, 1]
        for (int i = 0; i < this.nLayers - 2; i++) {
            int rows = this.layerSizes[i + 1];
            int cols = this.layerSizes[i];
            this.weightMatrices[i] = new double[rows + 1][cols + 1];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c <= cols; c++) {
                    //double val = rand.nextGaussian();
                    double val = 2 * Math.random() - 1;
                    this.weightMatrices[i][r][c] = val;
                }
            }
            for (int c = 0; c < cols; c++) {
                this.weightMatrices[i][rows][c] = 0;
            }
            this.weightMatrices[i][rows][cols] = 1;
        }
        // init W_{nLayers - 1}
        int rows = this.layerSizes[this.nLayers - 1];
        int cols = this.layerSizes[this.nLayers - 2] + 1;
        this.weightMatrices[this.nLayers - 2] = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                //double val = rand.nextGaussian();
                double val = 2 * Math.random() - 1;
                this.weightMatrices[this.nLayers - 2][r][c] = val;
                //this.weightMatrices[this.nLayers - 2][r][c] = 1 / (1 + Math.exp(-val));
            }
        }

        this.updateLayers();
//        System.out.println("NeuralNetwork constructor: ");
//        Matrix.printMatrix(this.layers[this.nLayers - 1]);
//        System.out.println();
    }

    private double[][] createTrainingData(HashMap<Integer, ArrayList<double[]>> samples, int nSamples) {
        Integer[] keys = new Integer[10];
        samples.keySet().toArray(keys);
        Arrays.sort(keys);

        int trainingVectorSize = 28 * 28;
        double[][] trainingData = new double[nSamples * 10][trainingVectorSize + 1];
        for (Integer key : keys) {
            ArrayList<double[]> digitSamples = samples.get(key);
            for (int idx = 0; idx < nSamples; idx++) {
                double[] vec = new double[trainingVectorSize + 1];
                System.arraycopy(digitSamples.get(idx), 0, vec, 0, trainingVectorSize);
                vec[trainingVectorSize] = 1; // bias value
                trainingData[nSamples * key + idx] = vec;
            }
        }
        return Matrix.computeTranspose(trainingData);
    }

    private double[][] createIdealOutput(int nSamples) {
        double[][] outputVectors = new double[nSamples * 10][10];
        double[][] idealVectors = {
                {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        };

        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < nSamples; j++) {
                double[] vec = new double[10];
                System.arraycopy(idealVectors[i], 0, vec, 0, 10);
                outputVectors[nSamples * i + j] = vec;
            }
        }

        return Matrix.computeTranspose(outputVectors);
    }

    private void printSampleData(double[] list) {
        int rowIdx = 0;
        for (double i : list) {
            if (i == 0) {
                //System.out.print("_   ");
                System.out.print("_");
            } else {
                //System.out.printf("%.1f ", i);
                System.out.print("*");
            }
            rowIdx++;
            if (rowIdx == 28) {
                System.out.println();
                rowIdx = 0;
            }
        }
    }

    private void readInTestSample(File f, HashMap<Integer, ArrayList<double[]>> samples) {
        ArrayList<Double> sample = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(f))) {
            String line = reader.readLine();
            while (line != null) {
                // process line
                String[] values = line.split("\\s+");
                if (values.length == 1) {
                    Integer key = Integer.parseInt(values[0]);
                    double[] array = sample.stream().mapToDouble(i -> i).toArray();
                    if (!samples.containsKey(key)) {
                        ArrayList<double[]> digitSamples = new ArrayList<>();
                        digitSamples.add(array);
                        samples.put(key, digitSamples);
                    } else {
                        samples.get(key).add(array);
                    }
                } else {
                    for (String str : values) {
                        sample.add(Integer.parseInt(str) / 255.0);
                    }
                }
                line = reader.readLine();
            }
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
    }

    public void train() {

        System.out.println("Training ...");
        if (this.isTrained) return;

        double error = 1;
        double outputErr = 2000;
        int gen = 0;

        //while (gen <= 1000) {
        while (outputErr > 21) {
            error = updateWeights();
            updateLayers();
            outputErr = outputError();

            gen++;
            if (gen % 100 == 0) {
                System.out.printf("gen = %d, weight err = %.10f, output err = %.10f\n", gen, error, outputErr);
            }
        }
        System.out.printf("gen = %d, weight err = %.10f, output err = %.10f\n", gen, error, outputErr);
        updateLayers();
        this.isTrained = true;
        //Matrix.printMatrix(this.layers[this.nLayers - 1]);
    }

    private double outputError() {
        double[][] layer = this.layers[this.nLayers - 1];
        double diff = 0;
        int rows = layer.length;
        int cols = layer[0].length;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                diff += Math.pow(this.idealOutput[r][c] - layer[r][c], 2);
            }
        }
        return Math.sqrt(diff);
    }

    private void updateLayers() {
        for (int i = 1; i < this.nLayers; i++) {
            double[][] matrix = Matrix.product(this.weightMatrices[i - 1], this.layers[i - 1]);

            int rows = matrix.length;
            int cols = matrix[0].length;
            if (i < this.nLayers - 1) {
                rows -= 1;
            }
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    matrix[r][c] = 1 / (1 + Math.exp(-matrix[r][c]));
                }
            }

            this.layers[i] = matrix;
        }
    }

    private double updateWeights() {
        int rows;
        int cols;
        double weightDiffNormsSquared = 0;

        // compute the last delta
        int layerIdx = this.nLayers - 1; // last (output) layer
        double[][] layer = this.layers[layerIdx];
        rows = layer.length;
        cols = layer[0].length;
        double[][] delta = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double val = layer[r][c];
                delta[r][c] = (this.idealOutput[r][c] - val) * val * (1 - val);
            }
        }
        // compute the last dW
        weightDiffNormsSquared += updateMatrixWeight(this.weightMatrices[layerIdx - 1], this.layers[layerIdx - 1], delta);

        // all other deltas
        for (int i = layerIdx - 1; i >= 1; i--) {
            delta = updateDelta(this.weightMatrices[i], delta, this.layers[i]);
            weightDiffNormsSquared += updateMatrixWeight(this.weightMatrices[i - 1], this.layers[i - 1], delta);
        }
        return Math.sqrt(weightDiffNormsSquared);
    }

    private double[][] updateDelta(double[][] weightMatrix, double[][] delta, double[][] layer) {
        // compute ∆_i
        int rows = layer.length;
        int cols = layer[0].length;

        int deltaRows = delta.length;

        // W^T * ∆
        double[][] newDelta = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double val = layer[r][c];
                newDelta[r][c] = 0;
                for (int i = 0; i < deltaRows; i++) {
                    newDelta[r][c] += weightMatrix[i][r] * delta[i][c];
                }
                newDelta[r][c] *= val * (1 - val);
            }
        }
        return newDelta ;
    }

    private double updateMatrixWeight(double[][] weightMatrix, double[][] layer, double[][] delta) {
        int rows = weightMatrix.length;
        int cols = weightMatrix[0].length;

        double weightDiffSquared = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double dW = 0;
                for (int j = 0; j < this.nSamples; j++) {
                    dW += delta[r][j] * layer[c][j];
                }
                dW *= this.alpha;
                weightMatrix[r][c] += dW;
                weightDiffSquared += Math.pow(dW, 2);
            }
        }
        return weightDiffSquared;
    }

    public void printStats() {
        System.out.println("Layer Sizes:");
        System.out.println(Arrays.toString(this.layerSizes));

        for (int i = 0; i < this.weightMatrices.length; i++) {
            System.out.printf("W_%d\n", i);
            Matrix.printMatrix(this.weightMatrices[i]);
        }

        for (int i = 0; i < this.nLayers; i++) {
            System.out.printf("L_%d\n", i);
            Matrix.printMatrix(this.layers[i]);
        }
    }

    public int process(double[] input) {


        double[] result = new double[input.length + 1];
        System.arraycopy(input, 0, result, 0, input.length);
        result[input.length] = 1;

        for (int i = 0; i < this.weightMatrices.length; i++) {
            result = Matrix.product(this.weightMatrices[i], result);
            if (i < this.weightMatrices.length - 1) {
                for (int j = 0; j < result.length - 1; j++) {
                    result[j] = 1 / (1 + Math.exp(-result[j]));
                }
            } else {
                for (int j = 0; j < result.length; j++) {
                    result[j] = 1 / (1 + Math.exp(-result[j]));
                }
            }
        }

        double maxVal = Integer.MIN_VALUE;
        int maxIdx = -1;
        for (int i = 0; i < result.length; i++) {
            double val = result[i];
            if (val > maxVal) {
                maxVal = val;
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    public void writeToFile(String filename) {
        try {
            SerializationUtils.serialize(this, filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void test(String trainingDataFile) {

        System.out.print("Guessing ...");
        HashMap<Integer, ArrayList<double[]>> samples = new HashMap<>();

        File sampleDir = new File(trainingDataFile);
        File[] contents = sampleDir.listFiles();
        assert contents != null;
        for (File f : contents) {
            if (!f.isFile()) continue;
            readInTestSample(f, samples);
        }

        int nCorrect = 0;
        int total = 0;
        for (Integer key : samples.keySet()) {
            for (double[] vec : samples.get(key)) {
                total++;
                if (this.process(vec) == key) {
                    nCorrect++;
                }
            }
        }
        int percentage = (int) Math.round(((double) nCorrect / (double) total) * 100);
        System.out.printf("The network prediction accuracy: %d/%d, %d%s\n", nCorrect, total, percentage, "%");
    }
}