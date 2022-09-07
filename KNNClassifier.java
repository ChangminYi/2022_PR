
/**
 * 20170454 Yi Changmin
 * Pattern Recognition k-NN classifier project
 */

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.PriorityQueue;
import java.util.StringTokenizer;

class Data {
    private static int SAME_DIST = 0;
    private static int DIFF_DIST = 6;

    /**
     * param[0] hours-per-week
     * param[1] workclass
     * param[2] marital-status
     * param[3] fnlwgt
     * param[4] age
     * param[5] education-num
     */
    public double[] param = new double[KNNClassifier.DATA_DIM];
    public int classInInt;

    /**
     * @param str one-line string read from .csv file
     */
    Data(final String str) {
        StringTokenizer parser = new StringTokenizer(str, ",");
        for (int i = 0; i < KNNClassifier.DATA_DIM; i++) {
            param[i] = Integer.parseInt(parser.nextToken());
        }
        classInInt = ClassConverter.stringToInt(parser.nextToken());
    }

    /**
     * @param lhs left side operand
     * @param rhs right side operand
     * @return euclid distance of two Data class
     */
    public static double getDistance(final Data lhs, final Data rhs) {
        double diff_0 = lhs.param[0] - rhs.param[0];
        double diff_1 = lhs.param[1] == rhs.param[1] ? Data.SAME_DIST : Data.DIFF_DIST;
        double diff_2 = lhs.param[2] == rhs.param[2] ? Data.SAME_DIST : Data.DIFF_DIST;
        double diff_3 = lhs.param[3] - rhs.param[3];
        double diff_4 = lhs.param[4] - rhs.param[4];
        double diff_5 = lhs.param[5] - rhs.param[5];

        return diff_0 * diff_0 + diff_1 * diff_1 + diff_2 * diff_2 + diff_3 * diff_3 + diff_4 * diff_4 + diff_5 * diff_5;
    }
}

class pqNode implements Comparable<pqNode> {
    public int classInInt;
    public double distance;

    pqNode(final int classInInt, final double distance) {
        this.classInInt = classInInt;
        this.distance = distance;
    }

    public int compareTo(pqNode other) {
        if (this.distance == other.distance)
            return 0;
        else
            return (this.distance < other.distance ? -1 : 1);
    }
}

class ClassConverter {
    final private static String CLASS_0_STRING = "satisfied";
    final private static String CLASS_1_STRING = "unsatisfied";

    /**
     * @param classInString class in String
     * @return class in integer
     */
    public static int stringToInt(final String classInString) {
        return classInString.equals(CLASS_0_STRING) ? 0 : 1;
    }

    /**
     * @param classInInt class in integer
     * @return class in String
     */
    public static String intToString(final int classInInt) {
        return classInInt == 0 ? CLASS_0_STRING : CLASS_1_STRING;
    }
}

public class KNNClassifier {
    final private static String OUTPUT_FILENAME = "20170454.csv";
    final private static int K_NUMBER = 61; // should be odd number because the number of class is just 2
    final public static int DATA_DIM = 6;

    private static File trainFile, testFile, outputFile;
    private static BufferedReader reader;
    private static BufferedWriter writer;

    private static ArrayList<Data> trainData = new ArrayList<>();
    private static ArrayList<Data> testData = new ArrayList<>();
    private static ArrayList<Integer> classifyResult = new ArrayList<>();
    private static double correctCount = 0;

    private static double[] mean = new double[DATA_DIM];
    private static double[] stddev = new double[DATA_DIM];

    /**
     * @param args[0] training file name
     * @param args[1] test file name
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 2) { // checking errors of arguments
            System.out.println("Parameter format is wrong");
            return;
        } else {
            trainFile = new File(args[0]);
            testFile = new File(args[1]);
            if (!trainFile.exists()) {
                System.out.println("There's no training data named " + args[0]);
                return;
            } else if (!testFile.exists()) {
                System.out.println("There's no test data named " + args[1]);
                return;
            }
        }

        reader = new BufferedReader(new FileReader(trainFile)); // reading trainig data and test data
        for (String line; (line = reader.readLine()) != null;) {
            trainData.add(new Data(line));
        }
        reader.close();
        reader = new BufferedReader(new FileReader(testFile));
        for (String line; (line = reader.readLine()) != null;) {
            testData.add(new Data(line));
        }
        reader.close();

        getMeanAndStddev();
        normalizeData(); // normalize all data to Z-Score

        long startTime = System.nanoTime();
        for (final Data toClassify : testData) { // classifying test data
            int curResult = knnClassify(toClassify);
            classifyResult.add(curResult);
            if (curResult == toClassify.classInInt)
                correctCount += 100;
        }
        long endTime = System.nanoTime();

        outputFile = new File(OUTPUT_FILENAME); // writing result in output file and count correctness
        if (!outputFile.exists())
            outputFile.createNewFile();
        writer = new BufferedWriter(new FileWriter(outputFile));
        for (final Integer cur : classifyResult) {
            writer.write(ClassConverter.intToString(cur) + "\n");
        }
        writer.flush();
        writer.close();

        // print accuracy and running time
        System.out.printf("Accuracy : %.3f", correctCount / testData.size());
        System.out.println("%");
        System.out.printf("Runtime : %.3f msec", ((double)endTime - startTime) / 1000000);
    }

    /**
     * Using priority queue, it's possible to select neighbors in O(nklogk) time.
     * However, k is very small compared to n, it can be approximated to O(n) time.
     * 
     * @param toClassify : Data class from test data to classify
     */
    private static int knnClassify(final Data toClassify) {
        int[] neighborCount = new int[2];
        PriorityQueue<pqNode> pq = new PriorityQueue<>(Collections.reverseOrder());

        for (final Data cur : trainData) {
            double curDistance = Data.getDistance(toClassify, cur);
            if (pq.size() < K_NUMBER) {
                pq.add(new pqNode(cur.classInInt, curDistance));
            } else if (pq.peek().distance > curDistance) {
                pq.add(new pqNode(cur.classInInt, curDistance));
                pq.poll();
            }
        }

        Arrays.fill(neighborCount, 0);
        while (!pq.isEmpty()) {
            neighborCount[pq.poll().classInInt]++;
        }

        return (neighborCount[0] > neighborCount[1] ? 0 : 1);
    }

    /**
     * Z-Score Normalization.
     * Data's dimension size are not equal, so it need to be normalized.
     */
    private static void normalizeData() {
        for(Data cur : trainData) {
            for(int i = 0; i < DATA_DIM; i++) {
                if(i != 1 && i != 2) {
                    cur.param[i] = (cur.param[i] - mean[i]) / stddev[i];
                }
            }
        }
        for(Data cur : testData) {
            for(int i = 0; i <DATA_DIM;i++) {
                if(i != 1 && i != 2) {
                    cur.param[i] = (cur.param[i] - mean[i]) / stddev[i];
                }
            }
        }
    }

    /**
     * Get mean and standard deviation of each data dimension
     */
    private static void getMeanAndStddev() {
        double[] sum = new double[DATA_DIM];
        Arrays.fill(sum, 0);

        // mean
        for(Data cur : trainData) {
            for(int i = 0; i < DATA_DIM; i++) {
                sum[i] += cur.param[i];
            }
        }
        for(int i = 0; i < DATA_DIM; i++) {
            mean[i] = sum[i] / trainData.size();
        }

        // standard deviation
        Arrays.fill(sum, 0);
        for(Data cur : trainData) {
            for(int i = 0; i < DATA_DIM; i++) {
                sum[i] += (cur.param[i] - mean[i]) * (cur.param[i] - mean[i]);
            }
        }
        for(int i = 0; i < DATA_DIM; i++) {
            stddev[i] = Math.sqrt(sum[i] / trainData.size());
        }
    }
}