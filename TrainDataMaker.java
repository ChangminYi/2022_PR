
/**
 * 20170454 Yi Changmin
 * Pattern Recognition k-NN classifier project: train/test dataset maker
 * it usese Math.random() to determine if data is for test.
 */

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;

public class TrainDataMaker {
    final private static int fileLength = 20000;
    final private static int trainLength = 18000;
    final private static int testLength = 2000;
    final private static int fileCount = 10;

    private static BufferedReader reader;
    private static BufferedWriter trainWriter, testWriter;
    private static File originalFile, trainFile, testFile;

    private static String[] fileData = new String[fileLength];
    private static boolean[] isTestData = new boolean[fileLength];
    public static void main(String[] args) throws Exception {
        if(args.length != 1) {
            System.out.println("Argument format is wrong");
            return;
        } else {
            originalFile = new File(args[0]);
            if(!originalFile.exists()) {
                System.out.println("There's no file named " + args[0]);
                return;
            } else {
                reader = new BufferedReader(new FileReader(originalFile));
            }
        }

        for(int i = 0; i < fileLength; i++) {
            fileData[i] = reader.readLine();
        }
        reader.close();

        for(int i = 0; i < fileCount; i++) {
            Arrays.fill(isTestData, false);

            for(int cnt = 0, idx; cnt < testLength; ) {
                idx = (int)(Math.random() * fileLength);
                if(!isTestData[idx]) {
                    cnt++;
                    isTestData[idx] = true;
                }
            }

            trainFile = new File("train_" + i + ".csv");
            testFile = new File("test_" + i + ".csv");
            if(!trainFile.exists()) trainFile.createNewFile();
            if(!testFile.exists()) testFile.createNewFile();
            trainWriter = new BufferedWriter(new FileWriter(trainFile, true));
            testWriter = new BufferedWriter(new FileWriter(testFile, true));

            for(int j = 0, testCnt = 0, trainCnt = 0; j < fileLength; j++) {
                if(isTestData[j]) {
                    testWriter.write(fileData[j]);
                    if(++testCnt < testLength) {
                        testWriter.newLine();
                    }
                }
                else {
                    trainWriter.write(fileData[j]);
                    if(++trainCnt < trainLength) {
                        trainWriter.newLine();
                    }
                }
            }

            trainWriter.flush();
            testWriter.flush();
            trainWriter.close();
            testWriter.close();
        }

        return;
    }
}
