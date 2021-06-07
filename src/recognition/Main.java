package recognition;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {

        System.out.println("Working Directory = " + System.getProperty("user.dir"));
        NeuralNetwork network = null;
        try {
            network = (NeuralNetwork) SerializationUtils.deserialize("network");
            System.out.println("Network initialized");
        } catch (IOException | ClassNotFoundException | ClassCastException e) {
            System.out.println("No data present");
            String samplesDirStr = "/Users/andreizherebtsov/Dropbox/CS/Java/JetBrains_Java/Projects/resources/data";
            network = new NeuralNetwork(samplesDirStr, new int[]{28 * 28, 16, 16, 10});
            network.train();
            network.writeToFile("network");
        }

        stage(network);
    }

    public static void stage(NeuralNetwork network) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("in stage ...");
        while (true) {
            System.out.println("1. Learn the network");
            System.out.println("2. Guess all the numbers");
            System.out.println("3. Guess number from text file");
            System.out.println("0. Exit");
            System.out.print("Your choice: ");
            int choice = Integer.parseInt(scanner.nextLine().strip());
            if (choice == 1) {
                System.out.println("Learning...");
                network.train();
                network.writeToFile("network");
                System.out.println("Done! Saved to the file.");
            } else if (choice == 2) {
                network.test("/Users/andreizherebtsov/Dropbox/CS/Java/JetBrains_Java/Projects/resources/data");
                //guessNumber(network);
            } else if (choice == 3) {
                System.out.print("Enter filename: ");
                String fileName = scanner.nextLine().strip();
                guessNumber(network, fileName);
                return;
            } else if (choice == 0) {
                System.out.print("Exiting ...");
                return;
            } else {
                System.out.println("Invalid input");
            }
        }
    }

    private static void guessNumber(NeuralNetwork network, String fileName) {
        ArrayList<Double> sample = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            String line = reader.readLine();
            while (line != null) {
                // process line
                //System.out.println(line);
                String[] values = line.split("\\s+");
                for (String str : values) {
                    sample.add(Integer.parseInt(str) / 255.0);
                }
                line = reader.readLine();
            }
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
        double[] array = sample.stream().mapToDouble(i -> i).toArray();
        System.out.println("This number is " + network.process(array));
    }

    private static void guessNumber(NeuralNetwork network) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Input grid:");
        double[] input = new double[15];
        int idx = 0;
        for (int r = 0; r < 5; r++) {
            String line = scanner.nextLine();
            char[] pixels = line.toCharArray();
            for (int c = 0; c < 3; c++) {
                if (pixels[c] == '_') input[idx] = 0;
                else if (pixels[c] == 'X') input[idx] = 1;
                idx++;
            }
        }
        System.out.println("This number is " + network.process(input));
    }
}