package model;

import java.io.*;
import java.util.*;
import pacman.*;
import breakout.*;

public class NetworkInitializer {
  public static void main(String[] args) {
    String filePath = "src\\model\\best_solution_breakout.txt";
    FlexibleNeuralNetwork network = new FlexibleNeuralNetwork();

    if (initializeNetworkFromFile(filePath, network)) {
      System.out.println("Network initialized successfully with layers and parameters from the file.");
      System.out.println(network);
      Breakout breakout = new Breakout(network, 42);
    } else {
      System.out.println("Failed to initialize network from file.");
    }
  }

  private static boolean initializeNetworkFromFile(String filePath, FlexibleNeuralNetwork network) {
    try (Scanner scanner = new Scanner(new File(filePath))) {
      while (scanner.hasNextLine()) {
        String line = scanner.nextLine().trim();
        if (line.startsWith("Input Dimension:")) {
          int inputDim = Integer.parseInt(line.split(":")[1].trim());
          int outputDim = Integer.parseInt(scanner.nextLine().trim().split(":")[1].trim());
          String activationFunc = scanner.nextLine().trim().split(":")[1].trim();
          network.addLayer(inputDim, outputDim, activationFunc);
        } else if (line.startsWith("Parameters:")) {
          System.out.println("GOT IT");
          String paramsData = line.split("Parameters: ")[1];
          System.out.println(paramsData);
          processParameters(paramsData, network);
        }
      }
      return true; // Successfully initialized
    } catch (FileNotFoundException e) {
      System.err.println("File not found: " + e.getMessage());
      e.printStackTrace();
    } catch (IOException e) {
      System.err.println("Error reading from file: " + e.getMessage());
      e.printStackTrace();
    } catch (Exception e) {
      System.err.println("Error initializing network from file: " + e.getMessage());
      e.printStackTrace();
    }
    return false; // Initialization failed
  }

  private static void processParameters(String paramsData, FlexibleNeuralNetwork network) {
    try {
      // Remove any non-numeric characters potentially present before the first '['
      // and after the last ']'
      int startIdx = paramsData.indexOf('[');
      int endIdx = paramsData.lastIndexOf(']') + 1;
      if (startIdx != -1 && endIdx != -1 && endIdx > startIdx) {
        paramsData = paramsData.substring(startIdx, endIdx);
        double[] paramsArray = Arrays.stream(paramsData.replace("[", "").replace("]", "").split(","))
            .mapToDouble(Double::parseDouble)
            .toArray();
        network.setParameters(paramsArray);
      } else {
        System.err.println("Parameters format error: could not find proper parameter boundaries.");
      }
    } catch (NumberFormatException e) {
      System.err.println("Number format error in parameters: " + e.getMessage());
    }
  }
}
