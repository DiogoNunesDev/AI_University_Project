package model;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import utils.GameController;

public class FlexibleNeuralNetwork implements GameController {
  private List<Layer> layers;
  private Random random;

  public FlexibleNeuralNetwork() {
    this.layers = new ArrayList<>();
    this.random = new Random();
  }

  public void addLayer(int inputDim, int outputDim, String activationFunc) {
    if (!layers.isEmpty()) {
      inputDim = layers.get(layers.size() - 1).getOutputDim();
    }
    Layer layer = new Layer(inputDim, outputDim, activationFunc, random);
    layers.add(layer);
  }

  public double[] forward(double[] inputs) {
    double[] activations = inputs;
    for (Layer layer : layers) {
      activations = layer.forward(activations);
    }
    return activations;
  }

  public void setParameters(double[] values) {
    int index = 0;
    for (Layer layer : layers) {
      // Set weights
      for (int i = 0; i < layer.inputDim; i++) {
        for (int j = 0; j < layer.outputDim; j++) {
          if (index < values.length) {
            layer.weights[i][j] = values[index++];
          }
        }
      }
      // Set biases
      for (int i = 0; i < layer.outputDim; i++) {
        if (index < values.length) {
          layer.biases[i] = values[index++];
        }
      }
    }
    if (index != values.length) {
      throw new IllegalArgumentException(
          "The number of values provided does not match the total number of parameters in the network");
    }
  }

  public double[] generateRandomParameters() {
    List<Double> params = new ArrayList<>();
    for (Layer layer : layers) {
      // Randomize weights
      for (int i = 0; i < layer.inputDim; i++) {
        for (int j = 0; j < layer.outputDim; j++) {
          params.add(random.nextDouble() * 2 - 1); // range [-1, 1]
        }
      }
      // Randomize biases
      for (int i = 0; i < layer.outputDim; i++) {
        params.add(random.nextDouble() * 2 - 1); // range[-1, 1]
      }
    }
    return params.stream().mapToDouble(Double::doubleValue).toArray();
  }

  private double[] normalize(int[] data) {
    double min = Double.MAX_VALUE;
    double max = Double.MIN_VALUE;
    double normalizedData[] = new double[data.length];
    // Find min and max values
    for (double value : data) {
      min = Math.min(min, value);
      max = Math.max(max, value);
    }

    // Normalize the data
    for (int i = 0; i < data.length; i++) {
      normalizedData[i] = (double) ((data[i] - min) / (max - min));
    }

    return normalizedData;
  }

  private double[] softmax(double[] inputs) {
    double[] exps = new double[inputs.length];
    double sumExps = 0.0;
    for (int i = 0; i < inputs.length; i++) {
        exps[i] = Math.exp(inputs[i]);
        sumExps += exps[i];
    }
    for (int i = 0; i < inputs.length; i++) {
        exps[i] /= sumExps;
    }
    return exps;
}

  @Override
  public int nextMove(int[] currentState) {
    if (currentState.length <= 7) {
      double[] normalizedData = normalize(currentState);
      double[] output = forward(normalizedData);
      output = softmax(output);
      if (output[0] < output[1]) {
        return 1;
      }
      return 2;
    } else {
      double[] normalizedData = normalize(currentState);
      double[] output = forward(normalizedData);
      output = softmax(output);
      if (output[0] > output[1] && output[0] > output[2] && output[0] > output[3]) {
        return 1;
      } else if (output[1] > output[0] && output[1] > output[2] && output[1] > output[3]) {
        return 2;
      } else if (output[2] > output[0] && output[2] > output[1] && output[2] > output[3]) {
        return 3;
      } else if (output[3] > output[0] && output[3] > output[1] && output[3] > output[2]) {
        return 4;
      } else {
        return 0;
      }
    }
  }

  public String toString() {
    StringBuilder builder = new StringBuilder();
    for (int i = 0; i < layers.size(); i++) {
      builder.append("Layer ").append(i + 1).append(":\n");
      builder.append(layers.get(i).toString()).append("\n");
    }
    return builder.toString();
  }

  private static class Layer {
    private double[][] weights;
    private double[] biases;
    private int inputDim;
    private int outputDim;
    private String activationFunc;

    public Layer(int inputDim, int outputDim, String activationFunc, Random rand) {
      this.inputDim = inputDim;
      this.outputDim = outputDim;
      this.activationFunc = activationFunc;
      this.weights = new double[inputDim][outputDim];
      this.biases = new double[outputDim];
      initializeWeights(rand);
    }

    private void initializeWeights(Random rand) {
      for (int i = 0; i < inputDim; i++) {
        for (int j = 0; j < outputDim; j++) {
          weights[i][j] = rand.nextDouble() * 2 - 1; // Random weights initialization
        }
      }
      for (int i = 0; i < outputDim; i++) {
        biases[i] = rand.nextDouble() * 2 - 1; // Random biases initialization
      }
    }

    public double[] forward(double[] inputs) {
      double[] outputs = new double[outputDim];
      for (int j = 0; j < outputDim; j++) {
        outputs[j] = biases[j];
        for (int i = 0; i < inputDim; i++) {
          outputs[j] += weights[i][j] * inputs[i];
        }
        outputs[j] = applyActivation(outputs[j]);
      }
      return outputs;
    }

    private double applyActivation(double x) {
      switch (activationFunc) {
        case "sigmoid":
          return 1.0 / (1 + Math.exp(-x));
        case "relu":
          return Math.max(0, x);
        default:
          return x; // linear activation by default
      }
    }

    public int getOutputDim() {
      return outputDim;
    }

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      builder.append("Input Dimension: ").append(inputDim).append("\n");
      builder.append("Output Dimension: ").append(outputDim).append("\n");
      builder.append("Activation Function: ").append(activationFunc).append("\n");
      return builder.toString();
    }
  }

}
