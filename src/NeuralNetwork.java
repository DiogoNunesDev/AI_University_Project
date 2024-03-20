import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import utils.GameController;

public class NeuralNetwork implements GameController{
    private int inputDim;
    private int hiddenDim;
    private int outputDim;
    private double[][] hiddenWeights;
    private double[] hiddenBiases;
    private double[][] outputWeights;
    private double[] outputBiases;

    /*
     * Breakout Game: 2 output neurons -> [LEFT, RIGHT]
     * - 1st approach hidden layer will be a fully connected layer with shape 4 times as the input layer shape 
     * 
     * 
     * 
     * 
     * 
     */

    @Override
    public int nextMove(int[] currentState) {
        double[] inputValues = new double[currentState.length];
        for (int i = 0; i < currentState.length; i++) {
            inputValues[i] = (double) currentState[i];
        }
        double[] output = forward(inputValues);
        if (output[0] > output[1]) {
            return 1;
        } 
        return 2;
    }


    public NeuralNetwork(int inputDim, int hiddenDim, int outputDim) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        this.hiddenWeights = new double[inputDim][hiddenDim];
        this.hiddenBiases = new double[hiddenDim];
        this.outputWeights = new double[hiddenDim][outputDim];
        this.outputBiases = new double[outputDim];
        initializeParameters();

    }

    public NeuralNetwork(int inputDim, int hiddenDim, int outputDim, double[] values) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        this.hiddenWeights = new double[inputDim][hiddenDim];
        this.hiddenBiases = new double[hiddenDim];
        this.outputWeights = new double[hiddenDim][outputDim];
        this.outputBiases = new double[outputDim];
        int index = 0;

        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                hiddenWeights[i][j] = values[index++];
            }
        }

        for (int i = 0; i < hiddenDim; i++) {
            hiddenBiases[i] = values[index++];
        }

        for (int i = 0; i < hiddenDim; i++) {
            for (int j = 0; j < outputDim; j++) {
                outputWeights[i][j] = values[index++];
            }
        }

        for (int i = 0; i < outputDim; i++) {
            outputBiases[i] = values[index++];
        }
    }

    public void initializeParameters(){
        Random random = new Random();
        for(int i = 0; i < hiddenDim; i++){
            hiddenBiases[i] = random.nextDouble() * 2 - 1;
            for(int j = 0; j < outputDim; j++) {
                outputWeights[i][j] = random.nextDouble() * 2 - 1;
            }

        }
        for(int j = 0; j < outputDim; j++) {
            outputBiases[j] = random.nextDouble() * 2 - 1;
        }
        for(int i = 0; i < inputDim; i++){
            for(int j = 0; j < hiddenDim; j++) {
                this.hiddenWeights[i][j] = random.nextDouble() * 2 - 1;
            }

        }
    }

    public void setNeuralNetwork(double[] values) {
        int index = 0;
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                hiddenWeights[i][j] = values[index++];
            }
        }

        for (int i = 0; i < hiddenDim; i++) {
            hiddenBiases[i] = values[index++];
        }

        for (int i = 0; i < hiddenDim; i++) {
            for (int j = 0; j < outputDim; j++) {
                outputWeights[i][j] = values[index++];
            }
        }

        for (int i = 0; i < outputDim; i++) {
            outputBiases[i] = values[index++];
        }
    }

    private double sigmoid(double x){
        return 1.0 / (1 + Math.exp(-x));
    }

    public double[] forward(double[] inputValues) {
        double[] hiddenLayer = new double[hiddenDim];
        for(int i = 0; i < hiddenDim; i++){
            double value = this.hiddenBiases[i];
            for(int j = 0; j < inputDim; j++){
                value += (this.hiddenWeights[j][i] * inputValues[j]);
            }
            hiddenLayer[i] = sigmoid(value);
        }
        double[] outputLayer = new double[outputDim];
        for(int i = 0; i < outputDim; i++){
            double value = this.outputBiases[i];
            for(int j = 0; j < hiddenDim; j++){
                value += (this.outputWeights[j][i] * hiddenLayer[j]);
            }
            outputLayer[i] = sigmoid(value);
        }

        return outputLayer;
    }

    public double[] getNeuralNetwork() {
        List<Double> nn = new ArrayList<>();
        for(int i=0; i < this.inputDim; i++){
            for (int j = 0; j < this.hiddenDim; j++) {
                nn.add(this.hiddenWeights[i][j]);
            }
            nn.add(this.hiddenBiases[i]);
        }
        for(int i=0; i < this.hiddenDim; i++){
            for (int j = 0; j < outputDim; j++) {
                nn.add(this.outputWeights[i][j]);
            }
            nn.add(this.outputBiases[i]);
        }
        double [] neuralNetwork = new double[nn.size()];
        for(int i = 0; i < nn.size(); i++){ neuralNetwork[i] = nn.get(i);}

        return neuralNetwork;
    }

    @Override
    public String toString() {
        String result = "Neural Network: \nNumber of inputs: "
                + inputDim + "\n"
                + "Weights between input and hidden layer with " + hiddenDim + " neurons: \n" ;
        String hidden = "";
        for (int input = 0; input < inputDim; input++) {
            for (int i = 0; i < hiddenDim; i++) {
                hidden += " input" + input + "-hidden" + i + ": "
                        + hiddenWeights[input][i] + "\n";
            }
        }
        result += hidden;
        String biasHidden = "Hidden biases: \n";
        for (int i = 0; i < hiddenDim; i++) {
            biasHidden += " bias hidden" + i + ": " + hiddenBiases[i] + "\n";
        }
        result += biasHidden;
        String output = "Weights between hidden and output layer with "
                + outputDim + " neurons: \n";
        for (int hiddenw = 0; hiddenw < hiddenDim; hiddenw++) {
            for (int i = 0; i < outputDim; i++) {
                output += " hidden" + hiddenw + "-output" + i + ": "
                        + outputWeights[hiddenw][i] + "\n";
            }
        }
        result += output;
        String biasOutput = "Ouput biases: \n";
        for (int i = 0; i < outputDim; i++) {
            biasOutput += " bias ouput" + i + ": " + outputBiases[i] + "\n";
        }
        result += biasOutput;
        return result;
    }

    
}
