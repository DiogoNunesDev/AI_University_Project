import utils.*;

import java.util.Arrays;
import breakout.*;
import model.*;
import pacman.*;
import utils.Commons.*;

public class Main {

  public static void main(String[] args) {

    FlexibleNeuralNetwork nn = new FlexibleNeuralNetwork();
    nn.addLayer(Commons.BREAKOUT_STATE_SIZE, 128, "relu");
    nn.addLayer(128, 64, "relu");
    nn.addLayer(64, 2, "sigmoid");
    System.out.println(nn.toString());

    int HIDDEN_DIM = 126;
    GeneticAlgo_V2 genAlgo = new GeneticAlgo_V2(nn, 1000, 1000, 0.25, 0.5, 5, 0.02);
    double[] solution = genAlgo.evolve();
    // NeuralNetwork nn = new NeuralNetwork(7, HIDDEN_DIM, 2);
    nn.setParameters(solution);
    //Pacman pacman = new Pacman(nn, true, 42);

    Breakout breakout = new Breakout(nn, 42);

  }

}
