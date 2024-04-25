import utils.*;

import java.util.Arrays;
import breakout.*;
import model.*;
import pacman.*;
import utils.Commons.*;

public class Main {

  public static void main(String[] args) {

    FlexibleNeuralNetwork pacmanNeuralNetwork = new FlexibleNeuralNetwork();
    pacmanNeuralNetwork.addLayer(Commons.PACMAN_STATE_SIZE, 256, "relu");
    pacmanNeuralNetwork.addLayer(256, 128, "relu");
    pacmanNeuralNetwork.addLayer(128, 64, "relu");
    pacmanNeuralNetwork.addLayer(64, 4, "linear");
    
    FlexibleNeuralNetwork breakoutNeuralNetwork = new FlexibleNeuralNetwork();
    breakoutNeuralNetwork.addLayer(Commons.BREAKOUT_STATE_SIZE, 128, "relu");
    breakoutNeuralNetwork.addLayer(128, 2, "linear");
    
    System.out.println(breakoutNeuralNetwork.toString());
    
    GeneticAlgo_V2 genAlgo = new GeneticAlgo_V2(breakoutNeuralNetwork, 2500, 1000, 0.25, 1, 10, 0.05);
    double[] solution = genAlgo.evolve();
    
    //int HIDDEN_DIM = 126;
    // NeuralNetwork nn = new NeuralNetwork(7, HIDDEN_DIM, 2);
    //double [] solution = {};

    //pacmanNeuralNetwork.setParameters(solution);
    //Pacman pacman = new Pacman(pacmanNeuralNetwork, true, 42);


    breakoutNeuralNetwork.setParameters(solution);
    Breakout breakout = new Breakout(breakoutNeuralNetwork, 42);

  }

}
