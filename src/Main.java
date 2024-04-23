import utils.*;

import java.util.Arrays;
import breakout.*;
import model.*;
import pacman.*;
import utils.Commons.*;

public class Main {

  public static void main(String[] args) {

    FlexibleNeuralNetwork pacmanNeuralNetwork = new FlexibleNeuralNetwork();
    pacmanNeuralNetwork.addLayer(Commons.PACMAN_STATE_SIZE, 128, "relu");
    pacmanNeuralNetwork.addLayer(128, 32, "relu");
    pacmanNeuralNetwork.addLayer(32, 4, "sigmoid");
    
    FlexibleNeuralNetwork breakoutNeuralNetwork = new FlexibleNeuralNetwork();
    breakoutNeuralNetwork.addLayer(Commons.BREAKOUT_STATE_SIZE, 32, "relu");
    breakoutNeuralNetwork.addLayer(32, 16, "relu");
    breakoutNeuralNetwork.addLayer(16, 2, "sigmoid");
    
    System.out.println(pacmanNeuralNetwork.toString());
    
    GeneticAlgo_V2 genAlgo = new GeneticAlgo_V2(pacmanNeuralNetwork, 50, 50, 0.25, 0.5, 5, 0.2);
    double[] solution = genAlgo.evolve();
    
    //int HIDDEN_DIM = 126;
    // NeuralNetwork nn = new NeuralNetwork(7, HIDDEN_DIM, 2);
    //double [] solution = {};

    pacmanNeuralNetwork.setParameters(solution);
    Pacman pacman = new Pacman(pacmanNeuralNetwork, true, 42);


    //breakoutNeuralNetwork.setParameters(solution);
    //Breakout breakout = new Breakout(breakoutNeuralNetwork, 42);

  }

}
