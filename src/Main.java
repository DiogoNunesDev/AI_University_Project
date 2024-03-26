import utils.*;

import java.util.Arrays;

import breakout.*;

public class Main {
  
  public static void main(String[] args) {
    
    //GeneticAlgorithm algorithm = new GeneticAlgorithm(5000, 10000, 0.01, 0.3, 5, 0.05,  Commons.BREAKOUT_STATE_SIZE, 512, 2);
    //double[] solution = algorithm.evolve();
    
    GeneticAlgo_V2 genAlgo = new GeneticAlgo_V2(500, 100, 0.1, 0.3, 5, 0.05,  Commons.BREAKOUT_STATE_SIZE, 252, 2);
    double[] solution = genAlgo.evolve();
    NeuralNetwork nn = new NeuralNetwork(7, 252, 2);
    nn.setNeuralNetwork(solution);
    BreakoutBoard bd = new BreakoutBoard(nn, false, 42);
    bd.runSimulation();
    System.out.println(bd.getFitness());
    Breakout breakout = new Breakout(nn, 42);
    System.out.println(Arrays.toString(solution));

  }

}
