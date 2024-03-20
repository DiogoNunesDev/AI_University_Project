import utils.*;


public class Main {
  
  public static void main(String[] args) {
    GeneticAlgorithm algorithm = new GeneticAlgorithm(100, 100000, 0.1, 0.2, 5, Commons.BREAKOUT_STATE_SIZE, 28 , 2);
    algorithm.evolve(); 
  }

}
