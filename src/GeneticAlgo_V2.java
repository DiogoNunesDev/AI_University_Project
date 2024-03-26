import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import breakout.*;

public class GeneticAlgo_V2 {

  private final int maxPopulation;
  private final int generationMax;
  private final double mutationProbability;
  private final double crossoverProbability;
  private final int tournamentSize;
  private final double elitismRate;
  private final Random random;
  private final int inputDim;
  private final int hiddenDim;
  private final int outputDim;
  private NeuralNetwork nn;
  private Map<double[], Double> fitnessMap;
    
  public GeneticAlgo_V2(int maxPopulation, int generationMax, double mutationProbability, double crossoverProbability, int tournamentSize, double elitismRate, int inputDim, int hiddenDim, int outputDim) {
    this.maxPopulation = maxPopulation;
    this.generationMax = generationMax;
    this.mutationProbability = mutationProbability;
    this.crossoverProbability = crossoverProbability;
    this.tournamentSize = tournamentSize;
    this.elitismRate = elitismRate;
    this.random = new Random();
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    this.outputDim = outputDim;
    this.nn = new NeuralNetwork(inputDim, hiddenDim, outputDim);
    this.fitnessMap = new HashMap<>();

  }

    public double[] evolve() {
      List<double[]> population = generatePopulation();
      double bestFitness = -Double.MAX_VALUE;
      double[] bestIndividual = null;

      for (int generation = 0; generation < generationMax; generation++) {
        // Calculating fitness for each individual if not already evaluated
        for (double[] individual : population) {
          fitnessMap.putIfAbsent(individual, getFitness(individual));
        }
            
        // Sort population based on fitness
        population.sort((individual1, individual2) -> fitnessMap.get(individual2).compareTo(fitnessMap.get(individual1)));

        // Update best individual
        if (fitnessMap.get(population.get(0)) > bestFitness) {
          bestFitness = fitnessMap.get(population.get(0));
          bestIndividual = population.get(0);
        }

        System.out.println("Generation " + generation + ": Best fitness = " + bestFitness);

        List<double[]> newPopulation = new ArrayList<>();

        // Elitism: Keeping the best individuals untouched
        int eliteCount = (int) (maxPopulation * elitismRate);
        newPopulation.addAll(population.subList(0, eliteCount));

        // Generate new individuals until the population is filled with the use of Tournament Selection and Crossover
        while (newPopulation.size() < maxPopulation) {
          double[] parent1 = tournamentSelection(population, fitnessMap);
          double[] parent2 = tournamentSelection(population, fitnessMap);

          // Crossover
          if (random.nextDouble() < crossoverProbability) {
            newPopulation.addAll(Arrays.asList(crossover(parent1, parent2)));
          } else {
            newPopulation.add(parent1.clone());
            newPopulation.add(parent2.clone());
          }
        }

        // Mutation: mutating only the non-elite individuals
        for (int i = eliteCount; i < newPopulation.size(); i++) {
          mutate(newPopulation.get(i), generation);
        }
        // Update population
        population = newPopulation.subList(0, maxPopulation);
        // Reseting fitness map for the new generation
        fitnessMap = new HashMap<>(); 

      }

      return bestIndividual;
    }
  
  private double getFitness(double[] solution){
    nn.setNeuralNetwork(solution);
    BreakoutBoard breakout = new BreakoutBoard(nn, false, 42);
    breakout.runSimulation();
    return breakout.getFitness();
  }
  
  private List<double[]> generatePopulation(){
    List<double[]> population = new ArrayList<>();
    for(int i=0; i < generationMax; i++){
      double[] array = createRandomWeightsAndBiases();;
      population.add(array);
    }
      return population;
  }

  private double[] createRandomWeightsAndBiases() {
    int totalSize = (inputDim * hiddenDim) + hiddenDim + (hiddenDim * outputDim) + outputDim;
    double[] weightsAndBiases = new double[totalSize];
    for (int i = 0; i < totalSize; i++) {
      weightsAndBiases[i] = random.nextDouble() * 2 - 1;
    }
    return weightsAndBiases;
  }
  
  private double[] tournamentSelection(List<double[]> population, Map<double[], Double> fitnessMap) {
    List<double[]> tournament = new ArrayList<>();
    for (int i = 0; i < tournamentSize; i++) {
      int randomIndex = random.nextInt(population.size());
      tournament.add(population.get(randomIndex));
    }
    tournament.sort(Comparator.comparing(fitnessMap::get).reversed());
    return tournament.get(0); 
  }
  
  private double[] crossover(double[] parent1, double[] parent2) {
    double[] offspring = new double[parent1.length];
    int crossoverPoint = random.nextInt(parent1.length);
    for (int i = 0; i < parent1.length; i++) {
      if (i < crossoverPoint) {
        offspring[i] = parent1[i];
      } else {
        offspring[i] = parent2[i];
      }
    }
    return offspring;
  }
  
  private void mutate(double[] individual, int gennerationIndicator) {
    for (int i = 0; i < individual.length; i++) {
      if (random.nextDouble() < mutationProbability) {

        if (gennerationIndicator < generationMax * (4/5)) {
          individual[i] += random.nextDouble() * 2 - 1; 
        } else {
          individual[i] += random.nextGaussian() * 0.1; 
        }
              
        if (individual[i] < -1) {
          individual[i] = -1;
        } else if (individual[i] > 1) {
          individual[i] = 1;
        }
      }
    }
  }
  

    
}
