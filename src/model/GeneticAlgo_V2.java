package model;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import breakout.*;
import pacman.*;

public class GeneticAlgo_V2 {

  private final int maxPopulation;
  private final int generationMax;
  private double mutationProbability;
  private final double crossoverProbability;
  private final int tournamentSize;
  private final double elitismRate;
  private final Random random;
  private FlexibleNeuralNetwork nn;
  private Map<double[], Double> fitnessMap;

  public GeneticAlgo_V2(FlexibleNeuralNetwork nn, int maxPopulation, int generationMax, double mutationProbability, double crossoverProbability,
      int tournamentSize, double elitismRate) {
    this.maxPopulation = maxPopulation;
    this.generationMax = generationMax;
    this.mutationProbability = mutationProbability;
    this.crossoverProbability = crossoverProbability;
    this.tournamentSize = tournamentSize;
    this.elitismRate = elitismRate;
    this.random = new Random();
    this.nn = nn;
    this.fitnessMap = new HashMap<>();

  }

  public double[] evolve() {
    List<double[]> population = generatePopulation();
    int seed1 = random.nextInt(100) + 1;
    int seed2 = random.nextInt(100) + 1;
    int[] seeds = new int[3];
    seeds[0] = seed1;
    seeds[1] = seed2;

    double bestFitness = -Double.MAX_VALUE;
    double[] bestIndividual = null;
    int count = 1;


    for (int generation = 0; generation < generationMax; generation++) {
      // Calculating fitness for each individual if not already evaluated
      for (double[] individual : population) {
        fitnessMap.putIfAbsent(individual, getFitnessBreakout(individual, seeds));
      }

      // Sort population based on fitness
      population.sort((individual1, individual2) -> fitnessMap.get(individual2).compareTo(fitnessMap.get(individual1)));

      // Update best individual
      if (fitnessMap.get(population.get(0)) > bestFitness) {
        bestFitness = fitnessMap.get(population.get(0));
        bestIndividual = population.get(0);
        count = 0;
        if(count==0 && mutationProbability > 0.4){
          mutationProbability = 0.25;
          System.out.println("Mutation rate decreased to: " + mutationProbability + " at generation " + generation + " due to improvement.");
        }
        writeBestSolutionToFile(bestIndividual, bestFitness);
      }else{
        count++;
      }

      System.out.println("Generation " + generation + ": Best fitness = " + bestFitness);

      /*
       * 
       * Mutation rate will change over time depending on the convergence rate of the algorithm.
       * 
       */

      if(count >= 25 && mutationProbability < 0.8){
        mutationProbability += 0.1;
        System.out.println("Mutation rate increased to: " + mutationProbability + " at generation " + generation + " due to stagnation."); 
        count = 0;
      }

      List<double[]> newPopulation = new ArrayList<>();


      // Elitism: Keeping the best individuals untouched
      int eliteCount = (int) (maxPopulation * elitismRate);
      newPopulation.addAll(population.subList(0, eliteCount));

      // Generate new individuals until the population is filled with the use of
      // Tournament Selection and Crossover
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

  private void writeBestSolutionToFile(double[] bestIndividual, double bestFitness) {
    try {
      FileWriter writer = new FileWriter("src\\model\\best_solution_pacman.txt");
      System.out.println("Printing best solution to file...");
      writer.write("Neural Network Architecture: " + nn.toString() + "\n\n");
      writer.write("Best Fitness: " + bestFitness + "\n");
      writer.write("Parameters: " + Arrays.toString(bestIndividual) + "\n\n");
      writer.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private double getFitnessBreakout(double[] solution, int[] seeds) {
    int randomSeed = random.nextInt(100) + 1;
    seeds[2] = randomSeed;
    nn.setParameters(solution);
    int totalPoints = 0;
    for (int seed : seeds) {
      BreakoutBoard breakout = new BreakoutBoard(nn, false, seed);
      breakout.runSimulation();
      totalPoints += breakout.getFitness();
    }

    return totalPoints / seeds.length;
  }

  private double getFitnessPacman(double[] solution, int[] seeds) {
    int randomSeed = random.nextInt(100) + 1;
    seeds[2] = randomSeed;
    nn.setParameters(solution);
    int totalPoints = 0;
    for (int seed : seeds) {
      PacmanBoard pacman = new PacmanBoard(nn, false, seed);
      pacman.runSimulation();
      totalPoints += pacman.getFitness();
    }

    return totalPoints / seeds.length;
  }

  /*
  private List<double[]> generatePopulation() {
    List<double[]> population = new ArrayList<>();
    for (int i = 0; i < maxPopulation; i++) {
      double[] array = createRandomWeightsAndBiases();
      ;
      population.add(array);
    }
    return population;
  }
  */
  private List<double[]> generatePopulation() {
    List<double[]> population = new ArrayList<>();
    for (int i = 0; i < maxPopulation; i++) {
      double[] array = nn.generateRandomParameters();
      ;
      population.add(array);
    }
    return population;
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

        if (gennerationIndicator < generationMax * (9 / 10)) {
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
