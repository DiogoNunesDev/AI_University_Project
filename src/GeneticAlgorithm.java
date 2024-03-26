import java.util.*;
import breakout.*;

public class GeneticAlgorithm {

    private final int max_population; 
    private final int genMax; 
    private final double mutationProb; 
    private final double selectionProb;
    private final int tournment_size;
    private final double elitism_percentage;
    private final int hiddenDim;
    private final int outputDim;
    private final int inputDim;
    private Random generator;
    private BreakoutBoard breakout;
    private NeuralNetwork nn;
    
    public GeneticAlgorithm(int max_population, int genMax, double mutationProb, double selectionProb, int tournment_size, double elitism_percentage, int inputDim, int hiddenDim, int outputDim) {
        this.max_population = max_population;
        this.genMax = genMax;
        this.mutationProb = mutationProb;
        this.selectionProb = selectionProb;
        this.tournment_size = tournment_size;
        this.elitism_percentage = elitism_percentage;
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        this.generator = new Random();
        this.nn = new NeuralNetwork(inputDim, hiddenDim, outputDim);
        this.breakout = new BreakoutBoard(nn, false, 42);
        
    }

    public double[] createRandom_Weights_And_Biases(){
        double[] array = new double[hiddenDim*inputDim + hiddenDim + outputDim*hiddenDim + outputDim];
        for(int i=0; i < array.length; i++){
            array[i] = generator.nextDouble() * 2 - 1;
        }
        return array;
    }

    private List<double[]> generate_population(){
        List<double[]> population = new ArrayList<>();
        for(int i=0; i < max_population; i++){
            double[] array = createRandom_Weights_And_Biases();;
            population.add(array);
        }
        return population;
    }

    private double getFitness(double[] solution){
        nn.setNeuralNetwork(solution);
        breakout.setController(nn);
        breakout.runSimulation();
        return breakout.getFitness();
    }

    private List<double[]> tournment_selection(List<double[]> population, double elitism_percentage){
        List<double[]> selected = new ArrayList<>();
        int elitism_count = (int) (max_population * elitism_percentage);
    
        // Adds elite members directly to the next generation
        for (int i = 0; i < elitism_count; i++) {
            selected.add(population.get(i));
        }
    
        // Running tournaments for the rest of the selections
        while (selected.size() < max_population * selectionProb) {
            List<double[]> tournament = new ArrayList<>();
            for (int i = 0; i < tournment_size; i++) {
                int randomIndex = generator.nextInt(population.size());
                tournament.add(population.get(randomIndex));
            }
            tournament.sort((array1, array2) -> Double.compare(getFitness(array2), getFitness(array1)));
            selected.add(tournament.get(0)); // Add the winner of the tournament
        }
    
        return selected;
    }
    

    /*
     * 
     * Using a simple one point crossover to create new solutions
     * 
     */
    private void crossOver(List<double[]> bestSolutions) {
    
        double[] parent_1 = bestSolutions.get(generator.nextInt(bestSolutions.size()));
        double[] parent_2 = bestSolutions.get(generator.nextInt(bestSolutions.size()));
        
        double[] first_child = new double[parent_1.length];
        double[] second_child = new double[parent_2.length];

        int crossOver_point = generator.nextInt(parent_1.length);
        for(int i=0; i < crossOver_point; i++){
            first_child[i] = parent_1[i];
            second_child[i] = parent_2[i];
        }
        for(int i=crossOver_point; i < parent_1.length; i++){
            first_child[i] = parent_2[i];
            second_child[i] = parent_1[i];
        }
    
        bestSolutions.add(first_child);
        bestSolutions.add(second_child);
    }


    /*
     * Using Gaussian mutation to mutate the solutions, this allows for fine tuning the model
     * This can help the neural network learn the precision to hit a single brick
     */
    private void gaussian_mutation(List<double[]> bestSolutions, double std){
        for(double[] s : bestSolutions){
            for(int i=0; i < s.length; i++){
                if(generator.nextDouble() <= mutationProb){
                    s[i] = generator.nextGaussian() * std;
                }
            }
        }
    }

    /*
     * Using simple mutation to mutate the solutions, this allows for more diversity in the initial stages of the algorithm
     * 
     */
    private void mutation(List<double[]> bestSolutions){
        for(double[] s : bestSolutions){
            for(int i=0; i < s.length; i++){
                if(generator.nextDouble() <= mutationProb){
                    s[i] = generator.nextDouble() * 2 - 1;
                }
            }
        }
    }
    //Generate solutions
    public double[] evolve(){
        List<double[]> solutions = generate_population();

        for(int genIndicator = 0; genIndicator <= genMax; genIndicator++){
            List<double[]> rankedSolutions = new ArrayList<>();
            for (int i = 0; i < solutions.size(); i++) {
                rankedSolutions.add(solutions.get(i));
            }

            //Sorting by descending order
            rankedSolutions.sort((array1, array2) -> Double.compare(getFitness(array2), getFitness(array1)));
            System.out.println("=== Gen " + genIndicator + " best solution === ");
            System.out.println("Solution Score: " + getFitness(rankedSolutions.get(0)));

            if(getFitness(rankedSolutions.get(0))==1600000.0){
                System.out.println("=== Best solution === " + rankedSolutions.get(0));
                break;
            }

            //Top 50 solutions by Tournament Selection
            List<double[]> bestSolutions = tournment_selection(rankedSolutions, this.elitism_percentage);
            
            //Crossover
            while(bestSolutions.size() < max_population){
                crossOver(bestSolutions);
            }

            //Mutation
            if (genIndicator < genMax * (4/5)) {
                mutation(bestSolutions);
            } else {
                gaussian_mutation(bestSolutions, 0.05);
            }

            solutions = bestSolutions;
        }

        return solutions.get(0);
        
    }


}