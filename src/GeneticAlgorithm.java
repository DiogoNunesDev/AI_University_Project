import java.util.*;
import breakout.*;

public class GeneticAlgorithm {

    private int max_population; 
    private int genMax; 
    private double mutationProb; 
    private double selectionProb;
    private int tournment_size;
    private int hiddenDim;
    private int outputDim;
    private int inputDim;
    private Random generator;
    private BreakoutBoard breakout;
    private NeuralNetwork nn;
    
    public GeneticAlgorithm(int max_population, int genMax, double mutationProb, double selectionProb, int tournment_size,int inputDim, int hiddenDim, int outputDim) {
        this.max_population = max_population;
        this.genMax = genMax;
        this.mutationProb = mutationProb;
        this.selectionProb = selectionProb;
        this.tournment_size = tournment_size;
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        this.generator = new Random();
        this.nn = new NeuralNetwork(7, 28, 2);
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
        for(int i=0; i <= max_population; i++){
            double[] array = createRandom_Weights_And_Biases();;
            population.add(array);
        }
        return population;
    }

    private double getFitness(double[] solution){
        nn.setNeuralNetwork(solution);
        breakout.runSimulation();
        return breakout.getFitness();
    }

    private List<double[]> tournment_selection(List<double[]> rankedSolutions){
        List<double[]> bestSolutions = new ArrayList<>();
        bestSolutions.addAll(rankedSolutions.subList(0, 5));

        while(bestSolutions.size() <= max_population*selectionProb){
            
            double[] best = rankedSolutions.get(5 + generator.nextInt(rankedSolutions.size() - 5));
            double best_fitness = getFitness(best);
            for(int i=0; i < tournment_size; i++){
                double[] next = rankedSolutions.get(5 + generator.nextInt(rankedSolutions.size() - 5));
                double next_fitness = getFitness(next);
                if(best_fitness < next_fitness){
                    best = next;
                    best_fitness = next_fitness;
                }
            bestSolutions.add(best);
            
            }

        }
        return bestSolutions;
    }

    private void crossOver(List<double[]> bestSolutions){
        double[] parent_1 = bestSolutions.get(generator.nextInt(bestSolutions.size()));
        double[] parent_2 = bestSolutions.get(generator.nextInt(bestSolutions.size()));

        int crossover_point = generator.nextInt(bestSolutions.get(0).length);

        double[] first_child = new double[bestSolutions.get(0).length];
        double[] second_child = new double[bestSolutions.get(0).length];
        for(int i=0; i < bestSolutions.get(0).length; i++){
            if(i <= crossover_point){
                first_child[i] = parent_1[i];
                second_child[i] = parent_2[i];
            }else{
                first_child[i] = parent_2[i];
                second_child[i] = parent_1[i];
            }
        }
        bestSolutions.add(first_child);
        bestSolutions.add(second_child);
    }

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
    public void evolve(){
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
            List<double[]> bestSolutions = tournment_selection(rankedSolutions);
            
            //Crossover
            while(bestSolutions.size() < max_population){
                crossOver(bestSolutions);
            }

            //Mutation
            mutation(bestSolutions);

            solutions = bestSolutions;
        }
        



    }


}