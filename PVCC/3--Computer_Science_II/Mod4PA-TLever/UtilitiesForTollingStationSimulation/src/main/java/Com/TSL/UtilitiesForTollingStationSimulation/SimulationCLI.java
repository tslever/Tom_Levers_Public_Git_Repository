package Com.TSL.UtilitiesForTollingStationSimulation;


//---------------------------------------------------------------------
//SimulationCLI.java       by Dale/Joyce/Weems               Chapter 4
//
// Modified by Tom Lever on 06/11/21.
//
//Simulates customers waiting in queues. Customers always enter
//the shortest queue.
//
// Modified to simulate cars waiting in queues before a tolling station. Customers always enter
// the shortest queue.
//
//Input consists of customer information:
// Minimum and maximum customer interarrival time.
// Minimum and maximum customer service time.
//Followed by a sequence of simulation instance information:
// Number of queues and customers.
//
// Modified such that there is no user input.
// Minimum and maximum car interarrival time is 10 seconds.
// Minimum and maximum car service time is 90 seconds.
// Number of queues ranges from 1 to 10.
// Number of cars is 100.
//
//Output includes, for each simulation instance:
// The average waiting time for a customer.
//
// Modified such that output includes
// A table of number of cashiers / tolling queues and the corresponding
// average per-car tolling durations.
// The optimum number of cashiers for which there is a minimum average per-car tolling duration.
// The corresponding tolling duration.
//----------------------------------------------------------------------
//package ch04.apps;

//import ch04.simulation.Simulation;
//import java.util.Scanner;

public class SimulationCLI 
{
	
	/**
	 * main is the entry point of this program, which displays a table of numbers of tolling cashiers and corresponding
	 * average per-car tolling durations, along with summary statistics.
	 * 
	 * @param args
	 */
	
	public static void main(String[] args)
	{
	
		int[] theNumbersOfCashiers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
		
		System.out.print("Number of cashiers:");
		for (int i = 0; i < theNumbersOfCashiers.length; i++) {
			System.out.print("\t" + theNumbersOfCashiers[i]);
		}
		
		
		System.out.print("\nAverage time:   ");
		
		float[] theAverageTollingDurations = new float[theNumbersOfCashiers.length];
		
		for (int i = 0; i < theNumbersOfCashiers.length; i++) {
			
			int minIAT = 10;
			int maxIAT = 10;
			int minST = 90;
			int maxST = 90;
			int numCustomers = 100;
			
			Simulation sim = new Simulation(minIAT, maxIAT, minST, maxST);
			sim.simulate(theNumbersOfCashiers[i], numCustomers);
			
			theAverageTollingDurations[i] = sim.getAvgWaitTime();
			System.out.print("\t" + sim.getAvgWaitTime());
			
		}
		
		System.out.println("\n");
		
		
		System.out.print("Optimum number of cashiers is: ");
		
		float theMinimumAverageTollingDuration = Float.MAX_VALUE;
		int theIndexOfTheMinimumAverageTollingDuration = -1;
		
		for (int i = 0; i < theNumbersOfCashiers.length; i++) {
			if (theAverageTollingDurations[i] < theMinimumAverageTollingDuration) {
				theMinimumAverageTollingDuration = theAverageTollingDurations[i];
				theIndexOfTheMinimumAverageTollingDuration = i;
			}
		}
		
		System.out.println(theNumbersOfCashiers[theIndexOfTheMinimumAverageTollingDuration]);
		
		
		System.out.println("Average processing time: " + theMinimumAverageTollingDuration);
		
		
		 /*Scanner conIn = new Scanner(System.in);
		
		 int minIAT;    // minimum interarrival time
		 int maxIAT;    // maximum interarrival time
		 int minST;     // minimum service time
		 int maxST;     // maximum service time
		 int numQueues; // number of queues
		 int numCust;   // number of customers
		
		 String skip;           // skip end of line after reading an integer
		 String more = null;    // used to stop or continue processing
		
		 // Get customer information
		 System.out.print("Enter minimum interarrival time: ");
		 minIAT = conIn.nextInt();
		 System.out.print("Enter maximum interarrival time: ");
		 maxIAT = conIn.nextInt();
		 System.out.print("Enter minimum service time: ");
		 minST = conIn.nextInt();
		 System.out.print("Enter maximum service time: ");
		 maxST = conIn.nextInt();
		 System.out.println();      
		
		 // create object to perform simulation
		 Simulation sim = new Simulation(minIAT, maxIAT, minST, maxST);
		
		 do
		 {
		   // Get next simulation instance to be processed.
		   System.out.print("Enter number of queues: ");
		   numQueues = conIn.nextInt();     
		   System.out.print("Enter number of customers: ");
		   numCust = conIn.nextInt();    
		   skip = conIn.nextLine();   // skip end of line
		   
		   // run simulation and output average waiting time
		   sim.simulate(numQueues, numCust);
		   System.out.println("Average waiting time is " + sim.getAvgWaitTime());
		
		   // Determine if there is another simulation instance to process
		   System.out.println(); 
		   System.out.print("Evaluate another simulation instance? (Y=Yes): ");
		   more = conIn.nextLine();
		 }
		 while (more.equalsIgnoreCase("y"));
		
		 System.out.println("Program completed.");*/
		
	}
	
}