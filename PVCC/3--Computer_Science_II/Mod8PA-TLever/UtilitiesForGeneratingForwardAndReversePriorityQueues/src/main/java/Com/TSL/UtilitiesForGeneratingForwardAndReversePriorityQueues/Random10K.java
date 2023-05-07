package Com.TSL.UtilitiesForGeneratingForwardAndReversePriorityQueues;


import org.apache.commons.math3.random.RandomDataGenerator;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-06
*
* Student name:  Tom Lever
* Completion date: 07/10/21
*
*	Random10K.txt: download and save as Random10K.java
*	
*	Uses the SortedABPriQ, building priority queues in increasing and decreasing order
*
* 	Student tasks: complete tasks specified in the file
*/

public class Random10K
{
	
  /**
   * main represents the entry point of this program, which creates a priority queue of random integers sorted in
   * ascending order, displays information about this priority queue, creates a priority queue of random integers sorted
   * in descending order, and displays information about this priority queue.
   * 
   * @param args
   */
	
  public static void main(String[] args)
  {
		//*** Task #1: declare a variable pq of PriQueueInterface type, and instantiate it as
		//		SortedABPriQ, in which you insert the elements in increasing order
      PriQueueInterface<Integer> pq = new SortedABPriQ<Integer>();
	  
		//*** Task #2: declare a variable rpq of PriQueueInterface type, and instantiate it as
		//		SortedABPriQ, in which you insert the elements in decreasing order
      PriQueueInterface<Integer> rpq = new SortedABPriQ<Integer>();
      
	Integer n=0;
	RandomDataGenerator theRandomDataGenerator = new RandomDataGenerator();
    for (int i=0;i<100;i++)
    {
		//*** Task #3: assign a random integer between 1 and 10000 to n
    	n = theRandomDataGenerator.nextInt(1, 10000);

		//*** Task #4: add the element to pq using the regular enqueue
    	pq.enqueue(n);
    	
		//*** Task #5: add the element to rpq using the renqueue method
    	rpq.renqueue(n);
	}
    
		//*** Task #6: display the priority queue with elements in increasing order
	System.out.println(pq);
	
		//*** Task #7: display size of priority queue pq
	System.out.print("Priority queue has " + pq.size() + " elements.\n\n");
	
	
	//*** Task #8: display the priority queue with elements in decreasing order
	System.out.println("Priority Queue in Reverse Order:");
	System.out.println(rpq);
	
	//*** Task #9: display size of priority queue pq
	System.out.println("The reverse priority queue has " + rpq.size() + " elements.");

 }
}