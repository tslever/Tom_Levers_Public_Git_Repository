package Com.TSL.UtilitiesForQueuingAndEvaluatingRandomNumbers;


/** *******************************************************************************
* @author EMILIA BUTU
* version 1.0
* since   2020-03
*
* Student name:  Tom Lever
* Completion date: 06/09/21
*
* Implements the application for RandomNumbers_Linked class, using queue.
* Buids a queue of random numbers and displays the content of the queue,
* adding a message for the values divisible by 5
********************************************************************************* */

public class RandomNumbers_LinkedDriver
{
	
  /** -----------------------------------------------------------------------------------------------------------------
   * main is the entry point of this program, which creates a linked list based queue, generates twenty random integers
   * between 1 and 100 inclusive, enqueues those integers, and displays those integers in the order in which they were
   * queued, along with their divisibility by 5.
   * 
   * @param args
   ----------------------------------------------------------------------------------------------------------------- */

  public static void main(String[] args)
  {
		//*** Task #1: define a queue with elements of type Integer using QueueInterface
    QueueInterface<Integer> randomQueue;
    
		//*** Task #2: instantiate the queue as RandomNumbers_Linked object
    randomQueue = new RandomNumbers_Linked<Integer>();
	
   	int number;
   	
		//*** Task #3: fill the queue with 20 integer values randomly generated in a range from 1 to 100
		
    for (int i = 1; i <= 20; i++)
    {
      number = (int)(Math.random()*100+1);
      randomQueue.enqueue(number);
    }
    
		//*** Task #4: displays the content of the queue, 
		//	for numbers divisible by 5, add the message "is divisible by 5"

    System.out.println("\nRandom numbers are:\n");
    while (!randomQueue.isEmpty())
    {
      	number  = randomQueue.dequeue();
       	System.out.println(number+((number%5==0)?"\tis divisible by 5":""));
    }
    
  }
  
}