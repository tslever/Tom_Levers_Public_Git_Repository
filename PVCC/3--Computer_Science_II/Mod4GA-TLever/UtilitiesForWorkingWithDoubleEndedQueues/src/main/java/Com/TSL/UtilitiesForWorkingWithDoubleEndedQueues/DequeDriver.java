package Com.TSL.UtilitiesForWorkingWithDoubleEndedQueues;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-03
*
* Student name:  Tom Lever
* Completion date: 06/10/21
*
* DequeueDriver.java
*
* Implements the application for Dequeue class.
*/

public class DequeDriver
{
	
	/**
	 * main is the entry point of this program, which creates a double-ended queue, enqueues random integers between 1
	 * and 100 at the front and rear, displays the double-ended queue, dequeues and displays, displays information about
	 * the double-ended queue, and clears and displays the double-ended queue.
	 * 
	 * @param args
	 */
	
	public static void main(String[] args)
	{
		
		//*** Task #1: object of type Dequeue and instantiate it with integers
		Deque dq = new Deque<Integer>();
		
		int value;

		// work with random numbers between 1 to 100
		// perform a number of dequeue operations
		//*** Task #2: perform insert at front followed by insert at rear three times

		for(int i=0;i<3; i++)
		{
			value=(int)(Math.random()*100+1);	
			dq.enqueueFront(value);
			System.out.println("Enqueued at front " + value + ". Deque is ");
			dq.display();
			
			value=(int)(Math.random()*100+1);
			dq.enqueueRear(value);
			System.out.println("Enqueued at rear " + value + ". Deque is ");
			dq.display();
			
		}
		System.out.println();
		
		System.out.println("Dequeue after inserting elements at front and at rear three times is: ");
		//*** Task #2: display dequeue
		dq.display();

		//*** Task #3: delete at front
		System.out.println(dq.dequeueFront() + " removed from front of queue");
		
		System.out.println("New queue is: ");
		//*** Task #4: display dequeue
		dq.display();

		//*** Task #5: delete at rear
		System.out.println(dq.dequeueRear() + " removed from rear of queue");

		//*** Task #6: display dequeue
		dq.display();
		
		//*** Task #7: display value at front
		System.out.println("Element at front is: " + dq.peekAtFront());

		//*** Task #8: display value at rear
		System.out.println("Element at rear is: " + dq.peekAtRear());

		//*** Task #9: display size
		System.out.println("Size of dequeue is: " + dq.size());

		//*** Task #10: check if queue is empty
		System.out.println("Dequeue is an empty queue is a " + dq.isEmpty() + " statement!");

		//*** Task #11: clear dequeue and display message to announcing the operation
		dq.clear();
		System.out.println("Dequeue has just been cleared!");
		
		//*** Task #12: display resulting dequeue
		dq.display();
	}

}