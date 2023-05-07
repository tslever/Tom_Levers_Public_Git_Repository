package Com.TSL.AdvisingScheduleUtilities;


import java.util.Scanner;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-03
*
* Student name: Tom Lever
* Completion date: 06/09/21
*
* Implements the application for AdvisingSchedule_Array class, using queue.
* Outputs strings in same order of entry.
*/

public class AdvisingSchedule_ArrayDriver
{
	
	/**
	 * main is the entry point of this program, which repeatedly prompts a user to enter a student name until "done" is
	 * entered. The program proceeds to enqueue each name in a queue of names. If a name has been entered but the queue
	 * is full, the program overwrites the name with "done" and outputs a warning. The program prints information about
	 * the number of names in the queue and displays the names in the queue in the order in which they were entered into
	 * the queue.
	 * 
	 * @param args
	 */
	
	public static void main(String[] args)
	{
		Scanner scan = new Scanner(System.in);
		
    		//*** Task #1: defime a queue with elements of type String using QueueInterface
		QueueInterface<String> nameQueue;
		
    		//*** Task #2: instantiate the queue using the constructor that provides the size, using a smaller number for test
		nameQueue = new AdvisingSchedule_Array<String>(5);

		String name="";
		
    		//*** Task #3: enter the names of students that ask for advising.
    		//* allow the user to terminate the entry
    		//* terminate the loop when the queue is full, and display appropriate message to signal it

		while(!name.equalsIgnoreCase("done"))
		{
			System.out.print("Enter student name. If you want to finish enter \"done\": ");
			name = scan.nextLine();
			if(!name.equalsIgnoreCase("done"))
				nameQueue.enqueue(name);

			// test if the queue is full
			if(nameQueue.isFull())
			  {
				  name="done";
				  System.out.println("There is no more spot available for advising!");
			  }
		}

    		//*** Task #4: identify the number of students in the advising queue and display it

		int noStudents = nameQueue.size();
		System.out.println("Number of students asking for advising: " + noStudents);

    		//*** Task #5: display the list of students presents in the advising queue


		System.out.println("\nScheduled students:\n");
		while (!nameQueue.isEmpty())
		{
		  name = nameQueue.dequeue();
		  System.out.println(name);
		}
		
	}
	
}