package Com.TSL.UtilitiesForQueuingAndEvaluatingRandomNumbers;


import org.junit.jupiter.api.Test;


/**
 * AdvisingScheduleTest encapsulates JUnit tests of the enqueue and dequeue methods of AdvisingSchedule_Array.
 * @author Tom Lever
 * @version 1.0
 * @since 06/09/21
 */

public class LinkedListOfRandomNumbersTest {
	
	
	/**
	 * testEnqueueWhenQueueIsNotFull tests dequeue by removing three names from an advising schedule with three names
	 * in it.
	 */

	@Test
	public void testDequeueWhenQueueIsNotEmpty() {
		
		System.out.println("Running testEnqueueWhenQueueIsNotEmpty.");
		
		RandomNumbers_Linked<String> theAdvisingSchedule = new RandomNumbers_Linked<String>();
		for (int i = 0; i < 3; i++) {
			theAdvisingSchedule.enqueue("aTestString");
		}
		
		try {
			for (int i = 0; i < 3; i++) {
				theAdvisingSchedule.dequeue();
			}
			System.out.println("Dequeued 3 strings.");
		}
		
		catch (QueueUnderflowException theQueueUnderflowException) {
			System.out.println(theQueueUnderflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	/**
	 * testEnqueueWhenQueueIsNotFull tests dequeue by attempting to remove 1000 names from an advising schedule with
	 * significantly fewer elements in it.
	 */
	
	@Test
	public void testDequeueWhenQueueIsEmpty() {
		
		System.out.println("Running testEnqueueWhenQueueIsEmpty.");
		
		RandomNumbers_Linked<String> theAdvisingSchedule = new RandomNumbers_Linked<String>();
		for (int i = 0; i < 3; i++) {
			theAdvisingSchedule.enqueue("aTestString");
		}
		
		try {
			for (int i = 0; i < 1000; i++) {
				theAdvisingSchedule.dequeue();
			}
			System.out.println("Dequeued 1000 strings.");
		}
		
		catch (QueueUnderflowException theQueueUnderflowException) {
			System.out.println(theQueueUnderflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
}