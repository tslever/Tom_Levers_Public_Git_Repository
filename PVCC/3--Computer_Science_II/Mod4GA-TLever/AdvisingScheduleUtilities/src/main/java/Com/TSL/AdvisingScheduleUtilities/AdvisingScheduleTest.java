package Com.TSL.AdvisingScheduleUtilities;


import org.junit.jupiter.api.Test;


/**
 * AdvisingScheduleTest encapsulates JUnit tests of the enqueue and dequeue methods of AdvisingSchedule_Array.
 * @author Tom Lever
 * @version 1.0
 * @since 06/09/21
 */

public class AdvisingScheduleTest {

	
	/**
	 * testEnqueueWhenQueueIsNotFull tests enqueue by adding three names to an advising schedule with a capacity of
	 * more than 3 names.
	 */
	
	@Test
	public void testEnqueueWhenQueueIsNotFull() {
		
		System.out.println("Running testEnqueueWhenQueueIsNotFull.");
		
		AdvisingSchedule_Array<String> theAdvisingSchedule = new AdvisingSchedule_Array<String>();
		
		try {
			for (int i = 0; i < 3; i++) {
				theAdvisingSchedule.enqueue("aTestString");
			}
			System.out.println("Queued 3 strings.");
		}
		
		catch (QueueOverflowException theQueueOverflowException) {
			System.out.println(theQueueOverflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	/**
	 * testEnqueueWhenQueueIsFull tests enqueue by attempting to add 1000 names to an advising schedule with a capacity
	 * significantly less than 1000.
	 */
	
	@Test
	public void testEnqueueWhenQueueIsFull() {
		
		System.out.println("Running testEnqueueWhenQueueIsFull.");
		
		AdvisingSchedule_Array<String> theAdvisingSchedule = new AdvisingSchedule_Array<String>();
		
		try {
			for (int i = 0; i < 1000; i++) {
				theAdvisingSchedule.enqueue("aTestString");
			}
			System.out.println("Queued 1000 strings.");
		}
		
		catch (QueueOverflowException theQueueOverflowException) {
			System.out.println(theQueueOverflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	/**
	 * testEnqueueWhenQueueIsNotFull tests dequeue by removing three names from an advising schedule with three names
	 * in it.
	 */

	@Test
	public void testDequeueWhenQueueIsNotEmpty() {
		
		System.out.println("Running testEnqueueWhenQueueIsNotEmpty.");
		
		AdvisingSchedule_Array<String> theAdvisingSchedule = new AdvisingSchedule_Array<String>();
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
		
		AdvisingSchedule_Array<String> theAdvisingSchedule = new AdvisingSchedule_Array<String>();
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
