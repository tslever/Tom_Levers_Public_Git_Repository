package Com.TSL.MatrixEditor;


import org.junit.jupiter.api.Test;


/** *****************************************************************************************************************
 * MainTest encapsulates JUnit tests of method main that test for an invalid number of input arguments, a non-integer
 * coordinate, and a coordinate being too high.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/06/21
 **************************************************************************************************************** */

public class MainTest {

	
	/** --------------------------------------------------------------------------------------------
	 * testMainForInvalidNumberOfInputArguments tests main for an invalid number of input arguments.
	 -------------------------------------------------------------------------------------------- */
	
	@Test
	public void testMainForInvalidNumberOfInputArguments()
	{
		
		System.out.println("Running testMainForInvalidNumberOfInputArguments.");
		
		String[] args = "pathToFile 1 2 3".split(" ");
		
		if (args.length != 3)
		{
			System.out.println("Exception: The number of input arguments is not equal to 3.");
		}
		else
		{
			System.out.println("The number of input arguments is equal to 3.");
		}
		
		System.out.println();
		
	}
	

	/** -----------------------------------------------------------------------
	 * testMainForNonIntegerCoordinate tests main for a non-integer coordinate.
	 ----------------------------------------------------------------------- */
	
	@Test
	public void testMainForNonIntegerCoordinate()
	{
		
		System.out.println("Running testMainForNonIntegerCoordinate.");
		
		String[] args = "pathToFile s 2".split(" ");
		
		
		try {
			Integer.parseInt(args[1]);
			System.out.println("Parsed coordinate " + args[1] + ".");
		}
		catch (NumberFormatException theNumberFormatException) {
			System.out.println(theNumberFormatException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	/** -----------------------------------------------------------------------
	 * testMainForCoordinateTooHigh tests main for a coordinate being too high.
	 ----------------------------------------------------------------------- */
	
	@Test
	public void testMainForCoordinateTooHigh()
	{
		
		System.out.println("Running testMainForCoordinateTooHigh.");
		
		String[] args = "pathToFile 100 200".split(" ");
		
		
		int theRowIndex = Integer.parseInt(args[1]);
		int theColumnIndex = Integer.parseInt(args[2]);
		
		if (theRowIndex >= 9 || theColumnIndex >= 12)
		{
			System.out.println("Exception: A coordinate is too high.");
		}
		else
		{
			System.out.println("Stored coordinates.");
		}
		
		System.out.println();
		
	}
	
	
}
