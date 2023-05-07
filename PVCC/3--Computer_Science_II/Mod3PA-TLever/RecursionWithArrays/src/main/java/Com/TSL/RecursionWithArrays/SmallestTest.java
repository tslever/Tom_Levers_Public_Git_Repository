package Com.TSL.RecursionWithArrays;


import java.util.Arrays;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.junit.jupiter.api.Test;


/** **************************************************************************************************************
 * SmallestTest encapsulates JUnit tests of core functionality of the method main of class RecursiveMethodsArrays.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 ************************************************************************************************************* */

public class SmallestTest {

	
	/** -----------------------------------------------------------------------------------------------------------------
	 * testSmallestWithOneParameter tests the method smallest with one parameter by continuously creating one-dimensional
	 * arrays of various sizes of various integers and demonstrating that smallest finds the same minimum integer as
	 * a min method in Java's Arrays package for each array. testSmallestWithOneParameter stops if a different minimum
	 * is found.
	 ----------------------------------------------------------------------------------------------------------------- */
	
	@Test
	public void testSmallestWithOneParameter ()
	{
		final int THE_LOWEST_ARRAY_SIZE = 1;
		final int THE_HIGHEST_ARRAY_SIZE = 100;		
		
		int[] theArrayForWhichToFindAMinimum;
		RandomDataGenerator theRandomDataGenerator = new RandomDataGenerator();
		
		while (true)
		{
			theArrayForWhichToFindAMinimum = new int [
			    theRandomDataGenerator.nextInt (THE_LOWEST_ARRAY_SIZE, THE_HIGHEST_ARRAY_SIZE)
			];
			
			
			for (int i = 0; i < theArrayForWhichToFindAMinimum.length; i++)
			{
				theArrayForWhichToFindAMinimum [i] =
					theRandomDataGenerator.nextInt (Integer.MIN_VALUE, Integer.MAX_VALUE);
			}
			
			
			if (Arrays.stream (theArrayForWhichToFindAMinimum).min ().getAsInt () ==
				RecursiveMethodsArrays.smallest (theArrayForWhichToFindAMinimum))
			{
				System.out.println (
					Arrays.stream (theArrayForWhichToFindAMinimum).min ().getAsInt () + " equals " +
					RecursiveMethodsArrays.smallest (theArrayForWhichToFindAMinimum) + "."
				);
			}
			else
			{
				System.out.println (
					Arrays.stream (theArrayForWhichToFindAMinimum).min ().getAsInt () + " does not equal " +
					RecursiveMethodsArrays.smallest (theArrayForWhichToFindAMinimum) + "."
				);
				break;
			}
						
		}
		
	}
	
	
	/** ---------------------------------------------------------------------------------------------------------------
	 * testSmallestWithTwoParameters tests the method smallest with two parameters by continuously creating
	 * two-dimensional arrays of various sizes of various integers and demonstrating that smallest finds the same
	 * minimum integer as a custom min method for each array. testSmallestWithOneParameter stops if a different minimum
	 * is found.
	 --------------------------------------------------------------------------------------------------------------- */
	
	@Test
	public void testSmallestWithTwoParameters ()
	{
		final int THE_LOWEST_ARRAY_SIZE = 1;
		final int THE_HIGHEST_ARRAY_SIZE = 100;		
		
		int[][] theArrayForWhichToFindAMinimum;
		RandomDataGenerator theRandomDataGenerator = new RandomDataGenerator();
		
		while (true)
		{
			theArrayForWhichToFindAMinimum = new int
			    [theRandomDataGenerator.nextInt (THE_LOWEST_ARRAY_SIZE, THE_HIGHEST_ARRAY_SIZE)]
			    [theRandomDataGenerator.nextInt (THE_LOWEST_ARRAY_SIZE, THE_HIGHEST_ARRAY_SIZE)]
			;
			
			
			for (int i = 0; i < theArrayForWhichToFindAMinimum.length; i++)
			{
				for (int j = 0; j < theArrayForWhichToFindAMinimum[0].length; j++)
				{
					theArrayForWhichToFindAMinimum [i][j] =
						theRandomDataGenerator.nextInt (Integer.MIN_VALUE, Integer.MAX_VALUE);
				}
			}
			
			
			if (getTheMinimumIntegerIn(theArrayForWhichToFindAMinimum) ==
				RecursiveMethodsArrays.smallest (theArrayForWhichToFindAMinimum))
			{
				System.out.println (
					getTheMinimumIntegerIn(theArrayForWhichToFindAMinimum) + " equals " +
					RecursiveMethodsArrays.smallest (theArrayForWhichToFindAMinimum) + "."
				);
			}
			else
			{
				System.out.println (
					getTheMinimumIntegerIn(theArrayForWhichToFindAMinimum) + " does not equal " +
					RecursiveMethodsArrays.smallest (theArrayForWhichToFindAMinimum) + "."
				);
				break;
			}
						
		}
		
	}
	
	
	/** -----------------------------------------------------------
	 * getTheMinimumIntegerIn gets the minimum integer in an array.
	 * 
	 * @param theArray
	 * @return
	 ---------------------------------------------------------- */
	
	private int getTheMinimumIntegerIn(int[][] theArray)
	{
		int theMinimumInteger = Integer.MAX_VALUE;
		for (int i = 0; i < theArray.length; i++) {
			for (int j = 0; j < theArray[0].length; j++) {
				if (theArray[i][j] < theMinimumInteger) {
					theMinimumInteger = theArray[i][j];
				}
			}
		}
		
		return theMinimumInteger;
	}
	
	
}
