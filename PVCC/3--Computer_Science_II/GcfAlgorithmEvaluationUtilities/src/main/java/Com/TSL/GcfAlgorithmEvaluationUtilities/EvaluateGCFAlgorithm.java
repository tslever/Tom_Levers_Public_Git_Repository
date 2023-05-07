package Com.TSL.GcfAlgorithmEvaluationUtilities;


import java.util.Arrays;
import org.junit.jupiter.api.Test;


/** *****************************************************************************************************************
 * AWayOfFindingTheGreatestCommonFactor represents the structure of ways of finding the greatest common factor of two
 * integers.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/04/21
 **************************************************************************************************************** */

interface AWayOfFindingTheGreatestCommonFactor
{
	void findTheGreatestCommonFactor(int theFirstInteger, int theSecondInteger);
}


/** *************************************************************************************************************
 * EvaluateGCFAlgorithm encapsulates utilities for evaluating the consistency and time of various GCF algorithms.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/04/21
 ************************************************************************************************************* */

class EvaluateGCFAlgorithm
{

	private int[][] THE_PAIRS_OF_INTEGERS = {{2, 6}, {200, 1}};
	private int THE_NUMBER_OF_GCF_ALGORITHMS = 3;
	private int THE_NUMBER_OF_ITERATIONS_FOR_EACH_TEST = 100000;
	
	
	/** -------------------------------------------------------------------------------------------------------
	 * theWaysOfFindingTheGreatestCommonFactor is an array of ways of finding the greatest common factor of two
	 * integers.
	 ------------------------------------------------------------------------------------------------------- */
	
	private AWayOfFindingTheGreatestCommonFactor[] theWaysOfFindingTheGreatestCommonFactor =
		new AWayOfFindingTheGreatestCommonFactor[]
	{	
		new AWayOfFindingTheGreatestCommonFactor()
		{
			public void findTheGreatestCommonFactor(int theFirstInteger, int theSecondInteger)
			{
				GCFAlgorithm.gcf1(theFirstInteger, theSecondInteger);
			}
		},
		
		new AWayOfFindingTheGreatestCommonFactor()
		{
			public void findTheGreatestCommonFactor(int theFirstInteger, int theSecondInteger)
			{
				GCFAlgorithm.gcf2(theFirstInteger, theSecondInteger);
			}
		},
		
		new AWayOfFindingTheGreatestCommonFactor()
		{
			public void findTheGreatestCommonFactor(int theFirstInteger, int theSecondInteger)
			{
				GCFAlgorithm.gcf3(theFirstInteger, theSecondInteger);
			}
		}
		
	};
	
	
	/** -----------------------------------------------------------------------------------------------------------
	 * verifyTheResultsOfThreeGcfAlgorithms checks for consistency between three GCF algorithms when evaluating two
	 * pairs of integers.
	 ----------------------------------------------------------------------------------------------------------- */
	
	@Test
	public void verifyTheResultsOfThreeGcfAlgorithms()
	{
		if (
			(GCFAlgorithm.gcf1(2, 6) == GCFAlgorithm.gcf2(2, 6)) &&
			(GCFAlgorithm.gcf1(2, 6) == GCFAlgorithm.gcf3(2, 6)) &&
			(GCFAlgorithm.gcf2(2, 6) == GCFAlgorithm.gcf3(2, 6)) &&
			(GCFAlgorithm.gcf1(200, 1) == GCFAlgorithm.gcf2(200, 1)) &&
			(GCFAlgorithm.gcf1(200, 1) == GCFAlgorithm.gcf3(200, 1)) &&
			(GCFAlgorithm.gcf2(200, 1) == GCFAlgorithm.gcf3(200, 1))
		) {
			System.out.println(
				"Assuming any one of three GCF algorithms is correct, all three are correct for two pairs of " +
				"integers."
			);
		}
		
		else {
			System.out.println("The three GCF algorithms are inconsistent.");
		}
		
	}
	
	
	/** ----------------------------------------------------------------------------------------------------------------
	 * compareTheTimesForThreeGcfAlgorithms creates tables comparing the mean and median times for three GCF algorithms.
	 ---------------------------------------------------------------------------------------------------------------- */
	
	@Test
	public void compareTheTimesForThreeGcfAlgorithms()
	{
		
		long[][][] theExecutionTimes = new long
			[this.THE_PAIRS_OF_INTEGERS.length][THE_NUMBER_OF_GCF_ALGORITHMS][THE_NUMBER_OF_ITERATIONS_FOR_EACH_TEST];
		
		long theStartTime;
		long theEndTime;
		long theDifferenceBetweenTheEndTimeAndTheStartTime;
		
		String theTableOfTheTimesToFindTheMeans =
			"The mean execution times in nanoseconds for evaluating three GCF algorithms over 100,000 iterations with\n" +
			"two different pairs of \na, b\tgcf1\tgcf2\tgcf3\n";
		String theTableOfTheTimesToFindTheMedians =
			"The median execution times in nanoseconds for evaluating three GCF algorithms over 100,000 iterations\n" +
			"with two different pairs of integers\na, b\tgcf1\tgcf2\tgcf3\n";
		
		for (int i = 0; i < this.THE_PAIRS_OF_INTEGERS.length; i++)
		{
			
			theTableOfTheTimesToFindTheMeans +=
				this.THE_PAIRS_OF_INTEGERS[i][0] + ", " + this.THE_PAIRS_OF_INTEGERS[i][1];
			theTableOfTheTimesToFindTheMedians +=
				this.THE_PAIRS_OF_INTEGERS[i][0] + ", " + this.THE_PAIRS_OF_INTEGERS[i][1];
			
			
			
			for (int j = 0; j < THE_NUMBER_OF_GCF_ALGORITHMS; j++)
			{

				for (int k = 0; k < THE_NUMBER_OF_ITERATIONS_FOR_EACH_TEST; k++)
				{
				
					theStartTime = System.nanoTime();
					theWaysOfFindingTheGreatestCommonFactor[j].findTheGreatestCommonFactor(
						this.THE_PAIRS_OF_INTEGERS[i][0], this.THE_PAIRS_OF_INTEGERS[i][1]
					);
					theEndTime = System.nanoTime();
					theDifferenceBetweenTheEndTimeAndTheStartTime = theEndTime - theStartTime;
					
					theExecutionTimes[i][j][k] = theDifferenceBetweenTheEndTimeAndTheStartTime;
					
				}
				
				theTableOfTheTimesToFindTheMeans +=
					"\t" + Arrays.stream(theExecutionTimes[i][j]).average().getAsDouble();
				theTableOfTheTimesToFindTheMedians += "\t" + getTheMedianOf(theExecutionTimes[i][j]); 
			
			}
			
			theTableOfTheTimesToFindTheMeans += "\n";
			theTableOfTheTimesToFindTheMedians += "\n";
		
		}
		
		System.out.println(theTableOfTheTimesToFindTheMeans);
		System.out.println(theTableOfTheTimesToFindTheMedians);		
		
	}
	
	
	/** ------------------------------------------
	 * getTheMedianOf gets the median of an array.
	 * 
	 * @param theExecutionTimes
	 * @return
	 ------------------------------------------ */
	
	private double getTheMedianOf(long[] theExecutionTimes)
	{
		Arrays.sort(theExecutionTimes);
		
		if (theExecutionTimes.length == 0)
		{
			
		}
		
		if (theExecutionTimes.length % 2 == 0)
		{
			return ((double)theExecutionTimes[theExecutionTimes.length / 2] + (double)theExecutionTimes[theExecutionTimes.length / 2 - 1]) / 2;
		}
		
		else
		{
			return (double)theExecutionTimes[theExecutionTimes.length / 2];
		}
		
	}
	
	
}
