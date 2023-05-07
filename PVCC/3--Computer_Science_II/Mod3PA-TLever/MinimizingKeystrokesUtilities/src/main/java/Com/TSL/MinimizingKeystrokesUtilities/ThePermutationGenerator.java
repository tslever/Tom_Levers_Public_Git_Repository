package Com.TSL.MinimizingKeystrokesUtilities;


import java.util.ArrayList;


class ThePermutationGenerator
{
	
	/** --------------------------------------------------------------------------------------------------------------
	 * permute(int[] thePrimaryArrayOfIndicesForSymbols) provides an array list of array lists of integers, where each
	 * array list of integers represents a permutation on an input array of integers.
	 * 
	 * @param thePrimaryArrayOfIndicesForSymbols
	 * @return
	 -------------------------------------------------------------------------------------------------------------- */
	
	static ArrayList<ArrayList<Integer>> permute(int[] thePrimaryArrayOfIndicesForSymbols)
	{
		ArrayList<ArrayList<Integer>> theArrayListOfThePermutations = new ArrayList<ArrayList<Integer>>();
		
		permute(thePrimaryArrayOfIndicesForSymbols, 0, theArrayListOfThePermutations);
		
		return theArrayListOfThePermutations;
	}

	
	
	/** ----------------------------------------------------------------------------------------------------------
	 * permute(<three parameters>) fills an array list for permutations of an array of integers with permutations.
	 * 
	 * @param theArrayOfIndicesForSymbols
	 * @param index
	 * @param theArrayListOfThePermutations
	 --------------------------------------------------------------------------------------------------------- */
	
	static private void permute(
		int[] theArrayOfIndicesForSymbols, int index, ArrayList<ArrayList<Integer>> theArrayListOfThePermutations
	)
	{
		if (index == theArrayOfIndicesForSymbols.length)
		{
			ArrayList<Integer> theArrayListOfIndicesForSymbols =
				new ArrayList<Integer>(theArrayOfIndicesForSymbols.length);
			
			for (int theIndexForASymbol : theArrayOfIndicesForSymbols)
			{
				theArrayListOfIndicesForSymbols.add(theIndexForASymbol);
			}
			
			theArrayListOfThePermutations.add(theArrayListOfIndicesForSymbols);
			
			return;
		}
		
		for (int i = index; i < theArrayOfIndicesForSymbols.length; i++)
		{
			swap(theArrayOfIndicesForSymbols, i, index);
			
			permute(theArrayOfIndicesForSymbols, index + 1, theArrayListOfThePermutations);
			
			swap(theArrayOfIndicesForSymbols, i, index);
		}
	}

	
	/** -----------------------------------------------------------
	 * swap swaps the integers in array of integers at two indices.
	 * 
	 * @param nums
	 * @param i
	 * @param index
	 ---------------------------------------------------------- */
	
	private static void swap(int[] thePrimaryArrayOfIndicesForSymbols, int theFirstIndex, int theSecondIndex)
	{
		int thePlaceholder = thePrimaryArrayOfIndicesForSymbols[theFirstIndex];
		
		thePrimaryArrayOfIndicesForSymbols[theFirstIndex] = thePrimaryArrayOfIndicesForSymbols[theSecondIndex];
		
		thePrimaryArrayOfIndicesForSymbols[theSecondIndex] = thePlaceholder;
	}
    
}