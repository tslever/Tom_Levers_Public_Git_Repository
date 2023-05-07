package Com.TSL.MinimizingKeystrokesUtilities;


import cc.redberry.combinatorics.CombinatorialIterator;
import cc.redberry.combinatorics.Combinatorics;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;


/** ********************************************************************************************************************
 * KeystrokeMinimizer encapsulates the entry point of this program, which displays a minimum number of key strokes among
 * structured permutations of indices for symbols. A structured permutation is associated with a total number of symbols
 * for a keypad, a total number of keys for the keypad, and limits for each key on the number of symbols for that key.
 * A number of key strokes for a structured permutation represents the number of key strokes on a keypad expected after
 * a large number of people collectively generate the equivalent of a research text corpus by generating symbols based
 * on repeated key strokes to a key. This program also displays all structured permutations corresponding to the minimum
 * number of key strokes.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/06/21
 ******************************************************************************************************************** */

class KeystrokeMinimizer 
{	
	
	private static HashMap<ArrayList<ArrayList<Integer>>, Integer>
		theHashMapOfStructuredPermutationsOfIndicesAndNumberOfKeyStrokes =
		new HashMap<ArrayList<ArrayList<Integer>>, Integer>();
	
	
	/** ---------------------------------------------------------------------------------------------------------------
	 * main is the entry point of this program, which displays a minimum number of key strokes among structured
	 * permutations of indices for symbols. A structured permutation is associated with a total number of symbols for a
	 * keypad, a total number of keys for the keypad, and limits for each key on the number of symbols for that key.
	 * A number of key strokes for a structured permutation represents the number of key strokes on a keypad expected
	 * after a large number of people collectively generate the equivalent of a research text corpus by generating
	 * symbols based on repeated key strokes to a key. This program also displays all structured permutations
	 * corresponding to the minimum number of key strokes.
	 * 
	 * @param args
	 ---------------------------------------------------------------------------------------------------------------- */

    public static void main (String[] args) throws AFrequenciesAndLettersDoNotMatchException
    {
    	
    	int[] fre = {5, 2, 20, 9, 90};
    	int[] keys = {2, 3};
    	
    	int res = minimalKeyStrokes(fre, keys);
    	System.out.println("The minimum number of key strokes: " + res);
    	
    	char[] letters = {'A', 'B', 'C', 'D', 'E'};
    	
    	if (letters.length != fre.length)
    	{
    		throw new AFrequenciesAndLettersDoNotMatchException(
    			"Exception: The arrays of frequencies does not have the same length as the array of letters.");
    	}
    	
    	String arrangement = keyPadArrangement(fre, keys, letters);
    	System.out.println(
            "The structured permutations of letters corresponding to the minimum number of key strokes:\n" +
            arrangement
        );
        
    }
    
    
    /** ----------------------------------------------------------------------------------------------------------
     * minimalKeyStrokes determines the minimum number of key strokes among structured permutations of indices for
     * symbols.
     * 
     * @param frequencies
     * @param keyLimits
     * @return
     --------------------------------------------------------------------------------------------------------- */
    
    public static int minimalKeyStrokes(int[] frequencies, int[] keyLimits)
    {
    	ArrayList<int[]> theArrayListOfCompositionsOfTheNumberOfSymbols = new ArrayList<int[]>();
    	
        CombinatorialIterator<int[]> theCombinatorialIterator =
        	Combinatorics.compositions(frequencies.length, keyLimits.length);
        
        while (theCombinatorialIterator.hasNext()) {
        	
        	int[] thePresentCompositionOfTheNumberOfSymbols = theCombinatorialIterator.next();
        	
        	boolean allPropositionsOfANumberOfSymbolsForAKeyInThePresentCompositionAreLessThanOrEqualToTheKeyLimit =
        			true;
        	
        	for (int i = 0; i < thePresentCompositionOfTheNumberOfSymbols.length; i++) 
        	{
        		if (thePresentCompositionOfTheNumberOfSymbols[i] > keyLimits[i])
        		{
        			allPropositionsOfANumberOfSymbolsForAKeyInThePresentCompositionAreLessThanOrEqualToTheKeyLimit =
        				false;
        		}
        	}
        	
        	if (allPropositionsOfANumberOfSymbolsForAKeyInThePresentCompositionAreLessThanOrEqualToTheKeyLimit)
        	{
        		theArrayListOfCompositionsOfTheNumberOfSymbols.add(thePresentCompositionOfTheNumberOfSymbols);
        	}
        	
        }
        
        int[] thePrimaryArrayOfIndicesForSymbols = getThePrimaryArrayOfIndicesForSymbolsBasedOn(frequencies);
        
        ArrayList<ArrayList<Integer>> theArrayListOfAllPermutationsOfAnArrayOfIndicesForSymbols =
        	ThePermutationGenerator.permute(thePrimaryArrayOfIndicesForSymbols);
        
        for (int[] theCompositionOfTheNumberOfSymbols : theArrayListOfCompositionsOfTheNumberOfSymbols)
        {
    		for (ArrayList<Integer> thePermutationOfThePrimaryArrayOfIndicesForSymbols :
    			 theArrayListOfAllPermutationsOfAnArrayOfIndicesForSymbols)
    		{	
    			ArrayList<ArrayList<Integer>> theStructuredPermutationOfIndicesForSymbols =
    				new ArrayList<ArrayList<Integer>>();
    			
    			int theNumberOfKeyStrokes = 0;
    			int theNumberOfSymbolsForAllPreviousKeys = 0;
    			
            	for (int i = 0; i < theCompositionOfTheNumberOfSymbols.length; i++)
            	{
            		ArrayList<Integer> theIndicesForTheNumberOfSymbols = new ArrayList<Integer>();
            		
            		for (int j = 0; j < theCompositionOfTheNumberOfSymbols[i]; j++)
            		{
            			theNumberOfKeyStrokes += (j + 1) *
            				frequencies[
            				    thePermutationOfThePrimaryArrayOfIndicesForSymbols.get(
            				    	theNumberOfSymbolsForAllPreviousKeys + j
            				    )
            				];
            			
            			theIndicesForTheNumberOfSymbols.add(
            				thePermutationOfThePrimaryArrayOfIndicesForSymbols.get(
            					theNumberOfSymbolsForAllPreviousKeys + j
            				)
            			);
            		}
            		
            		theNumberOfSymbolsForAllPreviousKeys += theCompositionOfTheNumberOfSymbols[i];
            		
            		theStructuredPermutationOfIndicesForSymbols.add(theIndicesForTheNumberOfSymbols);
            	}
            	
            	theHashMapOfStructuredPermutationsOfIndicesAndNumberOfKeyStrokes.put(
            		theStructuredPermutationOfIndicesForSymbols, theNumberOfKeyStrokes);
    		}
    		
        }
        
        return findTheMinimumNumberOfKeyStrokes();
    }
    
    
    /** ------------------------------------------------------------------------------------------------------------
     * findTheMinimumNumberOfKeyStrokes finds the minimum number of keystrokes in the sets of values of a dictionary
     * of structured permutations of indices and corresponding numbers of key strokes.
     * 
     * @return
     ----------------------------------------------------------------------------------------------------------- */
    
    private static int findTheMinimumNumberOfKeyStrokes()
    {
        int theMinimumNumberOfKeyStrokes = Integer.MAX_VALUE;
        
        for (Map.Entry<ArrayList<ArrayList<Integer>>, Integer> theKeyValuePair : theHashMapOfStructuredPermutationsOfIndicesAndNumberOfKeyStrokes.entrySet())
        {
        	if (theKeyValuePair.getValue() < theMinimumNumberOfKeyStrokes)
        	{
        		theMinimumNumberOfKeyStrokes = theKeyValuePair.getValue();
        	}
        }
        
        return theMinimumNumberOfKeyStrokes;
    }
    
    
    /** --------------------------------------------------------------------------------------------------------------
     * getThePrimaryArrayOfIndicesForSymbolsBasedOn gets the primary array of indices for symbols based on an array of
     * frequencies for symbols.
     * 
     * @param frequencies
     * @return
     ------------------------------------------------------------------------------------------------------------- */
    
    private static int[] getThePrimaryArrayOfIndicesForSymbolsBasedOn(int[] frequencies)
    {
        int[] thePrimaryArrayOfIndicesForSymbols = new int[frequencies.length];
        
        for (int i = 0; i < frequencies.length; i++)
        {
        	thePrimaryArrayOfIndicesForSymbols[i] = i;
        }
        
        return thePrimaryArrayOfIndicesForSymbols;
    }
    
    
    /** --------------------------------------------------------------------------------------------------------------
     *  keyPadArrangement provides a string representation of all structured permutations corresponding to the minimum
     *  number of key strokes.
     -------------------------------------------------------------------------------------------------------------- */
    
    public static String keyPadArrangement(int[] frequencies, int[] keyLimits, char[] letters)
    {
    	
    	int theMinimumNumberOfKeyStrokes = minimalKeyStrokes(frequencies, keyLimits);

    	ArrayList<ArrayList<ArrayList<Integer>>>
    		theStructuredPermutationsOfIndicesCorrespondingToTheMinimumNumberOfKeyStrokes =
    		new ArrayList<ArrayList<ArrayList<Integer>>>();
        
        for (Map.Entry<ArrayList<ArrayList<Integer>>, Integer> theKeyValuePair :
        	 theHashMapOfStructuredPermutationsOfIndicesAndNumberOfKeyStrokes.entrySet())
        {
        	if (theKeyValuePair.getValue() == theMinimumNumberOfKeyStrokes)
        	{
        		theStructuredPermutationsOfIndicesCorrespondingToTheMinimumNumberOfKeyStrokes.add(
        			theKeyValuePair.getKey()
        		);
        	}
        }
        
        ArrayList<ArrayList<ArrayList<Character>>> thePermutationsOfLettersCorrespondingToTheMinimumNumberOfKeyStrokes =
        	new ArrayList<ArrayList<ArrayList<Character>>>();
        
        for (ArrayList<ArrayList<Integer>> theStructuredPermutationOfIndicesForSymbols :
        	 theStructuredPermutationsOfIndicesCorrespondingToTheMinimumNumberOfKeyStrokes)
        {
        	ArrayList<ArrayList<Character>> theStructuredPermutationOfLetters = new ArrayList<ArrayList<Character>>();
        	
        	for (int i = 0; i < theStructuredPermutationOfIndicesForSymbols.size(); i++)
        	{
    			ArrayList<Character> theArrayOfLetters = new ArrayList<Character>();
        		
        		for (int j = 0; j < theStructuredPermutationOfIndicesForSymbols.get(i).size(); j++)
        		{        			
        			theArrayOfLetters.add(letters[theStructuredPermutationOfIndicesForSymbols.get(i).get(j)]);
        		}
        		
        		theStructuredPermutationOfLetters.add(theArrayOfLetters);
        	}
        	
        	thePermutationsOfLettersCorrespondingToTheMinimumNumberOfKeyStrokes.add(theStructuredPermutationOfLetters);
        }
        
        String theRepresentationOfTheStructuredPermutationsOfLetters = "";
        
        for (int i = 0; i < thePermutationsOfLettersCorrespondingToTheMinimumNumberOfKeyStrokes.size(); i++)
        {
        	ArrayList<ArrayList<Character>> theStructuredPermutation =
        		thePermutationsOfLettersCorrespondingToTheMinimumNumberOfKeyStrokes.get(i);
	        
        	theRepresentationOfTheStructuredPermutationsOfLetters += "\t";
        	
        	for (ArrayList<Character> theArrayOfCharacters : theStructuredPermutation)
        	{
        		theRepresentationOfTheStructuredPermutationsOfLetters += "[";
        		
        		for (int j = 0; j < theArrayOfCharacters.size(); j++)
        		{
        			theRepresentationOfTheStructuredPermutationsOfLetters += theArrayOfCharacters.get(j);
        		}
        		
        		theRepresentationOfTheStructuredPermutationsOfLetters += "]";
        	}
        	
        	theRepresentationOfTheStructuredPermutationsOfLetters += "\n";
	        
        }
        
        return theRepresentationOfTheStructuredPermutationsOfLetters;
        
    }
    
}
