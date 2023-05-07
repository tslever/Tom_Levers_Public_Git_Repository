package Com.TSL.MatrixEditor;


import java.io.File;
import java.io.IOException;
import java.io.FileReader;
import java.io.BufferedReader;
import java.nio.charset.StandardCharsets;


/** ******************************************************************************************************************
 * AreaFill encapsulates the entry point of this program, which loads a matrix of characters from a file, displays the
 * matrix, and interprets command-line arguments as a row / column coordinate pair. The program then runs a method that
 * determines whether or not the character at the coordinates that method received matches a program-defined character,
 * and, if so, replaces that character with an asterisk. Additionally, if the condition is true, the method then calls
 * itself, with new coordinates corresponding to the matrix cells immediately to the left, to the right, above, and
 * below the cell at the previous coordinates.
 * 
 * @author YINGJIN CUI, Tom Lever
 * @version 1.0
 * since   2020-02
 *
 * Student name: Tom Lever
 * Completion date: 06/05/21
 ***************************************************************************************************************** */

public class AreaFill{
     
    public static void main(String[] args) throws Exception
    { 
        // YOU ARE NOT SUPPOSED TO MAKE ANY CHANGES HERE
        char[][] matrix = buildMatrix (new File (args[0]));
        System.out.println ("Original Matrix:");
        print (matrix);
        
        int row=Integer.parseInt(args[1]);
        int col=Integer.parseInt(args[2]);
        fill(matrix, row, col, matrix[row][col]);
        
        System.out.println("after filling\n");
        print(matrix);
        
    } 

    
    /** -------------------------------------------------------------------------------------------------------------
     * buildMatrix builds a matrix of characters based on the contents of a file. The first line in the file contains
     * the number of rows and a number of columns for the matrix, separated by a space. The rest of the file contains
     * the characters that will fill the matrix completely.
     * 
     * @param file
     * @return
     * @throws AMatrixFileParsingException
     * @throws IOException
     ------------------------------------------------------------------------------------------------------------ */
      
    public static char[][] buildMatrix(File file) throws AMatrixFileParsingException, IOException
    { 
      
    // *** Student task #1 ***  

    /* 
    Requirements: 
    This method reads data from the file and build and returns a 2D char array. 
    The first line in the data file contains two numbers R and C, followed by R lines 
    and each line contains C characters.

    *** Enter your code below *** 
    */

        FileReader theFileReader = new FileReader(file, StandardCharsets.UTF_8); // throws IOException
        BufferedReader theBufferedReader = new BufferedReader(theFileReader);
        
        
        String theLineOfTheMatrixFileRepresentingTheHeightAndWidthOfTheMatrix =
        	theBufferedReader.readLine(); // throws IOException
        System.out.println(
        	"Original data in file\n\n" + theLineOfTheMatrixFileRepresentingTheHeightAndWidthOfTheMatrix
        );
        
        String[] theHeightAndWidthOfTheMatrixAsAnArrayOfStrings =
            theLineOfTheMatrixFileRepresentingTheHeightAndWidthOfTheMatrix.split(" "); // throws PatternSyntaxException
        
        if (theHeightAndWidthOfTheMatrixAsAnArrayOfStrings.length != 2)
        {
            throw new AMatrixFileParsingException(
         	    "Exception: The matrix file does not have a first line containing a matrix height and width."
         	);
        }
         
        String theHeightOfTheMatrixAsAString = theHeightAndWidthOfTheMatrixAsAnArrayOfStrings[0];
        String theWidthOfTheMatrixAsAString = theHeightAndWidthOfTheMatrixAsAnArrayOfStrings[1];
        
        
        int theHeightOfTheMatrix = Integer.parseInt(theHeightOfTheMatrixAsAString); // throws NumberFormatException
        int theWidthOfTheMatrix = Integer.parseInt(theWidthOfTheMatrixAsAString); // throws NumberFormatException
        
        if (theHeightOfTheMatrix < 0 || theWidthOfTheMatrix < 0)
        {
            throw new AMatrixFileParsingException(
         		"Exception: Either the height of the matrix or the width of the matrix is negative."
         	);
        }
 		
        
     	char[][] theMatrixOfCharacters = new char[theHeightOfTheMatrix][theWidthOfTheMatrix];
     	
     	String theLineRepresentingAMatrixRow;
     	int theNumberOfLinesRepresentingMatrixRows = 0;
     	int theIndexOfThePresentMatrixRow = 0;
     	
     	while ((theLineRepresentingAMatrixRow = theBufferedReader.readLine()) != null) // throws IOException
     	{
     		theNumberOfLinesRepresentingMatrixRows++;
     		if (theNumberOfLinesRepresentingMatrixRows > theHeightOfTheMatrix)
     		{
     			throw new AMatrixFileParsingException(
     				"Exception: The number of lines representing matrix rows is greater than the height of the " +
     				"matrix " + theHeightOfTheMatrix + "."
     			);
     		}
     		
     		System.out.println(theLineRepresentingAMatrixRow);
     		
     		if (theLineRepresentingAMatrixRow.length() != theWidthOfTheMatrix)
     		{
     			throw new AMatrixFileParsingException(
     				"Exception: The length of the line representing a matrix row \"" + theLineRepresentingAMatrixRow +
     				"\" is not equal to the width of the matrix " + theHeightOfTheMatrix + "."
     			);
     		}
     		
     		for (int i = 0; i < theLineRepresentingAMatrixRow.length(); i++)
     		{
     			theMatrixOfCharacters[theIndexOfThePresentMatrixRow][i] =
     				theLineRepresentingAMatrixRow.charAt(i); // throws IndexOutOfBoundsException
     		}
     		theIndexOfThePresentMatrixRow++;
     	}
     	
     	theBufferedReader.close(); // throws IOException
     	
     	if (theNumberOfLinesRepresentingMatrixRows < theHeightOfTheMatrix)
     	{
     		throw new AMatrixFileParsingException(
     			"Exception: The number of lines representing matrix rows is less than the height of the matrix " +
     			theHeightOfTheMatrix + "."
     		);
     	}

     	System.out.println("\n");
     	
        return theMatrixOfCharacters;

    }
    
    
    /** ---------------------------------------------------------------------------------------------------------------
     * If the character at input row and column coordinates matches the input character, fill fills the cell at the row
     * and column coordinates with an asterisk. fill recurs up, down, left, and right.
     * 
     * @param grid
     * @param row
     * @param col
     * @param ch
     --------------------------------------------------------------------------------------------------------------- */
    
    public static void fill(char[][] grid, int row, int col, char ch) {
    // *** Student task #2 ***  

    /* 
    Requirements: 
    This is a recursive method. It fills the cell at [row, col] 
    with * if matrix[row][col] is ch, then recurs up, down, left, and right, 
    (but NOT diagonally) and replaces similar characters with ‘*’.
    
    *** Enter your code below ***       
    */
    
    	if (grid[row][col] == ch)
    	// This if statement is specified in the in-code requirements for and in the detailed requirements in the
    	// problem prompt for this method, but not in the general requirements for this method in the problem prompt.
    	{
	    	grid[row][col] = '*';
	    	
	    	if (col > 0)
	    	{
	    	    fill(grid, row, col - 1, ch);
	    	}
	    	
	    	if (col < grid[0].length - 1)
	    	{
	    	    fill(grid, row, col + 1, ch);
	    	}
	    	
	    	if (row > 0)
	    	{
	    	    fill(grid, row - 1, col, ch);
	    	}
	    	
	    	if (row < grid.length - 1)
	    	{
	    	    fill(grid, row + 1, col, ch);
	    	}
    	
    	}
     
    }

    
    /** ------------------------------------
     * print displays a matrix of characters.
     * 
     * @param area
     ------------------------------------ */
    
    public static void print(char[][] area)
    {
    
    // *** Student task #3 ***  

    /* 
    Requirements: 
    This method prints the matrix in a table format as shown below:
        ....00
        .0..00
        ..0000

    *** Enter your code below *** 
    */

        for (int i = 0; i < area.length; i++)
        {
        	for (int j = 0; j < area[0].length; j++)
         	{
        		System.out.print(area[i][j]);
         	}
         	
         	System.out.println();
        }
        
        System.out.println("\n");
  
    }
     
}