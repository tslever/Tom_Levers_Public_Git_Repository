package Com.TSL.MatrixEditor;


import java.io.File;
import java.io.IOException;
import org.junit.jupiter.api.Test;


/** **********************************************************************************************
 * MainTest encapsulates a JUnit test of method buildMatrix that tests for an invalid File object.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/06/21
 ********************************************************************************************** */

public class BuildMatrixTest {

	
	/** --------------------------------------------------------------------------------------------
	 * testMainForInvalidNumberOfInputArguments tests main for an invalid number of input arguments.
	 -------------------------------------------------------------------------------------------- */
	
	@Test
	public void testBuildMatrixForNonExistentFile()
	{
		
		System.out.println("Running testBuildMatrixForNonExistentFile.");
		
		try {
			// Construction of a File object succeeds.
			AreaFill.buildMatrix(new File("filename"));
		}
		
		catch (IOException theIOException) {
			System.out.println(theIOException.getMessage());
		}
		
		catch (AMatrixFileParsingException theMatrixFileParsingException) {
			System.out.println(theMatrixFileParsingException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
}
