package converter_utilities;


import database_assembling_utilities.DatabaseAssembler;
import java.io.File;
import java.io.IOException;
import java.sql.SQLException;


public class Main {

	public static void main(String[] args)
		throws NotTwoInputsException, IOException, SQLException {

		InputParsingUtilities.checkNumberOfInputsIn(args);
		
		String nameOfFileContainingJSON = args[0];
		String contentsOfFile = InputParsingUtilities.getContentsOfFile(nameOfFileContainingJSON);
		
		String pathToDatabase = args[1];
		File file = new File(pathToDatabase);
		if (!file.exists()) {
			DatabaseAssembler.createDatabase(pathToDatabase);
		}
		DatabaseAssembler.createTables(pathToDatabase);
		
		DatabaseAssembler.insert(contentsOfFile, pathToDatabase);
		
	}

}
