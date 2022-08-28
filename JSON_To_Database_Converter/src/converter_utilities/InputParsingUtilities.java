package converter_utilities;


import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;


public class InputParsingUtilities {

	
	public static void checkNumberOfInputsIn(String[] args) throws NotTwoInputsException {
		
		if (args.length != 2) {
			throw new NotTwoInputsException(
				"You must specify path to file containing JSON and path to database.");
		}
		
	}
	
	
	public static String getContentsOfFile(String filename)
		throws FileNotFoundException, UnsupportedEncodingException, IOException {
		
		// DON'T DO THIS: USES Windows-1252 ENCODING.
		//File file = new File(filename);
		//FileReader fileReader = new FileReader(file); // throws FileNotFoundException
		//BufferedReader bufferedReader = new BufferedReader(fileReader);
		
		FileInputStream fileInputStream = new FileInputStream(filename);
		InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream, "UTF-8");
		// throws UnsupportedEncodingException
		BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
		StringBuilder stringBuilder = new StringBuilder();
		int charAsInt;
		while ((charAsInt = bufferedReader.read()) != -1) { // read throws IOException
			stringBuilder.append((char)charAsInt);
		}
		bufferedReader.close(); // throws IOException
		return stringBuilder.toString();
		
	}
	
}
